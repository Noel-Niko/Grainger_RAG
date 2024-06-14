import re
import numpy as np
import os
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import SnowballStemmer
from many_stop_words import get_stop_words
import contractions

download('stopwords')
download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreprocessor:
    def __init__(self):
        self.examples_df = None
        self.products_df = None
        self.sources_df = None
        self.preprocessing_complete = False

    def normalize_text(self, text):
        logging.info("Normalizing text")
        print("Normalizing text")
        if isinstance(text, str):
            text = text.lower()
            # Removes HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Removes newline characters
            text = text.replace('\n', ' ').replace('\r', '')
            # Remove emojis
            text = re.sub(r'[^\w\s]', '', text)
            # Expand contractions (e.g., "can't" -> "cannot")
            text = contractions.fix(text)
            tokens = nltk.word_tokenize(text)
            english_stop_words = set(stopwords.words('english'))
            spanish_stop_words = set(stopwords.words('spanish'))
            japanese_stop_words = set(get_stop_words('ja'))
            # Combine English, Spanish, and Japanese stopwords
            combined_stop_words = english_stop_words.union(spanish_stop_words).union(japanese_stop_words)
            filtered_tokens = [token for token in tokens if token not in combined_stop_words]

            # Stemming: Reduce words to their root form
            stemmer = SnowballStemmer('english')
            stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

            normalized_text = ' '.join(stemmed_tokens)
            return normalized_text
        else:
            logging.error("Expected a string while normalizing text.")
            print("Expected a string while normalizing text.")
            # Returning an empty string for non-string inputs instead of raising an error
            # to allow processing of remaining data
            return ""

    def normalize_text_batch(self, batch_series):
        logging.info("Normalizing text")
        normalized_texts = []
        for text in batch_series:
            try:
                normalized_texts.append(self.normalize_text(text))
            except Exception as e:
                logging.error(f"Exception occurred during normalization of text '{text}': {e}")
                normalized_texts.append("")
        return pd.Series(normalized_texts)

    def preprocess_data(self):
        logging.info("Starting data preprocessing...")
        # Dynamically determine the base directory and construct the full path to each file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'shopping_queries_dataset')
        examples_file = os.path.join(data_dir, 'shopping_queries_dataset_examples.parquet')
        products_file = os.path.join(data_dir, 'shopping_queries_dataset_products.parquet')
        sources_file = os.path.join(data_dir, 'shopping_queries_dataset_sources.csv')

        # Load the dataset files
        self.examples_df = pd.read_parquet(examples_file)
        self.products_df = pd.read_parquet(products_file)
        self.sources_df = pd.read_csv(sources_file)

        logging.info("Loaded DataFrames shapes:")
        logging.info(f"Examples DataFrame shape: {self.examples_df.shape}")
        logging.info(f"Products DataFrame shape: {self.products_df.shape}")
        logging.info(f"Sources DataFrame shape: {self.sources_df.shape}")
        print("Loaded DataFrames shapes:")
        print("Examples DataFrame shape:", self.examples_df.shape)
        print("Products DataFrame shape:", self.products_df.shape)
        print("Sources DataFrame shape:", self.sources_df.shape)

        try:
            # Data Cleaning
            self.examples_df = self.examples_df.dropna().drop_duplicates()
            # TODO: REDUCING THE SIZE OF THE FILE FOR INTEGRATION TESTING  - .sample(frac=0.1)
            self.products_df = self.products_df.dropna().drop_duplicates() #.sample(frac=0.01)

            self.sources_df = self.sources_df.dropna().drop_duplicates()

            # Feature Extraction
            logging.info("Creating combined_text feature.")
            self.products_df['combined_text'] = (self.products_df['product_title']
                                                 + " " + self.products_df['product_description']
                                                 + " " + self.products_df['product_bullet_point']
                                                 + " " + self.products_df['product_brand'])

            # Resetting the index to ensure continuous indexing
            self.products_df.reset_index(drop=True, inplace=True)

            # Apply the normalize_text function
            try:
                # Normalize combined_text
                logging.info("Normalizing combined_text.")
                print("Normalizing combined_text.")
                batch_size = 1000
                num_batches = int(np.ceil(len(self.products_df['combined_text']) / batch_size))
                for i in range(num_batches):
                    logging.info(f"Batch {i + 1}/{num_batches}")
                    print(f"Batch {i + 1}/{num_batches}")
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    # Using iloc for positional indexing to avoid KeyError
                    batch = self.products_df.iloc[start_idx:end_idx]['combined_text'].astype(str)
                    # Assigning the normalized text back to the DataFrame using iloc for consistency
                    self.products_df.iloc[start_idx:end_idx,
                    self.products_df.columns.get_loc('combined_text')] = self.normalize_text_batch(batch).values

                # Check the first few rows to confirm the dtype is an 'object' (which represents strings in pandas)
                logging.info(self.products_df['combined_text'].head())
                print(self.products_df['combined_text'].head())
            except Exception as e:
                logging.error(f"Exception occurred during normalization: {e}")

            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir, 'shopping_queries_dataset')
            output_files = {
                'examples': 'processed_examples.parquet',
                'products': 'processed_products.parquet',
                'sources': 'processed_sources.csv'
            }

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for df_name, file_name in output_files.items():
                df = getattr(self, f'{df_name}_df')
                file_path = os.path.join(output_dir, file_name)

                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"The file {file_path} has been deleted.")
                    print(f"The file {file_path} has been deleted.")
                else:
                    logging.info(f"The file {file_path} does not exist.")
                    print(f"The file {file_path} does not exist.")

                if df_name == 'sources':
                    print(f"Saving file to {file_path}")
                    logging.info(f"Saving file to {file_path}")
                    df.to_csv(file_path, index=False)
                else:
                    print(f"Saving file to {file_path}")
                    logging.info(f"Saving file to  {file_path}")
                    df.to_parquet(file_path)

            logging.info("Data preprocessing files saved.")

        except FileNotFoundError:
            logging.error(f"Error: One or more required files were not found. Please check the file paths.")
        except PermissionError:
            logging.error(f"Error: Permission denied when accessing a file or directory.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return

        # Improves search speed. Note: watch memory implications over time.
        product_mappings = {}
        for _, row in self.products_df.iterrows():
            product_id = row['product_id']
            product_mappings[product_id] = (row['product_title'], row['product_description'])
        logging.info("Data preprocessing completed successfully.")
        print("Data preprocessing completed successfully.")
        self.preprocessing_complete = True

    def is_preprocessing_complete(self):
        return self.preprocessing_complete


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_data()
