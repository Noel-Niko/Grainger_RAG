import re

import pandas as pd
import os
import logging
import pandas as pd
import dask.dataframe as dd
from dask.base import normalize_token
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from fugashi import Tagger
from langdetect import detect_langs
from dask import delayed

nltk.download('stopwords')
nltk.download('punkt')

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
        if isinstance(text, str):
            text = text.lower()
            tokens = nltk.word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [token for token in tokens if token not in stop_words]
            normalized_text = ' '.join(filtered_tokens)
            return normalized_text
        else:
            logging.error("Expected a string while normalizing text.")
            raise ValueError("Expected a string")

    #  @staticmethod
    # def normalize_text(text, stop_words=None):
    #     if isinstance(text, str):
    #         detected_languages = detect_langs(text)
    #         primary_language = detected_languages[0].lang
    #         if primary_language == 'en':
    #             # English normalization
    #             text = text.lower()
    #             tokens = word_tokenize(text)
    #             stop_words = set(stopwords.words('english'))
    #         elif primary_language == 'es':
    #             # Spanish normalization
    #             text = text.lower()
    #             tokens = word_tokenize(text)
    #             stop_words = set(stopwords.words('spanish'))
    #         # Tagger library has missing dependencies in the source
    #         # elif primary_language == 'ja':
    #         #     # Japanese normalization
    #         #     tagger = Tagger("-Oyomi")
    #         #     tokens = [word.surface for word in tagger.parse(text).split()]
    #         #     # TODO: NO package found for japanese stop words
    #         #     stop_words = set()
    #         else:
    #             logging.warning(f"Unsupported language: {primary_language}. No normalization applied.")
    #             return text
    #
    #         filtered_tokens = [token for token in tokens if token not in stop_words]
    #         normalized_text = ' '.join(filtered_tokens)
    #         return normalized_text
    #     else:
    #         logging.error("Expected a string while normalizing text.")
    #         raise ValueError("Expected a string")
    # @staticmethod
    # def normalize_text(text, stop_words=None):
    #     if isinstance(text, str):
    #         logging.debug(f"Processing text: {text[:50]}...")  # Log the first 50 characters of the text
    #         detected_languages = detect_langs(text)
    #         primary_language = detected_languages[0].lang
    #         logging.debug(f"Detected language: {primary_language}")  # Log the detected language
    #
    #         if primary_language == 'en':
    #             # English normalization
    #             text = text.lower()
    #             tokens = word_tokenize(text)
    #             stop_words = set(stopwords.words('english'))
    #         elif primary_language == 'es':
    #             # Spanish normalization
    #             text = text.lower()
    #             tokens = word_tokenize(text)
    #             stop_words = set(stopwords.words('spanish'))
    #         else:
    #             logging.warning(f"Unsupported language: {primary_language}. No normalization applied.")
    #             return text
    #
    #         filtered_tokens = [token for token in tokens if token not in stop_words]
    #         normalized_text = ' '.join(filtered_tokens)
    #         logging.debug(
    #             f"Normalized text: {normalized_text[:50]}...")  # Log the first 50 characters of the normalized text
    #         return normalized_text
    #     else:
    #         logging.error("Expected a string while normalizing text.")
    #         raise ValueError("Expected a string")

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
            # TODO: REDUCING THE SIZE OF THE FILE FOR INTEGRATION TESTING
            self.products_df = self.products_df.dropna().drop_duplicates().sample(frac=0.001)

            self.sources_df = self.sources_df.dropna().drop_duplicates()

            # Feature Extraction
            self.products_df['combined_text'] = (self.products_df['product_title']
                                                 + " " + self.products_df['product_description']
                                                 + " " + self.products_df['product_bullet_point']
                                                 + " " + self.products_df['product_brand'])
            # Normalize combined_text
            # self.products_df['combined_text'] = self.products_df['combined_text'].astype(str)
            # self.products_df['combined_text'] = self.products_df['combined_text'].apply(self.normalize_text)

            # Ensure combined_text is of type string
            self.products_df['combined_text'] = self.products_df['combined_text'].astype(str)

            # Check the first few rows to confirm the dtype is indeed 'object' (which represents strings in pandas)
            print(self.products_df['combined_text'].head())

            # Apply the normalize_text function
            try:
                self.products_df['combined_text'] = self.products_df['combined_text'].apply(DataPreprocessor.normalize_text)
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
                    print(f"The file {file_path} has been deleted.")
                else:
                    print(f"The file {file_path} does not exist.")

                if df_name == 'sources':
                    print(f"Saving file to {file_path}")
                    logging.info(f"Saving file to {file_path}")
                    df.to_csv(file_path, index=False)
                else:
                    print(f"Saving file to {file_path}")
                    logging.info(f"Saving file to  {file_path}")
                    df.to_parquet(file_path)

            logging.info("Data preprocessing completed successfully.")

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
