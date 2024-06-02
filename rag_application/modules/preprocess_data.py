import os
import logging
import pandas as pd
import dask.dataframe as dd
from dask.base import normalize_token
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fugashi import Tagger
from langdetect import detect_langs
from dask import delayed


nltk.download('stopwords')
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def normalize_text(text, stop_words=None):
    if isinstance(text, str):
        detected_languages = detect_langs(text)
        primary_language = detected_languages[0].lang
        if primary_language == 'en':
            # English normalization
            text = text.lower()
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
        elif primary_language == 'es':
            # Spanish normalization
            text = text.lower()
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('spanish'))
        elif primary_language == 'ja':
            # Japanese normalization
            tagger = Tagger("-Oyomi")
            tokens = [word.surface for word in tagger.parse(text).split()]
            # TODO: NO package found for japanese stop words
            stop_words = set()
        else:
            logging.warning(f"Unsupported language: {primary_language}. No normalization applied.")
            return text

        filtered_tokens = [token for token in tokens if token not in stop_words]
        normalized_text = ' '.join(filtered_tokens)
        return normalized_text
    else:
        logging.error("Expected a string while normalizing text.")
        raise ValueError("Expected a string")


class NormalizeTextWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __dask_tokenize__(self):
        return (NormalizeTextWrapper.__name__, normalize_token(self.func))


# Wrap the normalize_text function
normalize_text_wrapper = NormalizeTextWrapper(normalize_text)


class DataPreprocessor:
    def __init__(self):
        self.examples_df = None
        self.products_df = None
        self.sources_df = None
        self.preprocessing_complete = False

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
        self.sources_df = pd.read_csv(sources_file)

        logging.info("Loaded Examples and Sources DataFrames shapes:")
        logging.info(f"Examples DataFrame shape: {self.examples_df.shape}")
        logging.info(f"Sources DataFrame shape: {self.sources_df.shape}")
        print("Loaded Examples and Sources DataFrames shapes:")
        print("Examples DataFrame shape:", self.examples_df.shape)
        print("Sources DataFrame shape:", self.sources_df.shape)

        # Process the products file in chunks using Dask
        logging.info("Processing products file in chunks using Dask")
        products_ddf = dd.read_parquet(products_file)

        try:
            # Data Cleaning and Normalization
            logging.info("Data cleaning and initialization for products.")
            products_ddf = products_ddf.dropna().drop_duplicates()
            products_ddf['combined_text'] = (products_ddf['product_title']
                                             + " " + products_ddf['product_description']
                                             + " " + products_ddf['product_bullet_point']
                                             + " " + products_ddf['product_brand'])

            # Partition the Dask DataFrame
            # Can adjust npartitions as needed
            logging.info("Partitioning the Dask DataFrame.")
            products_ddf = products_ddf.repartition(npartitions=10)

            # Apply normalize_text to each partition separately
            num_partitions = products_ddf.npartitions
            for i in range(num_partitions):
                logging.info(f"Apply normalize_text to {i} of {len(num_partitions)} ")
                partition = products_ddf.get_partition(i).compute()  # Convert to Pandas DataFrame
                partition['combined_text'] = partition['combined_text'].apply(normalize_text)  # Apply normalize_text
                products_ddf = products_ddf.set_partitions(
                    [dd.from_pandas(partition, npartitions=1)])  # Convert back to Dask DataFrame

            # Convert Dask DataFrame back to Pandas DataFrame
            logging.info("Converting Dask data frame back to pandas dataframe.")
            self.products_df = products_ddf.compute()

            logging.info("Processed Products DataFrame shape:")
            logging.info(f"Products DataFrame shape: {self.products_df.shape}")
            print("Processed Products DataFrame shape:", self.products_df.shape)

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
