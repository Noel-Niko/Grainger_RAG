import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreprocessor:
    def __init__(self):
        self.examples_df = None
        self.products_df = None
        self.sources_df = None
        self.product_id_to_index = {}
        self.index_to_product_id = {}
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
            self.examples_df = self.examples_df.drop_duplicates()
            # TODO: REDUCING THE SIZE OF THE FILE FOR INTEGRATION TESTING -->> .sample(frac=0.001)
            self.products_df = self.products_df.drop_duplicates().sample(frac=0.001)
            self.sources_df = self.sources_df.drop_duplicates()

            # Ensure product_id is not null or empty
            if self.products_df is None or self.products_df.empty:
                logging.warning("Products DataFrame is None or empty. Skipping further processing.")
                return

            # Ensure product_id is not null or empty and remove empty and duplicated rows
            temp_df = self.products_df.dropna(how='all').drop_duplicates()
            temp_df = temp_df[temp_df['product_id'].notnull()]

            # Create mappings between product IDs and numeric indices because FAISS requires index values of type int.
            logging.info("Creating numerical index column...")
            print("Creating numerical index column...")

            if 'numeric_index' not in temp_df.columns:
                temp_df['numeric_index'] = None
            unique_ids = {}
            for idx, pid in enumerate(temp_df['product_id']):
                if pid not in unique_ids:
                    unique_ids[pid] = idx

            temp_df['numeric_index'] = temp_df['product_id'].map(unique_ids).astype(int)

            # Feature Extraction
            temp_df['combined_text'] = temp_df['product_title'] + " " + temp_df['product_description']

            self.products_df = temp_df

            logging.info("Completed creating numerical index column...")
            print("Completed creating numerical index column...")

            output_dir = 'shopping_queries_dataset'
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
