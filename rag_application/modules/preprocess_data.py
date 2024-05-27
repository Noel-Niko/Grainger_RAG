import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os


class DataPreprocessor:
    def __init__(self):
        self.examples_df = None
        self.products_df = None
        self.sources_df = None

    def preprocess_data(self):
        # Load the dataset files
        self.examples_df = pd.read_parquet('shopping_queries_dataset/shopping_queries_dataset_examples.parquet')
        self.products_df = pd.read_parquet('shopping_queries_dataset/shopping_queries_dataset_products.parquet')
        self.sources_df = pd.read_csv('shopping_queries_dataset/shopping_queries_dataset_sources.csv')

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
            self.products_df['combined_text'] = self.products_df['product_title'] + " " + self.products_df[
                'product_description']

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
                    df.to_csv(file_path, index=False)
                else:
                    df.to_parquet(file_path)

        except FileNotFoundError:
            print(f"Error: One or more required files were not found. Please check the file paths.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return

        # Improve search speed. Note: watch memory implications over time.
        product_mappings = {}
        for _, row in self.products_df.iterrows():
            product_id = row['product_id']
            product_mappings[product_id] = (row['product_title'], row['product_description'])

        print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_data()
