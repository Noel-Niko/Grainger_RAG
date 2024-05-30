import unittest
import pandas as pd
from rag_application.modules.preprocess_data import DataPreprocessor
import os


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = DataPreprocessor()
        # Dynamically determine the base directory and construct the full path to each file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'test_shopping_queries_dataset')
        self.examples_file = os.path.join(data_dir, 'shopping_queries_dataset_examples.parquet')
        self.products_file = os.path.join(data_dir, 'shopping_queries_dataset_products.parquet')
        self.sources_file = os.path.join(data_dir, 'shopping_queries_dataset_sources.csv')

        # Debugging: Print current working directory
        print(f"Current working directory: {os.getcwd()}")

    def test_data_loading(self):
        """Test loading of data before preprocessing."""
        df_examples_before = pd.read_parquet(self.examples_file)
        df_products_before = pd.read_parquet(self.products_file)
        df_sources_before = pd.read_csv(self.sources_file)

        # Assertions to check if data is loaded correctly
        self.assertIsNotNone(df_examples_before)
        self.assertIsNotNone(df_products_before)
        self.assertIsNotNone(df_sources_before)

    def test_data_cleaning(self):
        """Test data cleaning process."""
        self.preprocessor.preprocess_data()

        # Dynamically determine the base directory and construct the full path to each file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'test_shopping_queries_dataset')
        df_examples_after = os.path.join(data_dir, 'shopping_queries_dataset_examples.parquet')
        df_products_after = os.path.join(data_dir, 'shopping_queries_dataset_products.parquet')
        df_sources_after = os.path.join(data_dir, 'shopping_queries_dataset_sources.csv')

        df_examples_after = pd.read_parquet(df_examples_after)
        df_products_after = pd.read_parquet(df_products_after)
        df_sources_after = pd.read_csv(df_sources_after)

        # Assertions to check if data cleaning worked as expected
        self.assertTrue((df_examples_after.isnull().sum().sum() == 0))
        self.assertTrue((df_products_after.isnull().sum().sum() == 0))
        self.assertTrue((df_sources_after.isnull().sum().sum() == 0))

    def test_feature_extraction(self):
        df_products_before = pd.read_parquet(self.products_file)
        """Test feature extraction process."""
        self.preprocessor.preprocess_data()

        # Dynamically determine the base directory and construct the full path to each file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'test_shopping_queries_dataset')
        df_products_after = os.path.join(data_dir, 'shopping_queries_dataset_products.parquet')

        df_product_after = pd.read_parquet(df_products_after)

        # Assertions to check content
        self.assertIn('product_title', df_product_after.columns)
        print(df_product_after.columns)

if __name__ == "__main__":
    unittest.main()
