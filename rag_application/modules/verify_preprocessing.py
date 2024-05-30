import time
import unittest
import pandas as pd
from rag_application.modules.preprocess_data import DataPreprocessor
import os


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = DataPreprocessor()
        # Dynamically determine the base directory and construct the full path to each file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'shopping_queries_dataset')
        self.examples_file = os.path.join(data_dir, 'shopping_queries_dataset_examples.parquet')
        self.products_file = os.path.join(data_dir, 'shopping_queries_dataset_products.parquet')
        self.sources_file = os.path.join(data_dir, 'shopping_queries_dataset_sources.csv')

        # Debugging: Print current working directory
        print(f"Current working directory: {os.getcwd()}")

    def test_data_loading(self):
        """Test loading of data before preprocessing."""
        self.preprocessor.preprocess_data()
        while not self.preprocessor.is_preprocessing_complete():
            time.sleep(1)
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
        while not self.preprocessor.is_preprocessing_complete():
            time.sleep(1)

        # Dynamically determine the base directory and construct the full path to each file
        output_dir = 'shopping_queries_dataset'
        df_examples_after = os.path.join(output_dir, 'processed_examples.parquet')
        df_products_after = os.path.join(output_dir, 'processed_products.parquet')
        df_sources_after = os.path.join(output_dir, 'processed_sources.csv')

        df_examples_after = pd.read_parquet(df_examples_after)
        df_products_after = pd.read_parquet(df_products_after)
        df_sources_after = pd.read_csv(df_sources_after)

        # Assertions to check if data cleaning worked and numeric_index added as expected
        assert 'numeric_index' in df_products_after.columns, "numeric_index column does not exist."
        assert isinstance(int(df_products_after['numeric_index'].iloc[2]),
                          int), "numeric_index column is not of integer type."

        self.assertTrue(df_examples_after[df_examples_after['product_id'].eq('')].shape[0] == 0)
        self.assertTrue(df_products_after[df_examples_after['product_id'].eq('')].shape[0] == 0)
        self.assertTrue(df_sources_after[df_examples_after['query_id'].eq('')].shape[0] == 0)

    def test_feature_extraction(self):
        """Test feature extraction process."""
        # TODO: implement feature extraction in preprocessing and test here.
        self.preprocessor.preprocess_data()
        while not self.preprocessor.is_preprocessing_complete():
            time.sleep(1)
        # Dynamically determine the base directory and construct the full path to each file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'shopping_queries_dataset')
        df_products_after = os.path.join(data_dir, 'processed_products.parquet')

        df_product_after = pd.read_parquet(df_products_after)

        # Assertions to check content
        print(df_product_after.columns)
        self.assertIn('product_title', df_product_after.columns)
        self.assertIn('numeric_index', df_product_after.columns)
        print(df_product_after.columns)

if __name__ == "__main__":
    unittest.main()
