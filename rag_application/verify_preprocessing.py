import unittest
import pandas as pd
from rag_application.modules.preprocess_data import DataPreprocessor


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = DataPreprocessor()

        self.examples_file = 'test_shopping_queries_dataset/shopping_queries_dataset_examples.parquet'
        self.products_file = 'test_shopping_queries_dataset/shopping_queries_dataset_products.parquet'
        self.sources_file = 'test_shopping_queries_dataset/shopping_queries_dataset_sources.csv'

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
        df_examples_before = pd.read_parquet(self.examples_file)
        df_products_before = pd.read_parquet(self.products_file)
        df_sources_before = pd.read_csv(self.sources_file)

        """Test data cleaning process."""
        self.preprocessor.preprocess_data()

        df_examples_after = pd.read_parquet('test_shopping_queries_dataset/processed_examples.parquet')
        df_products_after = pd.read_parquet('test_shopping_queries_dataset/processed_products.parquet')
        df_sources_after = pd.read_csv('test_shopping_queries_dataset/processed_sources.csv')

        # Assertions to check if data cleaning worked as expected
        self.assertEqual(df_examples_after.shape, df_examples_before.shape)
        self.assertEqual(df_sources_after.shape, df_sources_before.shape)

        self.assertNotEqual(df_products_after.shape, df_products_before.shape)

        self.assertTrue((df_examples_after.isnull().sum().sum() == 0))
        self.assertTrue((df_products_after.isnull().sum().sum() == 0))
        self.assertTrue((df_sources_after.isnull().sum().sum() == 0))

    def test_feature_extraction(self):
        df_products_before = pd.read_parquet(self.products_file)
        """Test feature extraction process."""
        self.preprocessor.preprocess_data()

        df_product_after = pd.read_parquet('test_shopping_queries_dataset/processed_products.parquet')

        # Assertions to check if feature extraction worked as expected
        self.assertIn('combined_text', df_product_after.columns)
        self.assertNotEqual(len(df_products_before['product_title'].unique()),
                            len(df_product_after['combined_text'].unique()))
        print(df_product_after.columns)

if __name__ == "__main__":
    unittest.main()
