import unittest
import pandas as pd
from rag_application.modules.preprocess_data import preprocess_data

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.examples_file = 'rag_application/shopping_queries_dataset/shopping_queries_dataset_examples.parquet'
        self.products_file = 'rag_application/shopping_queries_dataset/shopping_queries_dataset_products.parquet'
        self.sources_file = 'rag_application/shopping_queries_dataset/shopping_queries_dataset_sources.csv'

    def test_preprocess_data(self):
        # Load data before preprocessing
        examples_file = 'rag_application/shopping_queries_dataset/shopping_queries_dataset_examples.parquet'
        product_file = 'rag_application/shopping_queries_dataset/shopping_queries_dataset_products.parquet'
        sources_file = 'rag_application/shopping_queries_dataset/shopping_queries_dataset_sources.csv'

        df_examples_before = pd.read_parquet(examples_file)
        df_products_before = pd.read_parquet(product_file)
        df_sources_before = pd.read_csv(sources_file)

        print("Data samples before preprocessing:")
        print("Examples dataframe (first 10 rows):")
        print(df_examples_before.head(10))
        print("\nExamples dataframe columns:")
        print(df_examples_before.columns)

        print("****************Data products before preprocessing:")
        print("Products dataframe (first 10 rows):")
        print(df_products_before.head(10))
        print("\nProducts dataframe columns:")
        print(df_products_before.columns)

        print("\nSources dataframe (first 10 rows):")
        print(df_sources_before.head(10))
        print("\nSources dataframe columns:")
        print(df_sources_before.columns)

        # Perform preprocessing
        try:
            preprocess_data()
            print("Data preprocessing completed successfully.")
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            self.fail("Failed to preprocess data")

        # Load data after preprocessing
        df_examples_after = pd.read_parquet('rag_application/shopping_queries_dataset/processed_examples.parquet')
        df_product_after = pd.read_parquet('rag_application/shopping_queries_dataset/processed_products.parquet')
        df_sources_after = pd.read_csv('rag_application/shopping_queries_dataset/processed_sources.csv')

        # Assertions to check if preprocessing changed or maintained the data as expected
        self.assertEqual(df_examples_after.shape, df_examples_before.shape)
        self.assertEqual(df_sources_after.shape, df_sources_before.shape)

        self.assertNotEqual(df_product_after.shape, df_products_before.shape)


        # Verify that dropna and drop_duplicates
        self.assertTrue((df_examples_after.isnull().sum().sum() == 0))
        self.assertTrue((df_product_after.isnull().sum().sum() == 0))
        self.assertTrue((df_sources_after.isnull().sum().sum() == 0))

        # Verify presence of 'combined_text' column in products_df
        self.assertIn('combined_text', df_product_after.columns)

        # Verify count of unique values
        print("\nNumber of Unique Product Titles", len(df_products_before['product_title'].unique()))
        print("\nNumber of Unique Combined Product Text", len(df_product_after['combined_text'].unique()))
        self.assertNotEqual(len(df_products_before['product_title'].unique()),
                            len(df_product_after['combined_text'].unique()))

        print("\nData samples after preprocessing:")
        print("Examples dataframe (first 10 rows):")
        print(df_examples_after.head(10))
        print("\nExamples dataframe columns:")
        print(df_examples_after.columns)

        print("\nData samples after preprocessing:")
        print("********Products dataframe (first 10 rows):")
        print(df_product_after.head(10))
        print("\n*******Products dataframe columns:")
        print(df_product_after.columns)

        print("\nSources dataframe (first 10 rows):")
        print(df_sources_after.head(10))
        print("\nSources dataframe columns:")
        print(df_sources_after.columns)

if __name__ == "__main__":
    unittest.main()
