import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_data():
    # Load the dataset files
    examples_df = pd.read_parquet('rag_application/shopping_queries_dataset/shopping_queries_dataset_examples.parquet')
    products_df = pd.read_parquet('rag_application/shopping_queries_dataset/shopping_queries_dataset_products.parquet')
    sources_df = pd.read_csv('rag_application/shopping_queries_dataset/shopping_queries_dataset_sources.csv')

    print("Examples DataFrame shape:", examples_df.shape)
    print("Products DataFrame shape:", products_df.shape)
    print("Sources DataFrame shape:", sources_df.shape)

    # Data Cleaning
    examples_df = examples_df.dropna().drop_duplicates()
    products_df = products_df.dropna().drop_duplicates()
    sources_df = sources_df.dropna().drop_duplicates()

    # Feature Extraction
    products_df['combined_text'] = products_df['product_title'] + " " + products_df['product_description']

    # Save Processed Data
    examples_df.to_parquet('rag_application/shopping_queries_dataset/processed_examples.parquet')
    products_df.to_parquet('rag_application/shopping_queries_dataset/processed_products.parquet')
    sources_df.to_csv('rag_application/shopping_queries_dataset/processed_sources.csv', index=False)

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_data()
