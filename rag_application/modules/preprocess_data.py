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

    try:
        # Data Cleaning
        try:
            examples_df = examples_df.dropna().drop_duplicates()
            products_df = products_df.dropna().drop_duplicates()
            sources_df = sources_df.dropna().drop_duplicates()
        except ValueError as ve:
            print(f"ValueError encountered during data cleaning: {ve}")
            return

        # Feature Extraction
        products_df['combined_text'] = products_df['product_title'] + " " + products_df['product_description']

        # Save Processed Data
        examples_df.to_parquet('rag_application/shopping_queries_dataset/processed_examples.parquet')
        products_df.to_parquet('rag_application/shopping_queries_dataset/processed_products.parquet')
        sources_df.to_csv('rag_application/shopping_queries_dataset/processed_sources.csv', index=False)
    except FileNotFoundError:
        print(f"Error: One or more required files were not found. Please check the file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Improve search speed. Note: watch memory implications over time.
    product_mappings = {}
    for _, row in products_df.iterrows():
        product_id = row['product_id']
        product_mappings[product_id] = (row['product_title'], row['product_description'])

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_data()
