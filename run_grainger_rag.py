import pandas as pd

from rag_application.modules.preprocess_data import preprocess_data


def load_dataset():
    examples_file = 'rag_application/modules/preprocessing/shopping_queries_dataset_examples.parquet'
    products_file = 'rag_application/modules/preprocessing/shopping_queries_dataset_products.parquet'
    sources_file = 'rag_application/modules/preprocessing/shopping_queries_dataset_sources.csv'

    df_examples = pd.read_parquet(examples_file)
    df_products = pd.read_parquet(products_file)
    df_sources = pd.read_csv(sources_file)

    # df_products,
    return df_examples, df_sources


def preprocess_data():
    # df_products,
    df_examples, df_sources = load_dataset()

    # Data Cleaning
    df_examples = df_examples.dropna().drop_duplicates()
    df_products = df_products.dropna().drop_duplicates()
    df_sources = df_sources.dropna().drop_duplicates()

    # Feature Extraction
    df_products['combined_text'] = df_products['product_title'] + " " + df_products['product_description']

    # Save Processed Data
    df_examples.to_parquet('rag_application/test_shopping_queries_dataset/processed_examples.parquet')
    df_products.to_parquet('rag_application/test_shopping_queries_dataset/processed_products.parquet')
    df_sources.to_csv('rag_application/test_shopping_queries_dataset/processed_sources.csv', index=False)

    print("Data preprocessing completed successfully.")


def main():
    # Load dataset
    # df_products,
    df_examples, df_sources = load_dataset()
    preprocess_data()
    print("Examples dataframe:")
    print(df_examples.head())

    print("\nProducts dataframe:")
    print(df_products.head())

    print("\nSources dataframe:")
    print(df_sources.head())


if __name__ == "__main__":
    main()
