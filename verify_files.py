import pandas as pd

examples_file = 'rag_application/shopping_queries_dataset/shopping_queries_dataset_examples.parquet'
# products_file = 'rag_application/shopping_queries_dataset/shopping_queries_dataset_products.parquet'
sources_file = 'rag_application/shopping_queries_dataset/shopping_queries_dataset_sources.csv'

try:
    df_examples = pd.read_parquet(examples_file)
    # df_products = pd.read_parquet(products_file)
    df_sources = pd.read_csv(sources_file)
    print("Files are correctly formatted and readable.")
except Exception as e:
    print(f"Error reading files: {e}")
