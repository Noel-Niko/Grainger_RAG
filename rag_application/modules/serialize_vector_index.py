import logging
import os
import pickle
from rag_application.modules.vector_index_faiss import VectorIndex

logging.basicConfig(level=logging.INFO)
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'shopping_queries_dataset')
products_file = os.path.join(output_dir, 'processed_products.parquet')

logging.info(f"Initializing from {products_file}")
vector_index_instance = VectorIndex.get_instance(products_file=products_file)

# Serialize the VectorIndex instance to a file
logging.info("Completing Pickle Dump")
with open('vector_index.pkl', 'wb') as file:
    pickle.dump(vector_index_instance, file)
    logging.info("Completed Pickle Dump")
