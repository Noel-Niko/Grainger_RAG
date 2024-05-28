import pandas as pd
import numpy as np
from pyChatGPT import ChatGPT
from transformers import AutoTokenizer, AutoModel
import faiss
from typing import Tuple, List

from rag_application.modules.initialize_llm_model import LLMInteraction

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VectorIndex:
    """
    VectorIndex for creating and querying a FAISS index using BERT embeddings.
    Uses batch processing to avoid loading the entire dataset into memory at once.
    Ensures that the FAISS index is created once and reused throughout the application life of the container.
    """
    _instance = None
    _index = None
    _products_df = None

    # @classmethod
    # def getInstance(cls):
    #     """Static access method to get the singleton instance."""
    #     if cls._instance is None:
    #         cls._instance = cls()
    #     return cls._instance

    @classmethod
    def getInstance(cls, **kwargs):
        """Static access method to get the singleton instance, enforcing required arguments."""
        if cls._instance is None:
            # Safely retrieve 'products_file' from kwargs, providing a default value if not found
            products_file = kwargs.get('products_file', '')

            # Check if 'products_file' is a string
            if not isinstance(products_file, str):
                raise TypeError("'products_file' argument must be a string")

            # Proceed with instantiation if 'products_file' is valid
            cls._instance = cls(products_file=products_file)

        return cls._instance

    def __init__(self, products_file=None, nlist=100, m=16, batch_size=32):
        self.llm = None
        self.products_file = products_file
        self.nlist = nlist
        self.m = m
        self.batch_size = batch_size
        self.embeddings_dict = {}
        print("VectorIndex instance created.")
        logging.info("VectorIndex instance created.")

    # def __init__(self, products_file: str, nlist: int = 100, m: int = 16, batch_size: int = 32):
    #     self.llm = None
    #     self.products_file = products_file
    #     self.nlist = nlist
    #     self.m = m
    #     self.batch_size = batch_size
    #     self.embeddings_dict = {}
    #     print("VectorIndex instance created.")

    def load_processed_products(self):
        """Loads the processed products data with error handling."""
        print("Loading preprocessed products.")
        try:
            self.products_df = pd.read_parquet(self.products_file)
            logging.info("Completed loading preprocessed products.")
            print("Completed loading preprocessed products.")
        except FileNotFoundError:
            logging.error(f"File {self.products_file} not found.")
            print(f"File {self.products_file} not found.")
        except Exception as e:
            logging.error(f"An error occurred while loading the file: {e}")
            print(f"An error occurred while loading the file: {e}")

    def encode_text_to_embedding(self, texts: List[str]):
        """Encodes a list of texts to BERT embeddings with error handling."""
        logging.info("Encoding text to embedding.")
        print("Encoding text to embedding.")
        embeddings = []
        logging.info("Tokenizing...")
        print("Tokenizing...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        logging.info("Creating model via AutoModel.from_pretrained('bert-base-uncased')...")
        print("Creating model via AutoModel.from_pretrained('bert-base-uncased')...")
        model = AutoModel.from_pretrained('bert-base-uncased')

        total_batches = (len(texts) + self.batch_size - 1)
        for batch in range(0, len(texts), self.batch_size):
            print(f"Encoding text batch {batch} of {total_batches}.")
            batch_texts = texts[batch:batch + self.batch_size]
            if not batch_texts:
                continue
            try:
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
                embeddings.extend(batch_embeddings)
                print(f"Finished encoding text batch {batch} of {total_batches}.")
            except Exception as e:
                print(f"An error occurred during embedding extraction: {e}")
        logging.info("Returning embeddings.")
        print("Returning embeddings.")
        return np.array(embeddings)

    def create_faiss_index(self):
        """Creates an FAISS IVF-HC index for efficient vector similarity search with batch processing."""
        logging.info("Creating an FAISS IVF-HC index for efficient vector similarity search with batch processing.")
        print("Creating an FAISS IVF-HC index for efficient vector similarity search with batch processing.")
        combined_texts = self.products_df['combined_text'].tolist()
        embeddings = self.encode_text_to_embedding(combined_texts)
        expected_dim = 768  # Example: BERT base model has 768 dimensions
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions. Expected {expected_dim}, got {embeddings.shape[1]}")

        d = embeddings.shape[1]  # Dimensionality of the embeddings

        # Create the quantizer and index
        logging.info("Creating quantizer")
        print("Creating quantizer")
        quantizer = faiss.IndexFlatL2(d)
        self._index = faiss.IndexIVFFlat(quantizer, d, self.nlist)

        # Ensure embeddings is a numpy array
        embeddings_np = np.array(embeddings)

        # Train the index and add embeddings
        logging.info("Training...")
        print("Training...")
        self._index.train(embeddings_np)
        logging.info("Embedding...")
        print("Embedding...")
        self._index.add(embeddings_np)

    def search_index(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches for the k nearest neighbors of the query.

        :param query: The text query to search for.
        :param k: Number of nearest neighbors to return.
        :return: A tuple containing distances and indices of the nearest neighbors.
        """
        logging.info("Searching for the k nearest neighbors of the query.")
        print("Searching for the k nearest neighbors of the query.")
        # Check if the index is initialized
        if self._index is None:
            logging.error("Index is not initialized.")
            raise RuntimeError("Index is not initialized.")
        # Check if the query is empty and raise a ValueError if it is
        if not query.strip():
            logging.error("Query string cannot be empty.")
            raise ValueError("Query string cannot be empty.")
        # Check if k is an integer > 0
        if not isinstance(k, int) or k <= 0:
            logging.error("Search radius must be an integer greater than 0.")
            raise TypeError("Search radius must be an integer greater than 0.")

        # Convert the query string into a numerical vector
        query_vector = self.encode_text_to_embedding([query])

        # Ensure the query vector has the correct shape for FAISS search
        query_vector = np.expand_dims(query_vector, axis=0)

        # Search the FAISS index
        logging.info("Searching the FAISS index.")
        print("Searching the FAISS index.")
        distance, result_index = self._index.search(query_vector[0], k)

        try:
            logging.info(f"Returning distance: {str(distance.tolist()[0])}")
            print(f"Returning distance: {str(distance.tolist()[0])}")
        except Exception as e:
            logging.error("Error: Unable to convert distance data to string")
            print("Error: Unable to convert distance data to string")

        try:
            logging.info(f"Returning result_index: {str(result_index.tolist()[0])}")
            print(f"Returning result_index: {str(result_index.tolist()[0])}")
        except Exception as e:
            logging.error("Error: Unable to convert results data to string")
            print("Error: Unable to convert results data to string")

        return distance, result_index

    def find_changed_products(self, old_descriptions, new_descriptions):
        """
        Identifies products whose descriptions have changed.

        Parameters:
        - old_descriptions (dict): Mapping of product IDs to their old descriptions.
        - new_descriptions (dict): Mapping of product IDs to their new descriptions.

        Returns:
        - set: Set of product IDs whose descriptions have changed.
        """
        logging.info("Searching for changed products.")
        print("Searching for changed products.")
        changed_products = set()
        for product_id, new_desc in new_descriptions.items():
            old_desc = old_descriptions.get(product_id)
            if old_desc != new_desc:
                changed_products.add(product_id)
        try:
            logging.info(f"Returning changed_products: {str(changed_products)}")
            print(f"Returning changed_products: {str(changed_products)}")
        except Exception as e:
            logging.error("Error: Unable to convert changed_products data to string")
            print("Error: Unable to convert changed_products data to string")

        return changed_products

    def update_product_descriptions(self, updates):
        """
        Batch updates the descriptions of multiple products and regenerates their embeddings.

        Parameters:
        - updates (dict): Mapping of product IDs to their new descriptions.

        Raises:
        - KeyError: If a product ID in updates is not found in the DataFrame.
        """
        logging.info("Making batch updates for the descriptions of multiple products and regenerating their embeddings.")
        print("Making batch updates for the descriptions of multiple products and regenerating their embeddings.")
        # Find products whose descriptions have changed
        changed_products = self.find_changed_products(
            {pid: row['product_description'] for pid, row in self.products_df.iterrows()}, updates)

        # Update descriptions in the DataFrame
        for product_id, new_description in updates.items():
            if product_id not in self.products_df['product_id'].values:
                raise KeyError(f"Product ID {product_id} not found in the DataFrame.")
            product_index = self.products_df[self.products_df['product_id'] == product_id].index[0]
            self.products_df.at[product_index, 'product_description'] = new_description
            self.products_df.at[
                product_index, 'combined_text'] = f"{self.products_df.at[product_index, 'product_title']} {new_description}"

        # Regenerate embeddings only for changed products
        try:
            logging.info(f"Changed products list: {str(list(changed_products))}")
            print(f"Changed products list: {str(list(changed_products))}")
        except Exception as e:
            logging.error("Error: Unable to convert changed products list to string")
            print("Error: Unable to convert changed products list to string")
        if changed_products:
            self.update_embeddings_for_changed_products(list(changed_products))

    def update_embeddings_for_changed_products(self, changed_product_ids: List[str]):
        """Re-encodes and re-adds embeddings for products whose descriptions were changed."""
        logging.info("Re-encoding and re-adding embeddings for products whose descriptions were changed.")
        print("Re-encoding and re-adding embeddings for products whose descriptions were changed.")
        for product_id in changed_product_ids:
            try:
                print(f"Product: {product_id}...")
                product_index = self.products_df[self.products_df['product_id'] == product_id].index[0]
                combined_text = f"{self.products_df.at[product_index, 'product_title']} {self.products_df.at[product_index, 'product_description']}"
                new_embedding = self.encode_text_to_embedding([combined_text])[0]
                self._index.add_with_ids(new_embedding.reshape(1, -1), np.array([product_id]))
                self.embeddings_dict[product_id] = new_embedding
            except Exception as e:
                raise RuntimeError(f"Error updating embeddings for product ID {product_id}: {e}")
        logging.error("Completed embedding updates.")
        print("Completed embedding updates.")

    def remove_product_by_id(self, product_id: str):
        """Removes a product by ID from the index and the underlying data store."""
        logging.info(f"Removing a product by {product_id} from the index and the underlying data store.")
        print(f"Removing a product by {product_id} from the index and the underlying data store.")
        if product_id not in self.products_df['product_id'].values:
            raise RuntimeError("product_id not found.")

        product_index = self.products_df[self.products_df['product_id'] == product_id].index[0]
        self.products_df.drop(product_index, inplace=True)
        self.create_faiss_index()  # Re-create the index after removing the product
        logging.info(f"Completed removing a product by {product_id} from the index and the underlying data store.")
        print(f"Completed removing a product by {product_id} from the index and the underlying data store.")

    def get_all_product_ids(self):
        """Returns all unique product IDs from the products_df DataFrame."""
        logging.info("Returning all unique product IDs from the products_df DataFrame...")
        print("Returning all unique product IDs from the products_df DataFrame...")
        return self.products_df['product_id'].unique().tolist()

    def get_embedding(self, product_id):
        """Fetches the embedding for a given product ID."""
        logging.info(f"Fetching the embedding for {product_id}.")
        print(f"Fetching the embedding for {product_id}.")
        embedding = self.embeddings_dict.get(product_id)
        if embedding is None:
            logging.error("Embedding not found")
            print("Embedding not found")
            raise RuntimeError("Embedding not found")
        logging.info("Returning embedding")
        print("Returning embedding")
        return embedding

    def get_first_10_vectors(self):
        """Returns the first 10 vectors in the index dataframe. Used for testing."""
        return self.products_df.head(10)

    def refine_query_with_chatgpt(self, query: str) -> Tuple[str, ChatGPT]:
        logging.info("Refining query with ChatGPT")
        self.llm = LLMInteraction.initialize_llm_model(api_token=self.langchainApiKey)
        prompt_wrapper = ('I received the following message and want a short list of key words to use to search my '
                          'FAISS Index for relevant products and descriptions.')
        logging.info(f"Sending {prompt_wrapper} - {query}")
        refined_query = LLMInteraction.llm_interaction(f"{prompt_wrapper} - {query}", self.llm)
        logging.info(f"Returning: {refined_query}")
        return refined_query, self.llm

    def search_and_generate_response(self, refined_query: str, llm, k: int = 5) -> str:
        _, relevant_product_indices = self.search_index(refined_query, k=k)
        product_info = ", ".join(
            [f"ID: {pid}, "
             f"Name: {self.products_df.loc[pid, 'product_title']}, "
             f"Description: {self.products_df.loc[pid, 'product_description']},"
             f"Key Facts: {self.products_df.loc[pid, 'product_bullet_point']},"
             f"Brand: {self.products_df.loc[pid, 'product_brand']},"
             f"Color: {self.products_df.loc[pid, 'product_color']},"
             f"Location: {self.products_df.loc[pid, 'product_locale']},"
             for pid in relevant_product_indices]
        )
        logging.info(f"From search_and_generate_response returning: {product_info}")
        return product_info


if __name__ == "__main__":
    products_file = 'shopping_queries_dataset/processed_products.parquet'
    vector_index = VectorIndex(products_file, batch_size=32)
    vector_index.load_processed_products()
    vector_index.create_faiss_index()
    logging.info("FAISS index created successfully.")
    print("FAISS index created successfully.")
