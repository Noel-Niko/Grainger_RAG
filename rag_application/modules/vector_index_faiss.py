import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
from typing import Tuple, List


class VectorIndex:
    """
    VectorIndex for creating and querying a FAISS index using BERT embeddings.
    Uses batch processing to avoid loading the entire dataset into memory at once.
    """

    def __init__(self, products_file: str, nlist: int = 100, m: int = 16, batch_size: int = 32):
        self.products_file = products_file
        self.nlist = nlist
        self.m = m
        self.batch_size = batch_size
        self.index = None
        self.products_df = None  # indicates not yet loaded as this will be done in load_processed_products

    def load_processed_products(self):
        """Loads the processed products data with error handling."""
        try:
            self.products_df = pd.read_parquet(self.products_file)
        except FileNotFoundError:
            print(f"File {self.products_file} not found.")
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")

    def encode_text_to_embedding(self, texts: List[str]):
        """Encodes a list of texts to BERT embeddings with error handling."""
        embeddings = []
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')

        for batch in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch:batch + self.batch_size]
            if not batch_texts:
                continue
            try:
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"An error occurred during embedding extraction: {e}")

        return np.array(embeddings)

    def create_faiss_index(self):
        """Creates an FAISS IVF-HC index for efficient vector similarity search with batch processing."""
        combined_texts = self.products_df['combined_text'].tolist()
        embeddings = self.encode_text_to_embedding(combined_texts)
        expected_dim = 768  # Example: BERT base model has 768 dimensions
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions. Expected {expected_dim}, got {embeddings.shape[1]}")

        d = embeddings.shape[1]  # Dimensionality of the embeddings

        # Create the quantizer and index
        quantizer = faiss.IndexFlatL2(d)
        m = 8  # Number of sub-quantizers
        nbits = 8  # Number of bits per sub-quantizer
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, m, nbits)

        # Ensure embeddings is a numpy array
        embeddings_np = np.array(embeddings)

        # Train the index and add embeddings
        index.train(embeddings_np)

        self.index = index

    def searchIndex(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches for the k nearest neighbors of the query.

        :param query: The text query to search for.
        :param k: Number of nearest neighbors to return.
        :return: A tuple containing distances and indices of the nearest neighbors.
        """
        # Convert the query string into a numerical vector
        query_vector = self.encode_text_to_embedding([query])

        # Ensure the query vector has the correct shape for FAISS search
        query_vector = np.expand_dims(query_vector, axis=0)

        # Search the FAISS index
        distance, index = self.index.search(query_vector[0], k)

        # distance, index = self.index.search(query_vector.astype('float32'), k)

        return distance, index

    def update_product_description(self, product_id: str, new_description: str):
        """Updates the description of a product."""
        if product_id not in self.products_df['product_id'].values:
            print(f"Product ID {product_id} not found.")
            return

        product_index = self.products_df[self.products_df['product_id'] == product_id].index[0]
        self.products_df.at[product_index, 'product_description'] = new_description
        self.products_df.at[
            product_index, 'combined_text'] = f"{self.products_df.at[product_index, 'product_title']} {new_description}"
        self.update_embeddings_for_changed_products([product_id])

    def remove_product_by_id(self, product_id: str):
        """Removes a product by ID from the index and the underlying data store."""
        if product_id not in self.products_df['product_id'].values:
            print(f"Product ID {product_id} not found.")
            return

        product_index = self.products_df[self.products_df['product_id'] == product_id].index[0]
        self.products_df.drop(product_index, inplace=True)
        self.create_faiss_index()  # Re-create the index after removing the product

    def update_embeddings_for_changed_products(self, changed_product_ids: List[str]):
        """Re-encodes and re-adds embeddings for products whose descriptions were changed."""
        for product_id in changed_product_ids:
            try:
                product_index = self.products_df[self.products_df['product_id'] == product_id].index[0]
                combined_text = f"{self.products_df.at[product_index, 'product_title']} {self.products_df.at[product_index, 'product_description']}"
                new_embedding = self.encode_text_to_embedding([combined_text])[0]
                self.index.add_with_ids(new_embedding.reshape(1, -1), np.array([product_id]))
            except Exception as e:
                print(f"Error updating embeddings for product ID {product_id}: {e}")


if __name__ == "__main__":
    products_file = 'rag_application/shopping_queries_dataset/processed_products.parquet'
    vector_index = VectorIndex(products_file, batch_size=32)
    vector_index.load_processed_products()
    vector_index.create_faiss_index()
    print("FAISS index created successfully.")
