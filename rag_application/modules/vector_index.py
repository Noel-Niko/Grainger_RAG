# import numpy as np
# import faiss
# from typing import List
#
#
# class VectorIndex:
#     def __init__(self, dimension: int, index_path: str):
#         self.dimension = dimension
#         self.index_path = index_path
#         self.index = None
#         self.vectors = None
#         self.load_index()
#
#     def load_index(self):
#         try:
#             self.index = faiss.read_index(self.index_path)
#         except IOError:
#             print(f"Index not found at {self.index_path}. Initializing a new one.")
#             self.initialize_index()
#
#     def initialize_index(self):
#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.save_index()
#
#     def save_index(self):
#         faiss.write_index(self.index, self.index_path)
#
#     def add_vectors(self, vectors: np.ndarray):
#         assert vectors.shape[1] == self.dimension, "Vectors must have the correct dimension."
#         self.vectors = vectors
#         self.index.add(vectors)
#
#     def search(self, query: np.ndarray, k: int = 10) -> List[int]:
#         distances, indices = self.index.search(query.reshape(1, -1), k)
#         return indices.flatten().tolist()
#
#     def retrieve_documents(self, query: str) -> List[str]:
#         # Assuming you have a function to convert the query string to a vector
#         query_vector = self.query_to_vector(query)
#         results_indices = self.search(query_vector, k=5)
#         # Assuming you have a function to map indices to document IDs or titles
#         document_ids = self.indices_to_document_ids(results_indices)
#         return document_ids
#
#     # Placeholder functions for converting query to vector and mapping indices to document IDs
#     def query_to_vector(self, query: str) -> np.ndarray:
#         raise NotImplementedError("Implement this method to convert a query string to a vector.")
#
#     def indices_to_document_ids(self, indices: List[int]) -> List[str]:
#         raise NotImplementedError("Implement this method to map indices to document IDs or titles.")

#
# import numpy as np
# import faiss
# from typing import List, Dict
# import threading
#
#
# class VectorIndex:
#     def __init__(self, dimension: int, index_path: str):
#         self.dimension = dimension
#         self.index_path = index_path
#         self.index = None
#         self.lock = threading.Lock()  # Lock for thread safety
#         self.load_index()
#
#     def load_index(self):
#         try:
#             self.index = faiss.read_index(self.index_path)
#         except IOError:
#             print(f"Index not found at {self.index_path}. Initializing a new one.")
#             self.initialize_index()
#
#     def initialize_index(self):
#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.save_index()
#
#     def save_index(self):
#         with self.lock:
#             faiss.write_index(self.index, self.index_path)
#
#     def add_vectors(self, vectors: np.ndarray):
#         assert vectors.shape[1] == self.dimension, "Vectors must have the correct dimension."
#         with self.lock:
#             self.index.add(vectors)
#
#     def search(self, query: np.ndarray, k: int = 10) -> List[int]:
#         distances, indices = self.index.search(query.reshape(1, -1), k)
#         return indices.flatten().tolist()
#
#     def retrieve_documents(self, query: str, query_filter=None) -> List[str]:
#         # Assuming you have a function to convert the query string to a vector
#         query_vector = self.query_to_vector(query)
#         results_indices = self.search(query_vector, k=5)
#
#         # Apply filter if provided
#         if query_filter:
#             filtered_results_indices = [index for index in results_indices if query_filter(index)]
#             results_indices = filtered_results_indices
#
#         # Assuming you have a function to map indices to document IDs or titles
#         document_ids = self.indices_to_document_ids(results_indices)
#         return document_ids
#
#     # Placeholder functions for converting query to vector and mapping indices to document IDs
#     def query_to_vector(self, query: str) -> np.ndarray:
#         raise NotImplementedError("Implement this method to convert a query string to a vector.")
#
#     def indices_to_document_ids(self, indices: List[int]) -> List[str]:
#         raise NotImplementedError("Implement this method to map indices to document IDs or titles.")
#
#
# # Example usage
# if __name__ == "__main__":
#     index = VectorIndex(dimension=128, index_path='path/to/index')
#     # Add vectors to the index
#     vectors = np.random.rand(10000, 128).astype('float32')  # Example vectors
#     index.add_vectors(vectors)
#
#     # Retrieve documents
#     query = "example query"
#     document_ids = index.retrieve_documents(query)
#     print(document_ids)

# import numpy as np
# import faiss
#
# def create_faiss_index(text_vectors, nlist=100, m=16):
#     """
#     Creates an IVF-HC index for efficient vector similarity search.
#     IVH-HC chosen over Flat Index,  Heirarchical Navigable SmallW World, or Product Quantization with Polysemous Codes
#         for Approximate Nearest Neighbors Index.
#
#     Parameters:
#     - text_vectors: A 2D numpy array of shape (num_texts, embedding_dim) containing the embeddings of your texts.
#     - nlist: Number of clusters in the hierarchical clustering.
#     - m: Number of centroids per cluster.
#
#     Returns:
#     - An FAISS index object.
#     """
#     # Convert text vectors to float32
#     text_vectors = text_vectors.astype(np.float32)
#
#     # Create the IVF-HC index
#     quantizer = faiss.IndexFlatL2(text_vectors.shape[1])  # The underlying vector quantizer
#     index = faiss.IndexIVFFlat(quantizer, text_vectors.shape[1], nlist)  # The IVF index
#
#     # Train the index
#     index.train(text_vectors)
#
#     # Add vectors to the index
#     index.add(text_vectors)
#
#     return index
#
# def search(index, query_embedding, k=10):
#     """
#     Searches the FAISS index for the k most similar vectors to the given query vector.
#
#     Parameters:
#     - index: The FAISS index object.
#     - query_embedding: A 1D numpy array representing the query vector.
#     - k: Number of nearest neighbors to return.
#
#     Returns:
#     - Dists: A 2D numpy array of shape (k,) containing the distances to the nearest neighbors.
#     - Indices: A 1D numpy array of shape (k,) containing the indices of the nearest neighbors.
#     """
#     dists, indices = index.search(query_embedding.reshape(1, -1), k)
#     return dists.flatten(), indices.flatten()
#
#
#
# import numpy as np
# import faiss
# from typing import List
#
# class VectorIndex:
#     def __init__(self, dimension: int, index_path: str):
#         self.dimension = dimension
#         self.index_path = index_path
#         self.index = None
#         self.load_index()
#
#     def load_index(self):
#         try:
#             self.index = faiss.read_index(self.index_path)
#         except IOError:
#             print(f"Index not found at {self.index_path}. Initializing a new one.")
#             self.initialize_index()
#
#     def initialize_index(self):
#         self.index = faiss.IndexFlatL2(self.dimension)
#         self.save_index()
#
#     def save_index(self):
#         faiss.write_index(self.index, self.index_path)
#
#     def add_vectors(self, vectors: np.ndarray):
#         assert vectors.shape[1] == self.dimension, "Vectors must have the correct dimension."
#         self.index.add(vectors)
#
#     def search(self, query: np.ndarray, k: int = 10) -> List[int]:
#         distances, indices = self.index.search(query.reshape(1, -1), k)
#         return indices.flatten().tolist()
#
#     def retrieve_documents(self, query: str) -> List[str]:
#         # Placeholder for query processing and result retrieval
#         raise NotImplementedError("Implement this method to convert a query string to a vector and map indices to document IDs or titles.")
#
#     # Placeholder functions for converting query to vector and mapping indices to document IDs
#     def query_to_vector(self, query: str) -> np.ndarray:
#         raise NotImplementedError("Implement this method to convert a query string to a vector.")
#
#     def indices_to_document_ids(self, indices: List[int]) -> List[str]:
#         raise NotImplementedError("Implement this method to map indices to document IDs or titles.")

#
#
# import pandas as pd
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModel
# import faiss
#
# class VectorIndex:
#     def __init__(self, products_file: str, nlist: int = 100, m: int = 16):
#         self.products_file = products_file
#         self.nlist = nlist
#         self.m = m
#         self.index = None
#
#     def load_processed_products(self):
#         """Loads the processed products data."""
#         self.products_df = pd.read_parquet(self.products_file)
#
#     def encode_text_to_embedding(self, text):
#         """Encodes text to BERT embeddings."""
#         tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#         model = AutoModel.from_pretrained('bert-base-uncased')
#
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         outputs = model(**inputs)
#         return outputs.last_hidden_state[:, 0, :].detach().numpy()
#
#     def create_faiss_index(self):
#         """Creates an FAISS IVF-HC index for efficient vector similarity search."""
#         combined_texts = self.products_df['combined_text'].tolist()
#         embeddings = [self.encode_text_to_embedding(text)[0] for text in combined_texts]
#         d = embeddings[0].shape[0]  # Dimensionality of the embeddings
#
#         # Create the quantizer
#         quantizer = faiss.IndexFlatL2(d)
#
#         # Create the IVF-HC index
#         index_ivf = faiss.IndexIVFFlat(quantizer, d, self.nlist)
#
#         # Train the index
#         index_ivf.train(embeddings)
#
#         # Add embeddings to the index
#         index_ivf.add(embeddings)
#
#         self.index = index_ivf
#
#     def search(self, query: str, k: int = 10) -> List[int]:
#         query_embedding = self.encode_text_to_embedding(query)
#         distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
#         return indices.flatten().tolist()
#
# if __name__ == "__main__":
#     products_file = 'rag_application/shopping_queries_dataset/processed_products.parquet'
#     text_vector_index = VectorIndex(products_file)
#     text_vector_index.load_processed_products()
#     text_vector_index.create_faiss_index()
#     print("FAISS index created successfully.")


#
# import pandas as pd
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModel
# import faiss
#
# class VectorIndex:
#     def __init__(self, products_file: str, nlist: int = 100, m: int = 16):
#         self.products_file = products_file
#         self.nlist = nlist
#         self.m = m
#         self.index = None
#
#     def load_processed_products(self):
#         """Loads the processed products data."""
#         self.products_df = pd.read_parquet(self.products_file)
#
#     def encode_text_to_embedding(self, text):
#         """Encodes text to BERT embeddings."""
#         tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#         model = AutoModel.from_pretrained('bert-base-uncased')
#
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         outputs = model(**inputs)
#         return outputs.last_hidden_state[:, 0, :].detach().numpy()
#
#     def create_faiss_index(self):
#         """Creates an FAISS IVF-HC index for efficient vector similarity search."""
#         combined_texts = self.products_df['combined_text'].tolist()
#         embeddings = [self.encode_text_to_embedding(text)[0] for text in combined_texts]
#         d = embeddings[0].shape[0]  # Dimensionality of the embeddings
#
#         # Create the quantizer
#         quantizer = faiss.IndexFlatL2(d)
#
#         # Create the IVF-HC index
#         index_ivf = faiss.IndexIVFFlat(quantizer, d, self.nlist)
#
#         # Train the index
#         index_ivf.train(embeddings)
#
#         # Add embeddings to the index
#         index_ivf.add(embeddings)
#
#         self.index = index_ivf
#
#     def search(self, query: str, k: int = 10) -> List[int]:
#         query_embedding = self.encode_text_to_embedding(query)
#         distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
#         return indices.flatten().tolist()
#
# if __name__ == "__main__":
#     products_file = 'rag_application/shopping_queries_dataset/processed_products.parquet'
#     text_vector_index = VectorIndex(products_file)
#     text_vector_index.load_processed_products()
#     text_vector_index.create_faiss_index()
#     print("FAISS index created successfully.")

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from typing import List

class VectorIndex:
    """
        Using batch processing to avoid loading the entire dataset into memory at once.
        Batch size of 16 chosen as this app will be running on a local MAC.
        Monitor performance and increase if able.
    """
    def __init__(self, products_file: str, nlist: int = 100, m: int = 16, batch_size: int = 32):
        self.products_file = products_file
        self.nlist = nlist
        self.m = m
        self.batch_size = batch_size
        self.index = None
        self.products_df = pd.DataFrame

    def load_processed_products(self):
        """Loads the processed products data with error handling."""
        try:
            self.products_df = pd.read_parquet(self.products_file)
        except FileNotFoundError:
            print(f"File {self.products_file} not found.")
            return
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            return

    def encode_text_to_embedding(self, texts: List[str]):
        """Encodes a list of texts to BERT embeddings with error handling."""
        embeddings = []
        for batch in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch:batch+self.batch_size]
            if not batch_texts:
                continue
            try:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                model = AutoModel.from_pretrained('bert-base-uncased')
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = model(**inputs)
                embeddings.extend(outputs.last_hidden_state[:, 0, :].detach().numpy())
            except Exception as e:
                print(f"An error occurred during embedding extraction: {e}")
        return np.array(embeddings)

    def create_faiss_index(self):
        """Creates an FAISS IVF-HC index for efficient vector similarity search with batch processing."""
        combined_texts = self.products_df['combined_text'].tolist()
        embeddings = self.encode_text_to_embedding(combined_texts)
        d = embeddings.shape[1]  # Dimensionality of the embeddings

        # Create the quantizer
        quantizer = faiss.IndexFlatL2(d)

        # Create the IVF-HC index
        index_ivf = faiss.IndexIVFFlat(quantizer, d, self.nlist)

        # Train the index
        index_ivf.train(embeddings)

        # Add embeddings to the index
        index_ivf.add(embeddings)

        self.index = index_ivf

    # def create_faiss_index(self):
    #     """Creates an FAISS IVF-HC index for efficient vector similarity search with batch processing."""
    #     combined_texts = self.products_df['combined_text'].tolist()
    #     embeddings = self.encode_text_to_embedding(combined_texts)
    #     d = embeddings.shape[1]  # Dimensionality of the embeddings
    #
    #     # Create the quantizer
    #     quantizer = faiss.IndexFlatL2(d)
    #
    #     # Create the IVF-HC index
    #     index_ivf = faiss.IndexIVFFlat(quantizer, d, self.nlist)
    #
    #     # Train the index
    #     index_ivf.train(embeddings)
    #
    #     # Add embeddings to the index
    #     # index_ivf.add(embeddings)
    #
    #     # Select a smaller subset of embeddings
    #     subset_embeddings = embeddings[:min(500, embeddings.shape[0])]
    #
    #     # Create and train the index with the subset
    #     index_ivf = faiss.IndexIVFFlat(quantizer, d, self.nlist)
    #     index_ivf.train(subset_embeddings)
    #     print(subset_embeddings.shape)
    #
    #     index_ivf.add(subset_embeddings)
    #
    #     self.index = index_ivf

    def search(self, query: str, k: int = 10) -> List[int]:
        query_embedding = self.encode_text_to_embedding([query])[0]
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        # 'product_id' is a unique identifier for each row in the DataFrame
        matching_product_ids = self.products_df.iloc[indices[0]].product_id.tolist()
        return matching_product_ids

    def update_product_description(self, product_id: str, new_description: str):
        """Updates the description of a product."""
        if product_id not in self.products_df['product_id'].values:
            print(f"Product ID {product_id} not found.")
            return

        # Find the index of the product in the DataFrame
        product_index = self.products_df[self.products_df['product_id'] == product_id].index[0]

        # Update the description
        self.products_df.at[product_index, 'product_description'] = new_description

        # Re-encode the updated description and add it back to the index
        self.products_df.at[
            product_index, 'combined_text'] = f"{self.products_df.at[product_index, 'product_title']} {new_description}"
        self.update_embeddings_for_changed_products([product_id])

    def remove_product_by_id(self, product_id: str):
        """Removes a product by ID from the index and the underlying data store."""
        if product_id not in self.products_df['product_id'].values:
            print(f"Product ID {product_id} not found.")
            return

        # Remove the product from the DataFrame
        self.products_df.drop(product_id, inplace=True)

        # Remove the corresponding embedding from the FAISS index
        product_indices = self.products_df[self.products_df['product_id'] == product_id].index.tolist()
        for idx in product_indices:
            del self.index[self.index.ntotal - idx - 1]  # Adjust index based on deletion order

    def update_embeddings_for_changed_products(self, changed_product_ids: List[str]):
        """Re-encodes and re-adds embeddings for products whose descriptions were changed."""
        for product_id in changed_product_ids:
            self.products_df.at[
                self.products_df[self.products_df['product_id'] == product_id].index[0], 'combined_text'] = \
                f"{self.products_df.at[self.products_df[self.products_df['product_id'] == product_id].index[0], 'product_title']} " + \
                self.encode_text_to_embedding([self.products_df.at[
                                                   self.products_df[self.products_df['product_id'] == product_id].index[
                                                       0], 'product_description']])[0]
        self.create_faiss_index()  # Re-create the index after changes



if __name__ == "__main__":
    products_file = 'rag_application/shopping_queries_dataset/processed_products.parquet'
    text_vector_index = VectorIndex(products_file, batch_size=32)
    text_vector_index.load_processed_products()
    text_vector_index.create_faiss_index()
    print("FAISS index created successfully.")
