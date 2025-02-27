import os
import pickle
import time
from datetime import datetime

import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import faiss
from typing import List, Dict, Tuple, Set
import logging
from rag_application import constants

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_creation_time(file_path):
    ctime = os.path.getctime(file_path)
    creation_time = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"File '{file_path}' was created on {creation_time}")


class VectorIndex:
    """
    VectorIndex for creating and querying a FAISS index using BERT embeddings.
    Uses batch processing to avoid loading the entire dataset into memory at once.
    Ensures that the FAISS index is created once and reused throughout the application life of the container.
    """
    _instance = None

    @classmethod
    def get_instance(cls, **kwargs):
        """Static access method to get the singleton instance, enforcing required arguments."""
        logging.info("Entering get_instance method")
        if cls._instance is None:
            logging.info("Instance is None, creating new instance")
            pickle_file = kwargs.get('pickle_file', 'vector_index.pkl')
            products_file = kwargs.get('products_file', '')

            # Check if 'products_file' is a string
            if not isinstance(products_file, str):
                logging.error("'products_file' argument must be a string")
                raise TypeError("'products_file' argument must be a string")

            if os.path.exists(pickle_file):
                logging.info(f"Loading VectorIndex instance from {pickle_file}")
                try:
                    with open(pickle_file, 'rb') as file:
                        cls._instance = pickle.load(file)
                    logging.info("VectorIndex instance loaded from pickle file.")
                except Exception as e:
                    logging.error(f"Failed to load VectorIndex from pickle file: {e}")
                    raise
            else:
                logging.info("Creating new instance of VectorIndex...")
                cls._instance = cls(products_file=products_file)
                try:
                    cls._instance.verify_or_wait_for_file_creation()
                    cls._instance.load_processed_products()
                    cls._instance.create_faiss_index()
                    with open(pickle_file, 'wb') as file:
                        pickle.dump(cls._instance, file)
                    logging.info("VectorIndex instance created and serialized to pickle file.")
                except Exception as e:
                    logging.error(f"Failed to initialize the FAISS index: {str(e)}")
                    raise RuntimeError(f"Error initializing the FAISS index: {str(e)}")
        else:
            logging.info("Using existing instance of VectorIndex")

        return cls._instance

    def __init__(self, products_file=None, batch_size=32):  # m=16
        self.products_df = None
        self.llm = None
        self.products_file = products_file
        self.batch_size = batch_size
        self.embeddings_dict = {}
        self._index = None
        self._is_index_created = False

    def load_processed_products(self):
        """Loads the processed products data with error handling."""
        logging.info("Loading preprocessed products.")

        try:
            self.products_df = pd.read_parquet(self.products_file)
            # Ensure 'product_id' is both an index and a column
            if 'product_id' not in self.products_df.columns:
                raise KeyError("'product_id' must be a column in the DataFrame.")
            logging.info("Completed loading preprocessed products.")
        except FileNotFoundError:
            logging.error(f"File {self.products_file} not found.")
        except Exception as e:
            logging.error(f"An error occurred while loading the file: {e}")

    def encode_text_to_embedding(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts to BERT embeddings with error handling."""
        logging.info("Encoding text to embedding.")
        embeddings = []
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Log transformers library version
        logging.info(f"Transformers library version: {transformers.__version__}")
        model = AutoModel.from_pretrained('bert-base-uncased')

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch in range(0, len(texts), self.batch_size):
            logging.info(f"Encoding text batch {batch // self.batch_size + 1} of {total_batches}.")
            batch_texts = texts[batch:batch + self.batch_size]
            if not batch_texts:
                continue
            try:
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = model(**inputs)
                # Regarding 0, in  batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
                # See notes at end of file.
                batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()

                # Dimensionality check
                expected_dim = 768  # BERT-base embeddings have 768 dimensions
                if batch_embeddings.ndim != 2 or batch_embeddings.shape[1] != expected_dim:
                    raise ValueError(
                        f"Inconsistent embedding dimensions. Expected {expected_dim}, got {batch_embeddings.shape[1]}")

                embeddings.extend(batch_embeddings)
            except Exception as e:
                logging.error(f"An error occurred during embedding extraction: {e}")
        logging.info("Returning embeddings.")
        # Normalize along the embedding axis
        embeddings = normalize(np.array(embeddings), axis=1)
        return embeddings

    def create_faiss_index(self):
        """Creates an FAISS IVF-HC index for efficient vector similarity search with batch processing."""
        logging.info("Creating an FAISS IVF-HC index for efficient vector similarity search with batch processing.")

        titles = self.products_df['product_title'].tolist()
        embeddings = self.encode_text_to_embedding(titles)

        # Update embeddings_dict with product_id as key and embedding as value
        for i, product_id in enumerate(self.products_df['product_id']):
            self.embeddings_dict[product_id] = embeddings[i]

        logging.info("Embeddings dictionary updated.")

        expected_dim = 768  # BERT base model has 768 dimensions
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            msg = f"Inconsistent embedding dimensions. Expected {expected_dim}, got {embeddings.shape[1]}"
            logging.error(msg)
            raise ValueError(msg)

        d = embeddings.shape[1]  # Dimensionality of the embeddings

        # Create the quantizer and index.
        logging.info("Creating quantizer")
        quantizer = faiss.IndexFlatL2(d)

        # Each vector is split into m subvectors/subquantizers.
        m = 8

        # There's a trade-off between memory efficiency and search accuracy.
        # Using more bits per subquantizer generally leads to more accurate searches
        # but requires more memory.
        bits = 8  # Reduced bits to ensure it fits within the limitations

        # Calculate a suitable nlist value
        num_points = embeddings.shape[0]
        nlist = max(1, int(np.sqrt(num_points)))  # Ensure nlist is at least 1

        # Ensure nlist does not exceed the number of points
        if nlist > num_points:
            nlist = num_points

        # IVFPQ chosen for improved speed
        self._index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

        # Ensure embeddings is a numpy array
        embeddings_np = np.array(embeddings)

        # Generate numeric IDs for FAISS
        numeric_ids = np.arange(len(self.products_df)).astype(np.int64)

        # Train the index and add embeddings
        logging.info(f"Checking if trained: {self._index.is_trained}")
        if not self._index.is_trained:
            logging.info("Training...")
            self._index.train(embeddings_np)

        logging.info(f"Is trained: {self._index.is_trained}")

        logging.info("Embedding...")
        self._index.add_with_ids(embeddings_np, numeric_ids)
        logging.info(f"Embedding completed. nTotal = {self._index.ntotal}")
        self._is_index_created = True

    def get_first_10_vectors(self):
        """Returns the first 10 vectors in the index dataframe. Used for testing."""
        return self.products_df.head(10)

    def search_index(self, query_text: str, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Searches the FAISS index for the top_k most similar items to the query_text.
        Returns a tuple of lists: (indices, distances) of the top_k results.
        """
        logging.info("Searching the FAISS index for the top_k most similar items to the query_text.")

        # Check if the FAISS index is created
        if not self._is_index_created:
            raise RuntimeError("Index is not initialized.")

        # Check if query_text is empty or only whitespace
        if not query_text.strip():
            raise ValueError("Query string cannot be empty.")

        # Search By Id: Convert to uppercase for Product ID's which are stored in uppercase.
        logging.info(f"Searching by product id: {query_text.upper()}")
        _, id_index = self.search_by_product_id(query_text)
        id_distance = 0.0 if id_index is not None else None
        logging.info(f"Search by product id returned: {id_index}")

        # Search By Title
        logging.info(f"Searching product titles for: {query_text}")
        logging.info(f"Query text: {query_text.lower()}")
        # Encode query_text to get query_embedding
        query_embedding = self.encode_text_to_embedding([query_text.lower()])
        logging.info(f"Query embedding shape: {query_embedding.shape}")

        # Perform the search in FAISS index
        logging.info("Performing the search...")
        distances, indices = self._index.search(query_embedding, top_k)
        logging.info("Search completed.")

        # Initialize variables to store combined results
        combined_indices = []
        combined_distances = []

        # Check if the search returned any results
        if len(indices[0]) > 0:
            combined_indices = list(indices[0])
            combined_distances = list(distances[0])

            # Assuming id_index is a single integer
            if id_index is not None and id_index not in combined_indices:
                combined_indices.insert(0, id_index)
                combined_distances.insert(0, id_distance)
        else:
            logging.warning("FAISS search returned no results.")

        # Trim to top_k results if necessary
        combined_indices = combined_indices[:top_k]
        combined_distances = combined_distances[:top_k]

        # Filter out any non-integer values from combined_indices
        combined_indices = [idx for idx in combined_indices if isinstance(idx, np.int64)]

        return combined_indices, combined_distances

    def update_embeddings_for_changed_products(self, changed_product_ids: List[int]):
        """
        Updates the embeddings for the specified changed products and updates the FAISS index.
        """
        logging.info("Updating embeddings for changed products.")
        changed_products = self.products_df[self.products_df['product_id'].isin(changed_product_ids)]
        titles = changed_products['product_title'].tolist()
        new_embeddings = self.encode_text_to_embedding(titles)

        # Update embeddings_dict with new embeddings
        for i, product_id in enumerate(changed_products['product_id']):
            self.embeddings_dict[product_id] = new_embeddings[i]

        logging.info("Embeddings updated in dictionary. Updating FAISS index...")
        # Remove old embeddings for the changed products
        old_indices = np.array(changed_product_ids).astype(np.int64)
        self._index.remove_ids(old_indices)

        # Add new embeddings to the index
        new_embeddings_np = np.array(new_embeddings)
        new_numeric_ids = np.array(list(changed_products.index)).astype(np.int64)
        self._index.add_with_ids(new_embeddings_np, new_numeric_ids)
        logging.info("FAISS index updated.")

    def verify_or_wait_for_file_creation(self, timeout: int = 300, interval: int = 5):
        """
        Waits for the products file to be created within a given timeout period.
        """
        logging.info("Waiting for products file creation.")
        start_time = time.time()
        while not os.path.exists(self.products_file):
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise TimeoutError(f"File {self.products_file} was not created within {timeout} seconds.")
            time.sleep(interval)

        log_creation_time(self.products_file)
        logging.info("Products file verified.")

    def search_by_product_id(self, product_id):
        """
        Searches for an embedding by product ID in the FAISS index.

        Args:
            product_id (int or str): The product ID to search for.

        Returns:
            tuple: A tuple containing the embedding (numpy array) and the index of the nearest neighbor.
                   If the embedding is not found or no valid neighbor is found, returns (None, None).
        """
        if isinstance(product_id, str):
            product_id = product_id.upper()
        logging.info(f"Searching for embedding by product ID: {product_id}")

        if not self._index.is_trained:
            raise RuntimeError("FAISS index is not trained.")

        try:
            embedding = self.embeddings_dict[product_id]
            logging.info("Embedding found.")

            # Convert embedding to numpy array if needed
            embedding_np = np.array(embedding, dtype=np.float32)  # Assuming embedding is already a numpy array

            # Perform a search in the FAISS index to find the nearest neighbor
            D, I = self._index.search(np.expand_dims(embedding_np, axis=0), 1)

            if len(I) > 0 and I[0][0] != -1:  # Check if a valid index was found
                index = int(I[0][0])  # Extract the index of the nearest neighbor
                logging.info(f"Nearest neighbor index: {index}")
                return embedding_np, index
            else:
                logging.error("No valid nearest neighbor found in the FAISS index.")
                return None, None

        except KeyError:
            logging.error(f"Embedding not found for product ID: {product_id}")
            return None, None

    def search_and_generate_response(self, refined_query: str, llm, k: int = 15) -> str:
        # Search the FAISS index with the refined query
        logging.info(f"Searching the index for: {refined_query}")
        relevant_product_indices, distances = self.search_index(refined_query.lower(), top_k=k)
        logging.info(f"Returning: relevant_product_indices[{relevant_product_indices}]")
        logging.info(f"Returning: distances[{distances}]")
        # Extract the product information based on the returned indices
        logging.info(f"Extracting product details.")
        product_info_list = []
        for index in relevant_product_indices:
            try:
                if index >= 0:  # Check if a valid index was found
                    product_info = (
                        f"ID: {index}, "
                        f"Product ID: {self.products_df.loc[index, 'product_id']}, "
                        f"Name: {self.products_df.loc[index, 'product_title']}, "
                        f"Description: {self.products_df.loc[index, 'product_description']}, "
                        f"Key Facts: {self.products_df.loc[index, 'product_bullet_point']}, "
                        f"Brand: {self.products_df.loc[index, 'product_brand']}, "
                        f"Color: {self.products_df.loc[index, 'product_color']}, "
                        f"Location: {self.products_df.loc[index, 'product_locale']}"
                    )
                    product_info_list.append(product_info)
                else:
                    logging.error(f"relevant_product_indices includes an invalid index of: {index}")
            except KeyError:
                logging.warning(f"Product ID {self.products_df.loc[index, 'product_id']} not found in the DataFrame.")

        # Join the product information into a single string
        product_info_str = ", ".join(product_info_list)
        logging.info(f"Returning from search_and_generate_response: {product_info_str}")

        return product_info_str

    def remove_product_by_id(self, product_id):
        """Removes a product by ID from the index and the underlying data store."""
        logging.info(f"Removing product by ID {product_id} from the index and the underlying data store.")
        print(f"Removing product by ID {product_id} from the index and the underlying data store.")

        if product_id not in self.products_df['product_id'].values:
            raise ValueError(f"Product ID {product_id} not found.")

        # Remove the product by dropping the row with the given index label
        self.products_df = self.products_df[self.products_df['product_id'] != product_id]

        logging.info(f"Product {product_id} removed from DataFrame and index.")
        print(f"Product {product_id} removed from DataFrame and index.")

    @staticmethod
    def find_changed_products(old_descriptions, new_descriptions):
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

    # def update_product_descriptions(self, new_descriptions_map):
    #     """
    #     Batch updates the descriptions of multiple products and regenerates their embeddings.
    #
    #     Parameters:
    #     - new_descriptions_map (dict): Mapping of product IDs to their new descriptions.
    #
    #     Raises:
    #     - KeyError: If a product ID in updates is not found in the DataFrame.
    #     """
    #     logging.info(
    #         "Making batch updates for the descriptions of multiple products and regenerating their embeddings.")
    #     print("Making batch updates for the descriptions of multiple products and regenerating their embeddings.")
    #
    #     # Ensure product IDs are found in the DataFrame
    #     missing_ids = [product_id for product_id in new_descriptions_map if product_id not in self.products_df['product_id'].values]
    #     if missing_ids:
    #         raise KeyError(f"Product IDs {missing_ids} not found in the DataFrame.")
    #
    #     # Update descriptions in the DataFrame
    #     self.products_df.loc[self.products_df['product_id'].isin(new_descriptions_map.keys()), 'product_description'] = \
    #         self.products_df['product_id'].map(new_descriptions_map)
    #
    #     # Identify changed products
    #     changed_products = list(new_descriptions_map.keys())
    #
    #     if changed_products:
    #         self.update_embeddings_for_changed_products(set(changed_products))
    def update_product_descriptions(self, new_descriptions_map):
        """
        Batch updates the descriptions of multiple products and regenerates their embeddings.

        Parameters:
        - updates (dict): Mapping of product IDs to their new descriptions.

        Raises:
        - KeyError: If a product ID in updates is not found in the DataFrame.
        """
        logging.info(
            "Making batch updates for the descriptions of multiple products and regenerating their embeddings.")
        print("Making batch updates for the descriptions of multiple products and regenerating their embeddings.")

        # Ensure product IDs are found in the DataFrame
        missing_ids = [product_id for product_id in new_descriptions_map if product_id not in self.products_df['product_id'].values]
        if missing_ids:
            raise KeyError(f"Product IDs {missing_ids} not found in the DataFrame.")

        # Update descriptions in the DataFrame
        self.products_df.set_index('product_id', inplace=True)
        self.products_df.loc[new_descriptions_map.keys(), 'product_description'] = pd.Series(new_descriptions_map)
        self.products_df.reset_index(inplace=True)

        # Identify changed products
        changed_products = list(new_descriptions_map.keys())

        if changed_products:
            self.update_embeddings_for_changed_products(list(changed_products))


if __name__ == "__main__":
    try:
        VectorIndex.get_instance()
        logging.info("FAISS index created successfully.")
    except Exception as e:
        logging.error(f"Error creating the FAISS index: {e}")
        raise RuntimeError(f"Error creating the FAISS index: {e}")


"""
The most common approach when working with BERT embeddings for sentence-level tasks is to use the embedding of the `[CLS]` token (https://discuss.huggingface.co/t/significance-of-the-cls-token/3180/9). 
This token is specifically designed to aggregate information from the entire input sequence, making it a natural choice for 
sentence-level representations. However, other approaches can also be effective depending on the specific task and requirements. 
Here are some commonly used methods:

1. **[CLS] Token Embedding**:
   - The embedding of the `[CLS]` token (first token in the sequence) is often used as a summary of the whole sequence.
   ```python
   batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
   ```

2. **Mean Pooling**:
   - Average the embeddings of all tokens to get a single vector representation of the sequence.
   ```python
   batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
   ```

3. **Max Pooling**:
   - Take the maximum value for each dimension across all token embeddings in the sequence.
   ```python
   batch_embeddings, _ = outputs.last_hidden_state.max(dim=1)
   batch_embeddings = batch_embeddings.detach().numpy()
   ```

4. **Concatenation of Multiple Layers**:
   - Concatenate the embeddings from multiple layers of BERT to capture richer information.
   ```python
   # Example of concatenating last four layers
   hidden_states = outputs.hidden_states  # Ensure output_hidden_states=True when calling the model
   concatenated_layers = torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]), dim=-1)
   batch_embeddings = concatenated_layers[:, 0, :].detach().numpy()
   ```

5. **Weighted Sum of Multiple Layers**:
   - Use a weighted sum of the embeddings from multiple layers.
   ```python
   weights = [0.1, 0.2, 0.3, 0.4]  # Example weights for the last four layers
   hidden_states = outputs.hidden_states  # Ensure output_hidden_states=True when calling the model
   weighted_sum = sum(w * h for w, h in zip(weights, hidden_states[-4:]))
   batch_embeddings = weighted_sum[:, 0, :].detach().numpy()
   ```

The choice of method depends on the specific application and empirical performance. 
The `[CLS]` token approach is widely used for its simplicity and effectiveness in many tasks, 
but experimenting with other pooling strategies can sometimes yield better results for specific use cases.
"""
