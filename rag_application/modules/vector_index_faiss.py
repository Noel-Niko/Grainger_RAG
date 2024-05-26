import pandas as pd
import numpy as np
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
