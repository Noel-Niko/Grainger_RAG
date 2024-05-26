import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, Type, CollectionStatus, InsertParam, QueryType, MetricType
from typing import List

class VectorIndex:
    def __init__(self, collection_name: str, dimension: int, nlist: int = 100, m: int = 16, batch_size: int = 32):
        self.collection_name = collection_name
        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.batch_size = batch_size
        self.connect_milvus()

    def connect_milvus(self):
        """Connects to Milvus instance."""
        try:
            connections.connect()
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")

    def create_partitioned_collection(self):
        """Creates a new partitioned collection in Milvus."""
        fields = [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="id", dtype=DataType.INT64),
            FieldSchema(name=self.partition_key_field, dtype=DataType.INT64) if self.partition_key_field else None
        ]
        schema = CollectionSchema(fields, primary_field="id")
        collection = Collection(name=self.collection_name, schema=schema, partition_key=self.partition_key_field)
        if collection.status != CollectionStatus.OK:
            print(f"Collection creation failed: {collection.status}")
        else:
            print("Collection created successfully.")
        return collection

    def create_collection(self):
        """Creates a new collection in Milvus."""
        fields = [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="id", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, primary_field="id")
        collection = Collection(name=self.collection_name, schema=schema)
        if collection.status!= CollectionStatus.OK:
            print(f"Collection creation failed: {collection.status}")
        else:
            print("Collection created successfully.")
        return collection

    def insert_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Inserts vectors into the Milvus collection."""
        collection = self.get_collection()
        insert_param = InsertParam(collection_name=self.collection_name)
        collection.insert([insert_param], vectors.tolist())

    def get_collection(self):
        """Retrieves the Milvus collection."""
        collections = connections.list_collections()
        if self.collection_name not in collections:
            print(f"Collection {self.collection_name} does not exist. Creating a new one...")
            self.create_collection()
        return connections.get_connection().collection_manager.get_collection(self.collection_name)

    def search(self, query: np.ndarray, k: int = 10) -> List[int]:
        """Searches the Milvus collection for the k most similar vectors to the given query vector."""
        collection = self.get_collection()
        query_entities = [[query]]
        status, results = collection.search(query_entities, query_type=QueryType.L2, limit=k)
        if status!= 0:
            print(f"Search failed: {status}")
            return []
        return [result.id for result in results]

    def update_product_description(self, product_id: int, new_description: str):
        """Updates the description of a product."""
        collection = self.get_collection()

        # Determine the partition(s) for the product_id
        partition_filter = f"partition_key_field == {product_id}"  # Replace partition_key_field with your actual partition key field name

        # Fetch the existing vector and description for the product
        entities = collection.search([[None]], query_type=QueryType.L2, filter_expr=partition_filter, limit=1)
        if not entities:
            print(f"No product found with ID {product_id}. Cannot update.")
            return

        # Delete the existing document
        collection.delete_entity(entities[0].ids[0])

        # Encode the new description to obtain the vector
        new_vector = self.encode_text_to_embedding(
            new_description)  # Assume encode_text_to_embedding returns a float array

        # Re-insert the product with the new description
        self.insert_vectors(np.array(new_vector).reshape(1, -1), [product_id])

    def remove_product_by_id(self, product_id: int):
        """Removes a product by ID from the index and the underlying data store."""
        collection = self.get_collection()

        # Determine the partition(s) for the product_id
        partition_filter = f"partition_key_field == {product_id}"  # Replace partition_key_field with your actual partition key field name

        # Check if the product exists
        entities = collection.search([[None]], query_type=QueryType.L2, filter_expr=partition_filter, limit=1)
        if not entities:
            print(f"Product ID {product_id} not found.")
            return

        # Delete the product
        collection.delete_entity(entities[0].ids[0])


if __name__ == "__main__":
    collection_name = 'products'
    dimension = 768  # Assuming BERT embeddings
    text_vector_index = VectorIndex(collection_name, dimension, batch_size=32)
    # Load and process your data here
    # Example: vectors = np.random.rand(10000, dimension).astype('float32')
    # ids = range(len(vectors))
    # text_vector_index.insert_vectors(vectors, ids)
    # Perform search operations
