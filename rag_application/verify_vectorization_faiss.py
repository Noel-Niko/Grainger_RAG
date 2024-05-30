import os
import random
import tempfile
import unittest
import random
import string
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from faker import Faker

from rag_application.modules.vector_index_faiss import VectorIndex

fake = Faker()
searchable_terms = ['APPLE', 'hammer']


def generate_random_product_data(num_samples=10000, searchable_keywords=['PRODUCTS', 'EXTRA']):
    """Generates random product data for testing purposes, including a guaranteed searchable term."""
    product_ids = range(1, num_samples + 1)
    product_titles = [fake.catch_phrase() for _ in range(num_samples)]
    product_descriptions = [fake.text(max_nb_chars=200) for _ in range(num_samples)]
    product_bullet_points = [f"Key feature {i + 1}: {fake.word()}." for i in range(num_samples)]
    product_brands = [fake.company() for _ in range(num_samples)]
    product_colors = [fake.color_name() for _ in range(num_samples)]
    product_locales = [fake.city() for _ in range(num_samples)]

    combined_text = [f"{product_titles[i]} - {product_descriptions[i]}" for i in range(num_samples)]

    # Update a portion of the combined_text with varied searchable keywords
    for i, keyword in enumerate(searchable_keywords):
        combined_text[i] += f' {keyword}'

    product_data = pd.DataFrame({
        'product_id': product_ids,
        'product_title': product_titles,
        'product_description': product_descriptions,
        'product_bullet_point': product_bullet_points,
        'product_brand': product_brands,
        'product_color': product_colors,
        'product_locale': product_locales,
        'combined_text': combined_text
    })

    # Generate additional entries with varied combinations
    additional_entries = []
    for i in range(20):
        # Select a random sample
        base_entry = product_data.sample(n=1).iloc[0].copy()
        entry_with_keyword = base_entry.copy()

        # Choose a random keyword from the list
        keyword = searchable_keywords[i % len(searchable_keywords)]
        entry_with_keyword[
            'combined_text'] = f"{keyword} - {entry_with_keyword['product_title']} - {entry_with_keyword['product_description']}"

        # Introduce additional random variations
        entry_with_keyword['product_title'] = fake.catch_phrase()
        entry_with_keyword['product_description'] = fake.text(max_nb_chars=200)
        entry_with_keyword['product_bullet_point'] = f"New key feature {i + 1}: {fake.word()}."
        entry_with_keyword['product_brand'] = fake.company()
        entry_with_keyword['product_color'] = fake.color_name()
        entry_with_keyword['product_locale'] = fake.city()

        entry_with_keyword['product_id'] = len(product_data) + i + 1
        additional_entries.append(entry_with_keyword)

    additional_df = pd.DataFrame(additional_entries)
    product_data = pd.concat([product_data, additional_df], ignore_index=True)

    return product_data


class TestVectorIndex(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        # ********  Generate dummy product data  ********
        dummy_product_data_untrained = generate_random_product_data(num_samples=1000,
                                                                    searchable_keywords=searchable_terms)
        dummy_product_data_trained = dummy_product_data_untrained.dropna().drop_duplicates()

        # Create a temporary file and write the dummy product data to it
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.parquet') as temp_file:
            dummy_product_data_trained.to_parquet(temp_file.name, engine='pyarrow')
            self.vector_index = VectorIndex(products_file=temp_file.name, nlist=100, m=16, batch_size=16)
            self.temp_file_name = temp_file.name

        # ********  Generate from a created pickle file ********
        # Specify the pickle file path
        # vector_index_path = 'vector_index.pkl'
        # with open(vector_index_path, 'rb') as file:
        #     print("Loading vector index...from pickle file.")
        #     self.vector_index = pickle.load(file)
        #
        # self.assertIsNotNone(self.vector_index, "Failed to load VectorIndex instance from pickle file.")

    def tearDown(self):
        """Clean up resources after each test."""
        del self.vector_index
        if hasattr(self, 'test_vectors'):
            del self.test_vectors

    def set_up_data(self):
        # Verify that the VectorIndex instance has been properly initialized
        self.assertIsNotNone(self.vector_index, "VectorIndex instance is not properly initialized.")

        # Load the processed products data
        self.vector_index.load_processed_products()

        # Check if the products file exists before attempting to create the FAISS index
        self.assertTrue(os.path.exists(self.vector_index.products_file), "Products file does not exist.")

        # Create the FAISS index
        print("creating the FAISS index")
        self.vector_index.create_faiss_index()
        print("FAISS index created")

        # Verify that the FAISS index is now populated
        print("Verifying the FAISS index")
        self.assertIsNotNone(self.vector_index, "FAISS index is not created.")
        print("FAISS index verified")

    def test_create_faiss_index(self):
        """Test creating the index."""
        self.set_up_data()

        # Perform a simple search to verify the functionality of the index
        query_result = self.vector_index.search_index("sample query", k=3)
        self.assertIsInstance(query_result, tuple, "Search returned unexpected result type.")
        self.assertGreater(len(query_result[1].tolist()[0]), 0, "No results found for the sample query.")

    def test_single_word_search(self):
        """Test searching for nearest neighbors."""
        self.set_up_data()
        self.assertIsNotNone(self.vector_index, "VectorIndex instance is not properly initialized.")
        self.vector_index.load_processed_products()
        self.assertTrue(os.path.exists(self.vector_index.products_file), "Products file does not exist.")
        self.vector_index.create_faiss_index()
        self.assertIsNotNone(self.vector_index._index, "FAISS index is not created.")

        query_string = searchable_terms[0]
        distances, product_ids = self.vector_index.search_index(query_string, k=5)

        # Ensure distances is a numpy array
        self.assertIsInstance(distances, np.ndarray, "Distances are not a numpy array.")
        self.assertEqual(5, len(distances.tolist()[0]), "Number of distances does not match k.")

        # Ensure product_ids is a list
        self.assertIsInstance(product_ids, np.ndarray, "Product IDs are not a list.")
        self.assertGreaterEqual(len(product_ids.tolist()[0]), 1, "No product IDs returned.")
        self.assertLessEqual(len(product_ids.tolist()[0]), 6, "More than 5 product IDs returned.")

        # Ensure each product_id is an integer or string
        for pid in product_ids.tolist()[0]:
            self.assertIsInstance(pid, (int, str), "Product ID is not an integer or string.")

    # Test with Known Queries
    def test_search(self):
        """
        Test the search functionality with known queries.
        """
        known_queries = ["red shirt", "blue dress", "green shoes"]
        for query in known_queries:
            distances, indices = self.search_index(query)
            print(f"Query: {query}")
            print("Nearest Neighbors:")
            for dist, idx in zip(distances[0], indices[0]):
                print(f"Distance: {dist}, Index: {idx}")
            print()
    def test_search_and_generate_response(self):
        """Test search_and_generate_response method."""
        # Set up data
        self.vector_index.load_processed_products()
        self.vector_index.create_faiss_index()

        # Define a refined query
        refined_query = "sample query"

        # Call the method
        response = self.vector_index.search_and_generate_response(refined_query, llm=None, k=5)

        # Check the response
        self.assertIsInstance(response, str, "Response is not a string.")
        self.assertGreater(len(response), 0, "Response is empty.")

    # @patch.object(VectorIndex, 'search_index')
    # @patch.object(VectorIndex, 'products_df', new_callable=pd.DataFrame)
    # def test_search_and_generate_response(self, mock_products_df, mock_search_index):
    #     # Setup
    #     vector_index_instance = VectorIndex()
    #
    #     # Mock the search_index method to return predefined distances and indices
    #     mock_search_index.return_value = (np.array([[0.1, 0.2, 0.3]]), np.array([[1, 2, 3]]))
    #
    #     # Mock the products_df DataFrame
    #     mock_products_df.return_value = pd.DataFrame({
    #         'numeric_index': [1, 2, 3],
    #         'product_title': ['Title1', 'Title2', 'Title3'],
    #         'product_description': ['Desc1', 'Desc2', 'Desc3'],
    #         'product_bullet_point': ['Bullet1', 'Bullet2', 'Bullet3'],
    #         'product_brand': ['Brand1', 'Brand2', 'Brand3'],
    #         'product_color': ['Color1', 'Color2', 'Color3'],
    #         'product_locale': ['Locale1', 'Locale2', 'Locale3']
    #     })
    #
    #     # Call the method under test
    #     refined_query = "example query"
    #     llm = MagicMock()
    #     response = vector_index_instance.search_and_generate_response(refined_query, llm, k=3)
    #
    #     # Assertions
    #     expected_response = "ID: 1, Name: Title1, Description: Desc1, Key Facts: Bullet1, Brand: Brand1, Color: Color1, Location: Locale1, ID: 2, Name: Title2, Description: Desc2, Key Facts: Bullet2, Brand: Brand2, Color: Color2, Location: Locale2, ID: 3, Name: Title3, Description: Desc3, Key Facts: Bullet3, Brand: Brand3, Color: Color3, Location: Locale3"
    #     self.assertEqual(response, expected_response)
    #
    #     # Verify that search_index was called with the correct parameters
    #     mock_search_index.assert_called_once_with(refined_query, k=3)

    def test_empty_query_vector(self):
        """Test searching with an empty query vector."""
        self.set_up_data()
        with self.assertRaises(ValueError) as context:
            self.vector_index.search_index("", k=5)
        self.assertEqual(str(context.exception), "Query string cannot be empty.")

    def test_uninitialized_index_search(self):
        """Test searching with an uninitialized index."""
        # self.set_up_data()
        with self.assertRaises(RuntimeError) as context:
            self.vector_index.search_index(searchable_terms[0], k=5)
        self.assertIn("Index is not initialized.", str(context.exception))

    def test_find_changed_products(self):
        """Test finding products with changed descriptions."""
        self.set_up_data()

        first_10_vectors = self.vector_index.get_first_10_vectors()

        # Randomly select 3 vectors to update.
        selected_product_ids = first_10_vectors.sample(n=3)['product_id'].tolist()

        # Create new descriptions
        new_descriptions = {}
        for product_id in selected_product_ids:
            new_descriptions[product_id] = f"Updated description for product {product_id}"

        # Test identifying the changed vectors.
        changed_product_ids = self.vector_index.find_changed_products(
            first_10_vectors.set_index('product_id')['product_description'], new_descriptions)

        expected_changed_product_ids = set(selected_product_ids)
        self.assertEqual(changed_product_ids, expected_changed_product_ids, "Incorrect product IDs identified as "
                                                                            "changed")

    def test_update_product_descriptions(self):
        """Test updating product descriptions and regenerating embeddings using batch updates."""
        self.set_up_data()

        # Randomly select a subset of products to update.
        all_product_ids = self.vector_index.get_all_product_ids()
        selected_product_ids = random.sample(all_product_ids, k=3)

        # Create new descriptions for these products.
        new_descriptions = {product_id: f"Updated description for product {product_id}" for product_id in
                            selected_product_ids}

        self.vector_index.update_product_descriptions(new_descriptions)

        # Verify descriptions have been updated correctly.
        for product_id, new_description in new_descriptions.items():
            updated_row = self.vector_index.products_df.loc[self.vector_index.products_df['product_id'] == product_id]
            self.assertEqual(updated_row.iloc[0]['product_description'], new_description)
            self.assertEqual(updated_row.iloc[0]['combined_text'],
                             f"{updated_row.iloc[0]['product_title']} {new_description}")

        # TODO: Verify that the embeddings for the updated products have been regenerated correctly.
        # for product_id, _ in new_descriptions.items():
        #     # Fetch the original embedding
        #     original_embedding = self.vector_index.get_embedding(product_id)
        #
        #     # Fetch the updated embedding
        #     updated_embedding = self.vector_index.get_embedding(product_id)
        #
        #     # Compare the original and updated embeddings
        #     self.assertTrue(np.allclose(original_embedding, updated_embedding, atol=1e-6),
        #                 f"The embeddings for product {product_id} did not match after update.")

    def test_remove_product_by_id(self):
        """Test the remove_product_by_id method."""
        self.set_up_data()

        # Select product IDs for testing
        all_product_ids = self.vector_index.get_all_product_ids()
        product_to_remove = all_product_ids[0]
        other_product_ids = all_product_ids[1:3]  # Using the next two product IDs for verification

        # Remove a product
        self.vector_index.remove_product_by_id(product_to_remove)

        # Verify the product has been removed from the DataFrame
        remaining_rows = self.vector_index.products_df[self.vector_index.products_df['product_id'] != product_to_remove]
        self.assertEqual(len(remaining_rows), len(self.vector_index.products_df))

        # Verify the product's embedding has been removed from the FAISS index
        with self.assertRaises(Exception) as context:
            self.verify_embedding_removed(product_to_remove)
        self.assertTrue("Embedding not found" in str(context.exception),
                        "The embedding believed removed was still found")

        # Check that other products remain unchanged
        for product_id in other_product_ids:
            updated_row = self.vector_index.products_df.loc[self.vector_index.products_df['product_id'] == product_id]
            self.assertIsNotNone(updated_row.iloc[0])

        # Attempt  to remove a nonexistent product
        with self.assertRaises(Exception) as context:
            self.vector_index.remove_product_by_id("nonexistent_product_id")
        self.assertTrue("product_id not found." in str(context.exception),
                        "The product_id believed removed was still found")

    def verify_embedding_removed(self, product_id):
        """Verifies that the embedding for a given product ID has been removed from the FAISS index."""
        if product_id not in self.vector_index.products_df['product_id'].values:
            raise RuntimeError("Embedding not found")
        else:
            # If the embedding is fetched successfully.
            raise RuntimeError("Error: Embedding believed removed was found")


if __name__ == '__main__':
    unittest.main()
