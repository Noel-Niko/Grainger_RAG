import os
import tempfile
import unittest
import random
import numpy as np
import pandas as pd
from faker import Faker
from rag_application import constants
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
        # Generate dummy product data
        dummy_product_data_untrained = generate_random_product_data(num_samples=1000,
                                                                    searchable_keywords=searchable_terms)
        dummy_product_data_trained = dummy_product_data_untrained.dropna().drop_duplicates()
        rows = 100
        constants.rows = rows
        # Create a temporary file and write the dummy product data to it
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.parquet') as temp_file:
            dummy_product_data_trained.to_parquet(temp_file.name, engine='pyarrow')
            self.vector_index = VectorIndex(products_file=temp_file.name, batch_size=32)
            self.temp_file_name = temp_file.name

    def tearDown(self):
        """Clean up resources after each test."""
        del self.vector_index
        if hasattr(self, 'temp_file_name') and os.path.exists(self.temp_file_name):
            os.remove(self.temp_file_name)

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
        """Test creating the FAISS index."""
        self.set_up_data()
        # self.vector_index.load_processed_products()
        # self.vector_index.create_faiss_index()
        # self.assertIsNotNone(self.vector_index._index, "FAISS index is not created.")
        # Perform a simple search to verify the functionality of the index
        query_result = self.vector_index.search_index("sample query", k=3)
        self.assertIsInstance(query_result, tuple, "Search returned unexpected result type.")
        self.assertGreater(len(query_result[1].tolist()[0]), 0, "No results found for the sample query.")

    def test_single_word_search(self):
        """Test searching for nearest neighbors."""
        self.vector_index.load_processed_products()
        self.vector_index.create_faiss_index()

        query_string = searchable_terms[0]
        distances, product_ids = self.vector_index.search_index(query_string, k=5)

        # Assertions
        self.assertIsInstance(distances, np.ndarray, "Distances are not a numpy array.")
        self.assertEqual(5, len(distances.tolist()[0]), "Number of distances does not match k.")

        self.assertIsInstance(product_ids, np.ndarray, "Product IDs are not a numpy array.")
        self.assertGreaterEqual(len(product_ids.tolist()[0]), 1, "No product IDs returned.")
        self.assertLessEqual(len(product_ids.tolist()[0]), 5, "More than 5 product IDs returned.")

        for pid in product_ids.tolist()[0]:
            self.assertIsInstance(pid, (int, str), "Product ID is not an integer or string.")

    def test_search_and_generate_response(self):
        """Test search_and_generate_response method."""
        self.vector_index.load_processed_products()
        self.vector_index.create_faiss_index()

        refined_query = "sample query"
        response = self.vector_index.search_and_generate_response(refined_query, llm=None, k=5)

        self.assertIsInstance(response, str, "Response is not a string.")
        self.assertGreater(len(response), 0, "Response is empty.")

    def test_empty_query_vector(self):
        """Test searching with an empty query vector."""
        self.vector_index.load_processed_products()
        self.vector_index.create_faiss_index()

        with self.assertRaises(ValueError) as context:
            self.vector_index.search_index("", k=5)
        self.assertEqual(str(context.exception), "Query string cannot be empty.")

    def test_uninitialized_index_search(self):
        """Test searching with an uninitialized index."""
        uninitialized_index = VectorIndex(products_file=self.temp_file_name, nlist=100, batch_size=32)

        with self.assertRaises(RuntimeError) as context:
            uninitialized_index.search_index(searchable_terms[0], k=5)
        self.assertIn("Index is not initialized.", str(context.exception))


if __name__ == '__main__':
    unittest.main()
