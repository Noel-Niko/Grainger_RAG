import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from rag_application.modules.vector_index import VectorIndex
from faker import Faker

fake = Faker()
searchable_term = 'APPLE'


def generate_random_product_data(num_samples=3900, searchable_keyword='PRODUCTS'):
    """Generates random product data for testing purposes, including a guaranteed searchable term."""
    product_ids = range(1, num_samples + 1)
    product_titles = [fake.catch_phrase() for _ in range(num_samples)]
    product_descriptions = [fake.text(max_nb_chars=200) for _ in range(num_samples)]
    product_bullet_points = [f"Key feature {i + 1}: {fake.word()}." for i in range(num_samples)]
    product_brands = [fake.company() for _ in range(num_samples)]
    product_colors = [fake.color_name() for _ in range(num_samples)]
    product_locales = [fake.city() for _ in range(num_samples)]

    # Include the searchable keyword in a portion of the combined_text
    combined_text = [f"{product_titles[i]} - {product_descriptions[i]}" for i in range(num_samples)]
    combined_text_with_keyword = [ct.replace(' ', ' ') + ' ' + searchable_keyword for ct in
                                  combined_text[:num_samples // 2]]
    combined_text += combined_text_with_keyword[len(combined_text):len(combined_text) + len(combined_text_with_keyword)]

    return pd.DataFrame({
        'product_id': product_ids,
        'product_title': product_titles,
        'product_description': product_descriptions,
        'product_bullet_point': product_bullet_points,
        'product_brand': product_brands,
        'product_color': product_colors,
        'product_locale': product_locales,
        'combined_text': combined_text
    })


class TestVectorIndex(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        # Generate dummy product data
        dummy_product_data = generate_random_product_data(num_samples=100, searchable_keyword=searchable_term)

        # Create a temporary file and write the dummy product data to it
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.parquet') as temp_file:
            dummy_product_data.to_parquet(temp_file.name, engine='pyarrow')
            self.vector_index = VectorIndex(products_file=temp_file.name, nlist=100, m=16, batch_size=16)
            self.temp_file_name = temp_file.name

    def tearDown(self):
        """Clean up resources after each test."""
        del self.vector_index
        if hasattr(self, 'test_vectors'):
            del self.test_vectors
        os.unlink(self.temp_file_name)

    def test_create_faiss_index(self):
        """Test creating the index."""
        # Verify that the VectorIndex instance has been properly initialized
        self.assertIsNotNone(self.vector_index, "VectorIndex instance is not properly initialized.")

        # Load the processed products data
        self.vector_index.load_processed_products()

        # Check if the products file exists before attempting to create the FAISS index
        self.assertTrue(os.path.exists(self.vector_index.products_file), "Products file does not exist.")

        # Create the FAISS index
        self.vector_index.create_faiss_index()

        # Verify that the FAISS index is now populated
        self.assertIsNotNone(self.vector_index.index, "FAISS index is not created.")

        # Optionally, perform a simple search to verify the functionality of the index
        # Note: You might need to adjust the query and parameters based on your specific setup
        query_result = self.vector_index.search("sample query", k=3)
        self.assertIsInstance(query_result, list, "Search returned unexpected result type.")
        self.assertGreater(len(query_result), 0, "No results found for the sample query.")


    def test_single_word_search(self):
        """Test searching for nearest neighbors."""
        # Ensure VectorIndex is properly initialized and the index is created
        if not hasattr(self.vector_index, 'index'):
            raise AssertionError("VectorIndex is not properly initialized or the FAISS index was not created.")

        query_string = searchable_term

        distances, indices = self.vector_index.search(query_string, k=5)

        self.assertIsInstance(distances, np.ndarray)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(len(distances), len(indices))
        self.assertGreaterEqual(len(indices), 1)

    def test_empty_query_vector(self):
        """Test searching with an empty query vector."""
        with self.assertRaises(ValueError) as context:
            self.vector_index.search(searchable_term, k=5)
        self.assertEqual(str(context.exception), "Query vector cannot be empty.")

    def test_small_search_radius(self):
        """Test searching with a very small radius."""
        with self.assertRaises(ValueError) as context:
            self.vector_index.search(searchable_term, k=0.001)  # k is expected to be an integer
        self.assertEqual(str(context.exception), "Search radius must be an integer greater than 0.")

    def test_uninitialized_index_search(self):
        """Test searching with an uninitialized index."""
        with self.assertRaises(RuntimeError) as context:
            self.vector_index.search(searchable_term, k=5)
        self.assertIn("Index is not initialized.", str(context.exception))


if __name__ == '__main__':
    unittest.main()
