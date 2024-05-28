import os
import tempfile
from unittest.mock import Mock

from rag_application.modules.user_interface import RAGApplication
from rag_application.verify_vectorization_faiss import generate_random_product_data
import unittest

from vector_index_faiss import VectorIndex


class TestUiIntegration(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.llm_interaction = Mock()
        # Generate dummy product data
        dummy_product_data_untrained = generate_random_product_data(num_samples=1000)
        dummy_product_data_trained = dummy_product_data_untrained.dropna().drop_duplicates()

        # Create a temporary file and write the dummy product data to it
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.parquet') as temp_file:
            dummy_product_data_trained.to_parquet(temp_file.name, engine='pyarrow')
            self.vector_index = VectorIndex(products_file=temp_file.name, nlist=100, m=16, batch_size=16)
            self.temp_file_name = temp_file.name

    def tearDown(self):
        """Clean up resources after each test."""
        del self.vector_index
        if hasattr(self, 'test_vectors'):
            del self.test_vectors
        os.unlink(self.temp_file_name)

    def test_RAGApplication_initialization(self):
        generate_random_product_data(200)
        app = RAGApplication(products_file=self.temp_file_name)
        self.assertIsNotNone(app.vector_index, "VectorIndex instance is not properly initialized.")
        self.assertIsNotNone(app.llm_model, "LLMInteraction instance is not properly initialized.")

    def test_process_query(self):
        mock_query = "Sample query"
        mock_refined_query = "Refined query"
        mock_faiss_response = "FAISS response"

        # Mocking VectorIndex methods
        self.vector_index.refine_query_with_chatgpt = lambda query: (mock_refined_query, None)
        self.vector_index.search_and_generate_response = lambda refined_query, llm, k: mock_faiss_response

        app = RAGApplication(products_file=self.temp_file_name)
        response = app.process_query(mock_query)

        self.assertEqual(response, mock_faiss_response)

    def test_RAGApplication_VectorIndex_interaction(self):
        mock_query = "Sample query"
        mock_refined_query = "Refined query"
        mock_faiss_response = "FAISS response"

        # Mocking VectorIndex methods
        self.vector_index.refine_query_with_chatgpt = lambda query: (mock_refined_query, None)
        self.vector_index.search_and_generate_response = lambda refined_query, llm, k: mock_faiss_response

        app = RAGApplication(products_file=self.temp_file_name)
        response = app.process_query(mock_query)

        self.assertEqual(response, mock_faiss_response)

    def test_RAGApplication_LLMInteraction_interaction(self):
        mock_query = "Sample query"
        mock_response = "Mock response"

        # Mocking LLMInteraction methods
        self.llm_interaction.llm_interaction = lambda query, llm: mock_response

        app = RAGApplication(products_file=self.temp_file_name)
        response = app.process_query(mock_query)
        response = app.process_query(mock_query)

        self.assertEqual(response, mock_response)
