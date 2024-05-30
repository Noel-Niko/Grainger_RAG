import logging
import os
import pickle

import streamlit as st
from rag_application.modules.vector_index_faiss import VectorIndex
from rag_application.constants import chatOpenAiKey, initial_question_wrapper, prompt, no_matches
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain


class RAGApplication:
    def __init__(self, vector_index_instance=None):
        self.vector_index = vector_index_instance
        self.llm_connection = ChatOpenAI(api_key=chatOpenAiKey)
        self.current_query = None


    # Persist the state to prevent recreating the index
    # def get_vector_index(self):
    #     """Get or create the VectorIndex instance."""
    #     if 'vector_index' not in st.session_state:
    #         logging.info("Creating VectorIndex instance")
    #         try:
    #             with open('vector_index.pkl', 'rb') as file:
    #                 self.vector_index = pickle.load(file)
    #                 st.session_state['vector_index'] = self.vector_index
    #                 logging.info("VectorIndex instance loaded from file")
    #         except (FileNotFoundError, pickle.PickleError) as e:
    #             logging.error("Failed to load VectorIndex from file: {}".format(e))
    #             logging.info("Creating VectorIndex instance")
    #             base_dir = os.path.dirname(os.path.abspath(__file__))
    #             data_dir = os.path.join(base_dir, 'shopping_queries_dataset')
    #             products_file = os.path.join(data_dir, 'processed_products.parquet')
    #             self.vector_index = VectorIndex.get_instance(products_file=products_file)
    #             st.session_state['vector_index'] = self.vector_index
    #             logging.info("VectorIndex instance created and saved to session state")
    #         else:
    #             logging.info("VectorIndex instance already exists in session state")
    #             self.vector_index = st.session_state['vector_index']

    def get_vector_index(self):
        """Get or create the VectorIndex instance."""
        # Check if 'vector_index' exists in session state
        if 'vector_index' in st.session_state:
            logging.info("VectorIndex instance already exists in session state")
            self.vector_index = st.session_state['vector_index']
        else:
            # Attempt to load VectorIndex from file
            vector_index_path = 'vector_index.pkl'
            try:
                with open(vector_index_path, 'rb') as file:
                    self.vector_index = pickle.load(file)
                    st.session_state['vector_index'] = self.vector_index
                    logging.info(f"VectorIndex instance loaded from {vector_index_path}")
            except (FileNotFoundError, pickle.PickleError) as e:
                logging.error(f"Failed to load VectorIndex from {vector_index_path}: {e}")
                # Create VectorIndex if not found in session state or file
                logging.info("Creating VectorIndex instance")
                base_dir = os.path.dirname(os.path.abspath(__file__))
                data_dir = os.path.join(base_dir, 'shopping_queries_dataset')
                products_file = os.path.join(data_dir, 'processed_products.parquet')
                self.vector_index = VectorIndex.get_instance(products_file=products_file)
                st.session_state['vector_index'] = self.vector_index
                logging.info("VectorIndex instance created and saved to session state")

    def main(self):
        self.get_vector_index()
        st.title("Relevance-Aware Generation (RAG) Application")
        query = st.text_input("Enter your product-based question:")
        if st.button("Submit"):
            self.current_query = query
            response = self.process_query(query)
            st.write("Response:", response)

    def process_query(self, query):
        # Parse the query using the LLM
        refined_query = self.llm_connection.invoke(f"{initial_question_wrapper} {query}").content

        # Search for the refined query in the FAISS index

        context_faiss_response = self.vector_index.search_and_generate_response(refined_query, self.llm_connection, k=5)
        if context_faiss_response is None or context_faiss_response.strip() == "":
            context_faiss_response = no_matches

        # Pass the product information back to the LLM to form a response message
        document_chain = create_stuff_documents_chain(self.llm_connection, prompt)
        context_document = Document(page_content=context_faiss_response)
        return document_chain.invoke({
            "input": f"{query}",
            "context": [context_document]
        })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = RAGApplication()
    app.main()
