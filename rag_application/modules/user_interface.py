import logging
import os
import pickle
import time
from datetime import datetime
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
        st.title("Retrieval-Augmented Generation (RAG) Application")

        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

            # Display previous questions and answers
        for question, answer in st.session_state.conversation_history:
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {answer}")
            st.write("---")

        query = st.text_input("Enter your product-based question:", value="", placeholder="")

        # Using a separate session state variable to track submission
        if 'submitted' not in st.session_state:
            st.session_state.submitted = False

        col1, col2, col3 = st.columns(3)

        if col1.button("Submit"):
            st.session_state.submitted = True
        if col2.button("Clear Question"):
            query = ""
        if col3.button("Clear History"):
            st.session_state.conversation_history = []

        if st.session_state.submitted:
            self.current_query = query
            response = self.process_query(query)
            # Append the new question and answer to the conversation history
            st.session_state.conversation_history.append((query, response))
            st.write("Response:", response)
            st.session_state.submitted = False

    def process_query(self, query):
        # Concatenate conversation history with the current query
        conversation_context = " ".join(
            [f"Question: {q}, Answer: {a}" for q, a in st.session_state.conversation_history] + [f"Question: {query}"])

        # Parse the query using the LLM
        refined_query = self.llm_connection.invoke(f"{initial_question_wrapper} {conversation_context}").content

        logging.info(
            f"**************************    Searching in FAISS for {refined_query}    *******************************")
        # Search for the refined query in the FAISS index
        start_time = time.time()
        context_faiss_response = self.vector_index.search_and_generate_response(refined_query, self.llm_connection,
                                                                                k=15)
        end_time = time.time()
        search_duration = end_time - start_time
        logging.info(f"FAISS search completed in {search_duration:.2f} seconds.")
        with open('faiss_search_durations.txt', 'a') as file:
            file.write(f"FAISS search duration: {search_duration:.2f} seconds, Timestamp: {datetime.now()}\n")

        if context_faiss_response is None or context_faiss_response.strip() == "":
            context_faiss_response = no_matches

        # Pass the product information back to the LLM to form a response message
        document_chain = create_stuff_documents_chain(self.llm_connection, prompt)
        context_document = Document(page_content=context_faiss_response)
        self.current_query = ""
        return document_chain.invoke({
            "input": f"{query}",
            "context": [context_document]
        })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = RAGApplication()
    app.main()
