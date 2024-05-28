import streamlit as st
from rag_application.modules.vector_index_faiss import VectorIndex
from rag_application.modules.initialize_llm_model import LLMInteraction
from rag_application.constants import langchainApiKey


class RAGApplication:
    def __init__(self, products_file):
        self.products_file = products_file
        self.vector_index = VectorIndex.getInstance(products_file=self.products_file)
        self.llm_model = LLMInteraction.initialize_llm_model(api_token=langchainApiKey)
        self.current_query = None
        # self.llm_interaction = wrap_openai(self.llm_interaction)

    def main(self):
        st.title("Relevance-Aware Generation (RAG) Application")
        query = st.text_input("Enter your product-based question:")
        if st.button("Submit"):
            self.current_query = query
            response = self.process_query(query)
            st.write("Response:", response)

    def process_query(self, query):
        # Parse the query using the LLM
        refined_query, llm = self.vector_index.refine_query_with_chatgpt(query)

        # Search for the refined query in the FAISS index
        faiss_response = self.vector_index.search_and_generate_response(refined_query, llm, k=5)
        prompt_wrapper = (f"I was asked - {self.current_query}. Use this information to generate an answer to what I "
                          f"was asked. I found: {faiss_response}")
        return llm.llm_interaction(prompt_wrapper, llm)


if __name__ == "__main__":
    products_file = 'shopping_queries_dataset/processed_products.parquet'
    app = RAGApplication(products_file)
    app.main()
