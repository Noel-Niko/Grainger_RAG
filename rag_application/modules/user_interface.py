# # user_interface.py
# import streamlit as st
# from rag_application.modules.vector_index_faiss import VectorIndex
# from llm_interaction import LLMInteraction
#
#
# def main():
#     st.title("Relevance-Aware Generation (RAG) Application")
#
#     # Define the dimension and index path for VectorIndex
#     dimension = 768  # Example dimension; adjust based on your data
#     index_path = 'path/to/your/vector/index.flt'
#
#     vector_index = VectorIndex(dimension, index_path)
#     llm_interaction = LLMInteraction()
#
#     query = st.text_input("Enter your product-based question:")
#     if st.button("Submit"):
#         response = process_query(query, vector_index, llm_interaction)
#         st.write("Response:", response)
#
#
# def process_query(query, vector_index, llm_interaction):
#     # Retrieve relevant documents from the vector index
#     query_vector = vector_index.retrieve_documents(query)
#
#     # If no relevant documents are found, return a default response
#     if not query_vector:
#         return "No relevant documents found for the query."
#
#     # Generate a response using the language model
#     response = llm_interaction.generate_response(query)
#
#     # Return the generated response
#     return response
#
#
# if __name__ == "__main__":
#     main()


import streamlit as st
from rag_application.modules.vector_index_faiss import VectorIndex
from rag_application.modules.llm_interface import LLMInteraction
from rag_application.utils.constants import langchainApiKey
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Wrap the LLM interaction with LangSmith's tracing capabilities
llm_interaction = LLMInteraction(model_name='gpt-chat', api_token=langchainApiKey)
llm_interaction = wrap_openai(llm_interaction)


@traceable
def generate_response_wrapper(query: str):
    return llm_interaction.generate_response(query)


class RAGApplication:
    def __init__(self, products_file):
        self.vector_index = VectorIndex(products_file)
        self.llm_interaction = LLMInteraction(model_name='gpt-chat', api_token=langchainApiKey)
        self.llm_interaction = wrap_openai(self.llm_interaction)

    def main(self):
        st.title("Relevance-Aware Generation (RAG) Application")

        query = st.text_input("Enter your product-based question:")
        if st.button("Submit"):
            response = self.process_query(query)
            st.write("Response:", response)

    def process_query(self, query):
        relevant_product_ids = self.vector_index.search(query, k=5)  # Adjust k as needed
        product_info = ", ".join(
            [f"ID: {pid}, Name: {self.vector_index.products_df.loc[pid, 'product_name']}" for pid in
             relevant_product_ids]
        )
        refined_query = f"Refining query with product info: {product_info}"
        response = generate_response_wrapper(refined_query)
        return response


if __name__ == "__main__":
    products_file = 'path/to/your/products/file.parquet'  # Update this path
    app = RAGApplication(products_file)
    app.main()
