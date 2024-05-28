# # # user_interface.py
# # import streamlit as st
# # from rag_application.modules.vector_index_faiss import VectorIndex
# # from llm_interaction import LLMInteraction
# #
# #
# # def main():
# #     st.title("Relevance-Aware Generation (RAG) Application")
# #
# #     # Define the dimension and index path for VectorIndex
# #     dimension = 768  # Example dimension; adjust based on your data
# #     index_path = 'path/to/your/vector/index.flt'
# #
# #     vector_index = VectorIndex(dimension, index_path)
# #     llm_interaction = LLMInteraction()
# #
# #     query = st.text_input("Enter your product-based question:")
# #     if st.button("Submit"):
# #         response = process_query(query, vector_index, llm_interaction)
# #         st.write("Response:", response)
# #
# #
# # def process_query(query, vector_index, llm_interaction):
# #     # Retrieve relevant documents from the vector index
# #     query_vector = vector_index.retrieve_documents(query)
# #
# #     # If no relevant documents are found, return a default response
# #     if not query_vector:
# #         return "No relevant documents found for the query."
# #
# #     # Generate a response using the language model
# #     response = llm_interaction.generate_response(query)
# #
# #     # Return the generated response
# #     return response
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
# import streamlit as st
# from rag_application.modules.vector_index_faiss import VectorIndex  # Adjusted import statement
# from rag_application.modules.llm_interface import LLMInteraction
# from rag_application.constants import langchainApiKey
#
# llm_interaction = LLMInteraction(model_name='gpt-chat', api_token=langchainApiKey)
# llm_interaction = wrap_openai(llm_interaction)
#
#
# def generate_response_wrapper(query: str):
#     return llm_interaction.generate_response(query)
#
#
# class RAGApplication:
#     def __init__(self, products_file):
#         self.vector_index = VectorIndex.getInstance()  # Use the singleton pattern
#         self.llm_interaction = LLMInteraction(model_name='gpt-chat', api_token=langchainApiKey)
#         self.llm_interaction = wrap_openai(self.llm_interaction)
#
#     def main(self):
#         st.title("Relevance-Aware Generation (RAG) Application")
#         query = st.text_input("Enter your product-based question:")
#         if st.button("Submit"):
#             response = self.process_query(query)
#             st.write("Response:", response)
#
#     def process_query(self, query):
#         refined_query = generate_response_wrapper(query)
#         _, relevant_product_indices = self.vector_index.search_index(refined_query, k=5)  # Search for top 5 matches
#         product_info = ", ".join(
#             [f"ID: {pid}, Name: {self.vector_index.products_df.loc[pid, 'product_name']}" for pid in
#              relevant_product_indices]
#         )
#         response = f"Top 5 matching products: {product_info}"
#         return response
#
#
# if __name__ == "__main__":
#     products_file = 'path/to/your/products/file.parquet'
#     app = RAGApplication(products_file)
#     app.main()
import streamlit as st
from rag_application.modules.vector_index_faiss import VectorIndex
from rag_application.modules.initialize_llm_model import LLMInteraction
from rag_application.constants import langchainApiKey


class RAGApplication:
    def __init__(self, products_file):
        self.vector_index = VectorIndex.getInstance()
        self.llm_model = LLMInteraction.initialize_llm_model(model_name='gpt-chat', api_token=langchainApiKey)
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
