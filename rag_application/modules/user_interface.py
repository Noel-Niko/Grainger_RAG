# # user_interface.py
# import streamlit as st
# from vector_index import VectorIndex
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

def main():
    st.title("Simple Streamlit App")

    # Display a text input box
    user_input = st.text_input("Enter something:")

    # Display a button
    if st.button("Submit"):
        # Display the input text
        st.write("You entered:", user_input)


if __name__ == "__main__":
    main()
