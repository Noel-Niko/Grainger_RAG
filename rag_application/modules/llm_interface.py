# Import necessary modules
import ChatGPT


class LLMInteraction:
    def __init__(self, model_name='gpt-chat', api_token=None):
        """
        Initializes the LLM model with LangChain.
        """
        self.api_token = api_token
        self.llm = ChatGPT(model_name=model_name, api_token=self.api_token)

    def generate_response(self, query: str) -> str:
        """
            Interacts with the LLM to generate a response based on the given query.
            """
        response = self.llm.generate_response(query)
        return response

# def initialize_llm_model(model_name='gpt-chat'):
#     # Initialize the LLM model
#     llm = ChatGPT(model_name=model_name)
#     return llm

#
# def llm_interaction(query: str, llm: ChatGPT) -> str:
#     """
#     Interacts with the LLM to generate a response based on the given query.
#     """
#     # Generate a response from the LLM
#     response = llm.generate_response(query)
#     return response
#
#
# # Example usage
# if __name__ == "__main__":
#     # Initialize the LLM model
#     llm = initialize_llm_model()
#
#     # Simulate a query
#     query = "What are the top 5 features of the latest smartphone?"
#
#     # Get a response from the LLM
#     response = llm_interaction(query, llm)
#
#     # Print the response
#     print(response)
