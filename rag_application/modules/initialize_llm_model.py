# import os
# import logging
# from typing import Optional
# from ChatGPT import ChatGPT
#
# # Setup basic configuration for logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# def initialize_llm_model(model_name: str = 'gpt-chat', api_token: Optional[str] = None) -> ChatGPT:
#     """
#     Initializes the LLM model with LangChain.
#
#     :param model_name: The name of the model to use ('gpt-chat' by default).
#     :param api_token: The API token for authentication (optional).
#     :return: An instance of the ChatGPT model.
#     """
#     if not api_token:
#         logger.error("API token is required for initializing the LLM model.")
#         raise ValueError("API token is required.")
#
#     return ChatGPT(model_name=model_name, api_token=api_token)
#
#
# def llm_interaction(query: str, llm: ChatGPT) -> str:
#     """
#     Interacts with the LLM to generate a response based on the given query.
#
#     :param query: The query to send to the LLM.
#     :param llm: An instance of the ChatGPT model.
#     :return: The response generated by the LLM.
#     """
#     try:
#         response = llm.generate_response(query)
#         logger.info(f"Received response: {response[:50]}...")  # Log the start of the response for brevity
#         return response
#     except Exception as e:
#         logger.error(f"Failed to generate response: {e}")
#         raise
#
#
# # Example usage
# if __name__ == "__main__":
#     # Load the API token securely
#     api_token = os.getenv(langchainApiKey)
#     if not api_token:
#         logger.error("Environment variable LANGCHAIN_API_TOKEN is not set.")
#         raise EnvironmentError("API token is missing.")
#
#     # Initialize the LLM model
#     llm = initialize_llm_model(api_token=api_token)
#
#     # Simulate a query
#     query = "What are the top 5 features of the latest smartphone?"
#
#     # Get a response from the LLM
#     response = llm_interaction(query, llm)
#
#     # Print the response
#     print(response)
# Ω

import os
import logging
from typing import Optional
from pyChatGPT import ChatGPT

from rag_application.constants import langchainApiKey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInteraction:
    # @staticmethod
    # def initialize_llm_model(model_name: str = 'gpt-chat', api_token: Optional[str] = None) -> ChatGPT:
    #     """
    #     Initializes the LLM model with LangChain.
    #
    #     :param model_name: The name of the model to use ('gpt-chat' by default).
    #     :param api_token: The options API token for authentication.
    #     :return: An instance of the ChatGPT model.
    #     """
    #     if not api_token:
    #         logger.error("API token is required for initializing the LLM model.")
    #         raise ValueError("API token is required.")
    #
    #     return ChatGPT(model_name=model_name, api_token=api_token)
    # @staticmethod
    # def initialize_llm_model(api_token: Optional[str] = None) -> ChatGPT:
    #     """
    #     Initializes the LLM model with LangChain.
    #
    #     :param api_token: The options API token for authentication.
    #     :return: An instance of the ChatGPT model.
    #     """
    #     if not api_token:
    #         logger.error("API token is required for initializing the LLM model.")
    #         raise ValueError("API token is required.")
    #
    #     # Assuming ChatGPT now expects only an api_token for initialization
    #     return ChatGPT(api_token=api_token)

    from typing import Optional
    import logging

    logger = logging.getLogger(__name__)

    @staticmethod
    def initialize_llm_model(session_token: str, conversation_id: str, auth_type: str, email: str, password: str,
                             login_cookies_path: str, captcha_solver: str, solver_apikey: str, proxy: str,
                             chrome_args: list, moderation: bool, verbose: bool) -> ChatGPT:
        """
        Initializes the LLM model with LangChain using ChatGPT.

        :param session_token: The session token to use for authentication
        :param conversation_id: The conversation ID to use for the chat session
        :param auth_type: The authentication type to use
        :param email: The email to use for authentication
        :param password: The password to use for authentication
        :param login_cookies_path: The path to the cookies file to use for authentication
        :param captcha_solver: The captcha solver to use
        :param solver_apikey: The apikey of the captcha solver to use
        :param proxy: The proxy to use for the browser
        :param chrome_args: The arguments to pass to the browser
        :param moderation: Whether to enable message moderation
        :param verbose: Whether to enable verbose logging
        :return: An instance of the ChatGPT model.
        """
        # Collect all necessary parameters for ChatGPT initialization
        # This is a placeholder; you'll need to fill in the actual values
        return ChatGPT(
            session_token=session_token,
            conversation_id=conversation_id,
            auth_type=auth_type,
            email=email,
            password=password,
            login_cookies_path=login_cookies_path,
            captcha_solver=captcha_solver,
            solver_apikey=solver_apikey,
            proxy=proxy,
            chrome_args=chrome_args,
            moderation=moderation,
            verbose=verbose
        )

    @staticmethod
    def llm_interaction(query: str, llm: ChatGPT) -> str:
        """
        Interacts with the LLM to generate a response based on the given query.

        :param query: The query to send to the LLM.
        :param llm: An instance of the ChatGPT model.
        :return: The response generated by the LLM.
        """
        try:
            response = llm.driver.send_message(query)
            logger.info(f"Received response: {response[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise

    if __name__ == "__main__":
        api_token = os.getenv(langchainApiKey)
        if not api_token:
            logger.error("Environment variable LANGCHAIN_API_TOKEN is not set.")
            raise EnvironmentError("API token is missing.")

        # Initialize the LLM model
        llm = initialize_llm_model(api_token=api_token)
