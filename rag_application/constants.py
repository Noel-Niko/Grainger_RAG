# TODO: Store using secure vaults.
import os

from langchain_core.prompts import ChatPromptTemplate

langchainApiKey = os.getenv('LANGCHAIN_API_KEY')
chatOpenAiKey = os.getenv('CHAT_OPENAI_KEY')
email = os.getenv('EMAIL')

initial_question_wrapper = ('You are a chat-bot answering product queries. ALWAYS include the Product ID: {value} '
                            'with the product title in your answers. QUESTION:')
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context but 
provide as much information as you can:

        <context>
        {context}
        </context>

        Question: {input}""")

no_matches = "No matching products found."
