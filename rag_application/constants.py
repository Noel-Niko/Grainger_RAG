# TODO: Store using secure vaults.
import os

from langchain_core.prompts import ChatPromptTemplate

langchainApiKey = os.getenv('LANGCHAIN_API_KEY')
chatOpenAiKey = os.getenv('CHAT_OPENAI_KEY')
email = os.getenv('EMAIL')

initial_question_wrapper = ('I was asked the following question but only have access to data about products and their '
                            'descriptive characteristics. Refine the question to a text string form suitable for '
                            'searching my products faiss vector index. QUESTION:')
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}""")

no_matches = "No matching products found."
