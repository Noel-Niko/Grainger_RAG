# TODO: Store using secure vaults.
from langchain_core.prompts import ChatPromptTemplate

langchainApiKey = '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'
chatOpenAiKey = 'sk-proj-F7ow6Bu53kHPKaQDYbA0T3BlbkFJ2ykR1JffM2jA6Tg0qPOH'
initial_question_wrapper = ('I was asked the following question but only have access to data about products and their '
                            'descriptive characteristics. Refine the question to a text string form suitable for '
                            'searching my products faiss vector index. Add up to 10 additional key words that are '
                            'synonyms or examples of the key words you found. QUESTION:')
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}""")

no_matches = "No matching products found."
