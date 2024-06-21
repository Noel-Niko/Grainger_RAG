import nltk
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def refine_question(question):
    # Define a regular expression to detect product IDs (e.g., B07XYDG2R2)
    product_id_pattern = r'\b[A-Z0-9]{10}\b'

    # Check if the question contains a product ID
    product_id_match = re.search(product_id_pattern, question)

    if product_id_match:
        # If a product ID is found, return it
        return product_id_match.group(0)

    # Tokenize the question
    words = word_tokenize(question)

    # Perform POS tagging
    pos_tags = nltk.pos_tag(words)

    # List of words to exclude, converted to lowercase for case-insensitive comparison
    exclude_words = {word.lower() for word in
                     ['provide', 'details', 'about', 'tell', 'me', 'for', 'looking', 'what', 'is', 'the', 'product',
                      'id']}

    # Define relevant POS tags
    relevant_pos = {'NN', 'NNS', 'JJ', 'NNP', 'NNPS'}

    # Extract relevant words based on POS tags, excluding specific terms, and maintain original order
    keywords = [word for word, pos in pos_tags if pos in relevant_pos and word.lower() not in exclude_words]

    # If no keywords were found, return the original question
    if not keywords:
        return question

    # Join keywords to form the refined query
    refined_question = ' '.join(keywords)

    return refined_question
