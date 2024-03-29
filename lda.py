import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
import parser

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load API descriptions into a list of strings
new = parser.read_api_data('data/api.txt')
api_descriptions = [api['description'] for api in new]

# Preprocess the API descriptions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    """
    Preprocess the text
    :param text: Text to be preprocessed
    :return: Preprocessed text
    """
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    frequency = gensim.corpora.dictionary.Dictionary([tokens])
    tokens = [token for token in tokens if frequency.token2id[token] != -1]
    return tokens


def lda_train():
    """
    Train the LDA model
    :return: dictionary, tokenized_descriptions, corpus, lda_model
    """

    tokenized_descriptions = [preprocess(description) for description in api_descriptions]

    # Create a dictionary from the tokenized descriptions
    dictionary = corpora.Dictionary(tokenized_descriptions)

    # Convert the tokenized descriptions into a bag-of-words representation
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_descriptions]

    # Train the LDA model on the corpus of API descriptions
    num_topics = 10  # Set the number of topics to be learned
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    return dictionary, tokenized_descriptions, corpus, lda_model


def mashup_similarity(mashup_specification, dictionary, tokenized_descriptions, lda_model, corpus):
    """
    Calculate the similarity between the mashup specification and the API descriptions
    :param mashup_specification: Mashup specification
    :param dictionary: Dictionary of tokenized descriptions
    :param tokenized_descriptions: Tokenized descriptions
    :param lda_model: LDA model
    :param corpus: Corpus
    :return: None
    """
    # Preprocess the mashup specification
    tokenized_specification = preprocess(mashup_specification)

    # Convert the tokenized specification into a bag-of-words representation
    specification_bow = dictionary.doc2bow(tokenized_specification)

    # Infer the topic distribution for the mashup specification
    specification_topics = lda_model.get_document_topics(specification_bow)

    # Calculate the cosine similarity between the mashup specification topic distribution
    # and the topic distributions of each API description
    similarities = []
    for i, api_description in enumerate(api_descriptions):
        api_tokens = tokenized_descriptions[i]
        api_bow = corpus[i]
        api_topics = lda_model.get_document_topics(api_bow)
        similarity = gensim.matutils.cossim(specification_topics, api_topics)
        # print(new[i]['name'])
        # print(f"API {i + 1} similarity: {similarity}")
        obj = {'name': new[i]['name'], 'similarity': similarity}
        similarities.append(obj)
    return similarities


def main():
    """
    Main function
    :return: None
    """
    dictionary, token, corpus, lda = lda_train()
    mashup_specification = "This site is a demo to show the functionality of the shopzilla.com API. Supports the US " \
                           "and UK  API versions."
    similarities = mashup_similarity(mashup_specification, dictionary, token, lda, corpus)
    # mashup_similarity(mashup_specification, dictionary, token, lda, corpus)
    print(similarities)


if __name__ == "__main__":
    main()


def get_best_mashup(similarities, mashup_records):
    """
    Get the best mashup based on the similarity
    :param similarities: Similarity
    :param mashup_records: Mashup records
    :return: Best mashup
    """
    best_mashup = None
    best_similarity = 0
    index = -1
    for i, mashup in enumerate(mashup_records):
        if similarities[i]['similarity'] > best_similarity:
            best_similarity = similarities[i]['similarity']
            best_mashup = mashup
            index = i
    return best_mashup, index