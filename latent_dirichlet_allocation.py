import string

import gensim
import nltk
from nltk.corpus import stopwords

import parser

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def preprocess(api_description):
    """
    Preprocess the API description
    :param api_description: API description
    :return: Preprocessed API description
    """
    tokens = nltk.word_tokenize(api_description.lower())
    tokens = [token for token in tokens if token not in set(stopwords.words('english'))]
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [nltk.stem.WordNetLemmatizer().lemmatize(token) for token in tokens]
    frequency = gensim.corpora.dictionary.Dictionary([tokens])
    tokens = [token for token in tokens if frequency.token2id[token] > -1]
    return tokens


def train_lda_model(api_records):
    """
    Train the LDA model
    :param api_records: API records
    :return: LDA model
    """
    # Preprocess the API descriptions
    api_descriptions = [api['description'] for api in api_records]
    tokenized_api_descriptions = [preprocess(api_description) for api_description in api_descriptions]

    # Create a dictionary representation of the documents.
    dictionary = gensim.corpora.dictionary.Dictionary(tokenized_api_descriptions)

    # Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) 2-tuples.
    corpus = [dictionary.doc2bow(tokenized_api_description) for tokenized_api_description in tokenized_api_descriptions]

    # Train the LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=15, passes=10)

    # Print the topics
    # for idx, topic in lda_model.print_topics(-1):
    #     print('Topic: {} \nWords: {}'.format(idx, topic))

    # Return the LDA model, api_dictionary, tokenized_api_descriptions, and corpus
    return lda_model, dictionary, tokenized_api_descriptions, corpus


def get_similarities(mashup_specification, api_dictionary, tokenized_api_descriptions, lda_model, api_corpus):
    """
    Get the similarities between the mashup specification and the API descriptions
    :param mashup_specification: Mashup specification
    :param api_dictionary: API dictionary
    :param tokenized_api_descriptions: Tokenized API descriptions
    :param lda_model: LDA model
    :param api_corpus: API corpus
    :return: Similarities
    """
    # Preprocess the mashup specification
    tokenized_mashup_specification = preprocess(mashup_specification)

    # Create a bag-of-words representation of the mashup specification
    mashup_specification_bow = api_dictionary.doc2bow(tokenized_mashup_specification)

    # Get the topic distribution of the mashup specification
    mashup_specification_lda = lda_model[mashup_specification_bow]

    # Calculate the similarities between the mashup specification and the API descriptions
    similarities = gensim.similarities.Similarity('', api_corpus, num_features=len(api_dictionary))
    similarities = similarities[mashup_specification_lda]

    # Return the similarities
    return similarities


def main():
    api_records = parser.read_api_data('data/api.txt')
    lda_model, api_dictionary, tokenized_api_descriptions, api_corpus = train_lda_model(api_records)
    mashup_specification_1 = "A simple application for searching and browsing Freebase movie data."
    mashup_specification_2 = "A simple application for searching and browsing Freebase movie data. The application uses the Freebase API to search for movies and display the results in a list. Clicking on a movie in the list displays the movie's details in a panel. The application also uses the Freebase API to search for actors and display the results in a list. Clicking on an actor in the list displays the actor's details in a panel. The application also uses the Freebase API to search for directors and display the results in a list. Clicking on a director in the list displays the director's details in a panel."
    similarities = get_similarities(mashup_specification_2, api_dictionary, tokenized_api_descriptions, lda_model, api_corpus)
    print(similarities[146])


if __name__ == "__main__":
    main()
