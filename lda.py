import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
import main

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Load API descriptions into a list of strings
api_descriptions = [i['description'] for i in main.read_data('data/api.txt')]
print(api_descriptions)
# Preprocess the API descriptions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove punctuation and special characters
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove rare and common words
    frequency = gensim.corpora.dictionary.Dictionary([tokens])
    tokens = [token for token in tokens if frequency.token2id[token] != -1]

    return tokens


def lda_train():
    tokenized_descriptions = [preprocess(description) for description in api_descriptions]

    # Create a dictionary from the tokenized descriptions
    dictionary = corpora.Dictionary(tokenized_descriptions)

    # Convert the tokenized descriptions into a bag-of-words representation
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_descriptions]

    # Train the LDA model on the corpus of API descriptions
    num_topics = 10 # Set the number of topics to be learned
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    return dictionary, tokenized_descriptions, corpus, lda_model


def mashup_similarity(dictionary,tokenized_descriptions, lda_model,corpus):
    #Preprocess the mashup specification
    mashup_specification = "mashup specification text"
    tokenized_specification = preprocess(mashup_specification)

    # Convert the tokenized specification into a bag-of-words representation
    specification_bow = dictionary.doc2bow(tokenized_specification)

    # Infer the topic distribution for the mashup specification
    specification_topics = lda_model.get_document_topics(specification_bow)

    # Calculate the cosine similarity between the mashup specification topic distribution
    # and the topic distributions of each API description
    for i, api_description in enumerate(api_descriptions):
        api_tokens = tokenized_descriptions[i]
        api_bow = corpus[i]
        api_topics = lda_model.get_document_topics(api_bow)
        similarity = gensim.matutils.cossim(specification_topics, api_topics)
        print(f"API {i+1} similarity: {similarity}")

if __name__ == "__main__":
    dictionary,token,corpus,lda= lda_train()
    mashup_similarity(dictionary,token,lda,corpus)