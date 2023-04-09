import numpy as np

import latent_dirichlet_allocation
import parser
from matrix_factorization import matrix_factorization_based_collaborative_filtering, calculate_root_mean_square_error

# The weights of the relevance score and the popularity score
RELEVANCE_SCORE_WEIGHT = 0.65
POPULARITY_SCORE_WEIGHT = 0.35


def calculate_api_popularity_scores(api_records, mashup_records):
    """
    Calculate the popularity scores of APIs
    :param api_records: API records
    :param mashup_records: Mashup records
    :return: Popularity scores of APIs
    """
    # Initialize the popularity scores of APIs
    api_popularity_scores = np.zeros(len(api_records))

    # Calculate the popularity scores of APIs
    for mashup_record in mashup_records:
        for api in mashup_record['apis']:
            api_index = parser.get_api_index(api['name'], api_records)
            api_popularity_scores[api_index] += 1

    # Normalize the popularity scores of APIs
    api_popularity_scores /= len(mashup_records)

    return api_popularity_scores


def main():
    mashup_records, api_records = parser.read_data()

    lda_model, api_dictionary, tokenized_api_descriptions, api_corpus = latent_dirichlet_allocation.train_lda_model(api_records)
    mashup_specification = "A simple application for searching and browsing Freebase movie data"
    relevance_scores_from_lda = latent_dirichlet_allocation.get_similarities(mashup_specification, api_dictionary, tokenized_api_descriptions, lda_model, api_corpus)
    # print(list(relevance_scores_from_lda))

    matrix = parser.create_matrix(mashup_records, api_records)
    mashup_matrix, api_matrix = matrix_factorization_based_collaborative_filtering(matrix)
    relevance_scores_from_mf = np.dot(mashup_matrix, api_matrix)
    root_mean_square_error = calculate_root_mean_square_error(matrix, relevance_scores_from_mf)
    # print('Root mean square error: ' + str(root_mean_square_error))

    mashup_descriptions = [mashup_record['description'] for mashup_record in mashup_records]
    tokenized_mashup_descriptions = [latent_dirichlet_allocation.preprocess(mashup_description) for mashup_description in mashup_descriptions]
    mashup_similarities = latent_dirichlet_allocation.get_similarities(mashup_specification, api_dictionary, tokenized_mashup_descriptions, lda_model, api_corpus)
    # print(list(mashup_similarities))

    similar_mashup_index = np.argmax(mashup_similarities)
    similar_mashup = mashup_records[similar_mashup_index]
    print(similar_mashup)

    similar_mashup_relevance_scores = relevance_scores_from_mf[similar_mashup_index]

    combined_relevance_scores = relevance_scores_from_lda * similar_mashup_relevance_scores
    print(list(combined_relevance_scores))

    api_popularity_scores = calculate_api_popularity_scores(api_records, mashup_records)
    print(list(api_popularity_scores))

    weighted_combined_scores = combined_relevance_scores * RELEVANCE_SCORE_WEIGHT + api_popularity_scores * POPULARITY_SCORE_WEIGHT

    number_of_recommendations = 10
    recommended_api_indices = np.argsort(weighted_combined_scores)[::-1][:number_of_recommendations]

    print('Recommended APIs:')
    for recommended_api_index in recommended_api_indices:
        print(api_records[recommended_api_index]['name'])


if __name__ == "__main__":
    main()
