import numpy as np

import lda
import parser
from matrix_factorization import matrix_factorization_based_collaborative_filtering, calculate_root_mean_square_error


def main():
    api_records = parser.read_api_data('data/api.txt')
    mashup_records = parser.read_mashup_data('data/mashup.txt')

    dictionary, token, corpus, lda_model = lda.lda_train()
    mashup_specification = "I want to find a mashup that Tells Jokes"
    similarities = lda.mashup_similarity(mashup_specification, dictionary, token, lda_model, corpus)
    best_mashup, index = lda.get_best_mashup(similarities, mashup_records)

    matrix = parser.create_matrix(mashup_records, api_records)
    mashup_matrix, api_matrix = matrix_factorization_based_collaborative_filtering(matrix)
    predictions = np.dot(mashup_matrix, api_matrix)
    root_mean_square_error = calculate_root_mean_square_error(matrix, predictions)
    print('Root mean square error: ' + str(root_mean_square_error))

    print("Best mashup: " + best_mashup['name'])
    print("Best mashup index: " + str(index))
    result = predictions[index]
    # for i in range(len(result)):
    #     if result[i] > 0:
    #         print(i, api_records[i]['name'], result[i])


if __name__ == "__main__":
    main()
