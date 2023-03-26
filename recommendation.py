import numpy as np

import lda
import parser
from matrix_factorization import matrix_factorization_based_collaborative_filtering


def main():
    api_records = parser.read_api_data('data/api.txt')
    mashup_records = parser.read_mashup_data('data/mashup.txt')

    dictionary, token, corpus, lda_model = lda.lda_train()
    mashup_specification = "I want to find a mashup that can help me find the best restaurants in my area"
    similarity = lda.mashup_similarity(mashup_specification, dictionary, token, lda_model, corpus)

    matrix = parser.create_matrix(mashup_records, api_records)
    mashup_matrix, api_matrix = matrix_factorization_based_collaborative_filtering(matrix)
    predictions = np.dot(mashup_matrix, api_matrix)


if __name__ == "__main__":
    main()
