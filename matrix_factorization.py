import numpy as np
from sklearn.decomposition import NMF

import parser


def matrix_factorization_based_collaborative_filtering(matrix):
    """
    Matrix factorization based collaborative filtering
    :param matrix: Matrix
    :return: user matrix and item matrix
    """
    # Convert the matrix into a numpy array
    np_matrix = np.array(matrix)
    number_of_components = 5

    # Perform matrix factorization
    model = NMF(n_components=number_of_components, init='random', random_state=0)
    W = model.fit_transform(np_matrix)
    H = model.components_
    return W, H


def main():
    api_records = parser.read_api_data('data/api.txt')
    mashup_records = parser.read_mashup_data('data/mashup.txt')

    matrix = parser.create_matrix(mashup_records, api_records)
    mashup_matrix, api_matrix = matrix_factorization_based_collaborative_filtering(matrix)
    predictions = np.dot(mashup_matrix, api_matrix)
    print(predictions[0])


if __name__ == "__main__":
    main()
