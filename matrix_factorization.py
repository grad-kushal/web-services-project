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
    number_of_components = 15

    # Perform matrix factorization
    model = NMF(n_components=number_of_components, init='random', random_state=0)
    W = model.fit_transform(np_matrix)
    H = model.components_
    return W, H


def calculate_root_mean_square_error(matrix, predictions):
    """
    Calculate the root mean square error
    :param matrix: Matrix
    :param predictions: Predictions
    :return: Root mean square error
    """
    # Convert the matrix into a numpy array
    np_matrix = np.array(matrix)

    # Calculate the root mean square error
    error = np_matrix - predictions
    square_error = error ** 2
    mean_square_error = square_error.mean()
    root_mean_square_error = np.sqrt(mean_square_error)
    return root_mean_square_error


def main():
    api_records = parser.read_api_data('data/api.txt')
    mashup_records = parser.read_mashup_data('data/mashup.txt')

    matrix = parser.create_matrix(mashup_records, api_records)
    mashup_matrix, api_matrix = matrix_factorization_based_collaborative_filtering(matrix)
    predictions = np.dot(mashup_matrix, api_matrix)
    root_mean_square_error = calculate_root_mean_square_error(matrix, predictions)
    print('Root mean square error: ' + str(root_mean_square_error))


if __name__ == "__main__":
    main()
