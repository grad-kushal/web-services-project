import os; os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import implicit
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares




def MF():

    dfu = pd.read_csv('example.csv', encoding='windows-1252')
    user_rows = dfu['mashup_id'].unique().tolist()
    item_rows = dfu['api_id'].unique().tolist()

    user_map = {}
    for i, k  in enumerate(user_rows):
        user_map[k] = i

    item_map = {}
    for i, k  in enumerate(item_rows):
        item_map[k] = i

    dfu['mashup_id'] = dfu['mashup_id'].map(user_map)
    dfu['api_id'] = dfu['api_id'].map(item_map)

    user = np.array(dfu['mashup_id'])
    item = np.array(dfu['api_id'])
    data = np.array(dfu['probability'])

    matrix = coo_matrix((data,(user,item)),shape=(7393,12000))

    num_factors = 10
    alpha = 40
    reg = 0.1
    iterations = 50

    # Initialize the ALS model
    model = AlternatingLeastSquares(factors=num_factors, regularization=reg, iterations=iterations)

    # Fit the model to the user-item matrix
    model.fit((matrix * alpha).astype('double'))

    # Get the user and item latent factors
    user_factors = model.user_factors
    item_factors = model.item_factors

    # Compute the predicted usage matrix from the latent factors
    predicted_matrix = user_factors.dot(item_factors.T)

    # Scale the predicted matrix to be between 0 and 1
    min_pred,max_pred = predicted_matrix.min() , predicted_matrix.max()
    predicted_matrix = (predicted_matrix - min_pred) / (max_pred - min_pred)

    # Print the resulting predicted usage matrix
    print(predicted_matrix)



if __name__ == "__main__":
    MF()