import numpy as np 
from sklearn.model_selection import train_test_split

def chang_rate_data(X, y , new_rate = 1/ 15):
    #pos_Y luu thu tu cua y(+)
    #neg_y luu so thu tu cua y(-)
    pos_y = np.where(y == 1)[0]
    # permutation
    pos_y = np.random.permutation(pos_y)
    neg_y = np.where( y == -1 )[0]
    rate = new_rate #examble
    pos_y_choosed =int(pos_y.shape[0]-(pos_y.shape[0] - rate * y.shape[0])/( 1 -rate) )
    pos_y =  pos_y[0:pos_y_choosed]
    #gop index 
    y_index = np.concatenate((neg_y, pos_y), axis = None)
    X = X[y_index]
    y = y[y_index]
    return X, y