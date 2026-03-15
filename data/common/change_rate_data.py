# import library
import numpy as np


# Change rate of positive class in data

def change_rate_data(X, y, new_rate):
    #pos_Y luu thu tu cua y(+)
    #neg_y luu so thu tu cua y(-)
    pos_y = np.where(y == 1)[0]
    # permutation
    pos_y = np.random.permutation(pos_y)
    neg_y = np.where( y == -1 )[0]
    rate0 = pos_y.shape[0]/(y.shape[0])
    # pos_y_choosed =int((pos_y.shape[0] - new_rate * y.shape[0])/( 1 -new_rate) ) 
    pos_y_choosed = int(neg_y.shape[0]*new_rate)
    pos_y =  pos_y[0:pos_y_choosed]
    print(f"Number of positive_samples: {len(pos_y)}")
    #gop index 
    y_index = np.concatenate((pos_y, neg_y), axis = None)
    X = X[y_index]
    y = y[y_index]
    return X, y

def change_rate_data_cnn(X, y, new_rate):
    #pos_Y luu thu tu cua y(+)
    #neg_y luu so thu tu cua y(-)
    pos_y = np.where(y == 1)[0]
    # permutation
    pos_y = np.random.permutation(pos_y)
    neg_y = np.where( y == 0 )[0]
    rate0 = pos_y.shape[0]/(y.shape[0])
    # pos_y_choosed =int((pos_y.shape[0] - new_rate * y.shape[0])/( 1 -new_rate) ) 
    pos_y_choosed = int(neg_y.shape[0]*new_rate)
    pos_y =  pos_y[0:pos_y_choosed]
    #gop index 
    y_index = np.concatenate((pos_y, neg_y), axis = None)
    X = X[y_index]
    y = y[y_index]
    return X, y

# #Test
# import pandas as pd
# data = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/LEC7/madaboost_25112021/madaboost/data/datasets/Co_Author_100_500_1.csv')
# X = data.values[:, 0:-1]
# y = data.values[:, 7]
# X, y = change_rate_data(X, y, new_rate=1/9)
# X = np.array(X)
# print(X.shape)