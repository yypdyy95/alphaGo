"""
Phyton program for training and testing the fast rollout policy network based
on AlphaGo using a softmax layers

Instructions:
- comment/uncomment block for different task to fit your needs
- further information of util functions can be found in utilities.py

"""


import matplotlib.pyplot as plt
import numpy as np
import utilities as util
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.regularizers import *
#import operator
#import time
import gc

np.set_printoptions(threshold=np.inf)


###### Read patterns ###########################################################
#freq_patterns = np.genfromtxt('./patterns.dat', dtype=int, delimiter='; ', usecols=(0,), skip_footer=190000)


##### Load Data ################################################################

#number_of_moves, next_move, bit_board = util.load_data("./data/kgsgo-train10k.dat", max_moves=2000000)
number_of_moves, next_move, bit_board = util.load_data("./data/KGS-2004-19-12106-train10k.dat", max_moves=100000)
#number_of_moves, next_move, bit_board = util.load_data("./data/kgsgo-test.dat")


##### Plot positions in dataset ################################################
"""
# plot whole dataset as animation
util.plot_board(bit_board, next_move=next_move, delay_time=0.1)
"""
"""
# plot move number m
m = 224
util.plot_board(np.expand_dims(bit_board[m], axis=0))
"""

##### Visualize liberties ######################################################
"""
util.plot_liberties(bit_board, delay_time=0.5)
"""

##### Generate Input for Softmax Layer #########################################
"""
# list with number of feature planes of all models you want to compare
number_of_planes = [8, 7, 6]

# Model 1: 8 layers created by https://github.com/hughperkins/kgsgo-dataset-preprocessor
x_train8 = np.reshape(bit_board, (number_of_moves, -1))

# Model 2,3: 3 layers for stone color, 3 for liberties, and one additional
# layer for neighbors in model 2

x_train7 = np.zeros((number_of_moves, number_of_planes[1], 19, 19))
x_train6 = np.zeros((number_of_moves, number_of_planes[2], 19, 19))

for move in np.arange(number_of_moves):
    # create feature planes:
    planes_pos = util.get_board(bit_board[move])
    planes_lib = util.get_liberties(bit_board[move])
    plane_neighbors = util.get_neighbors(bit_board[move], next_move[move-1])

    # concatenate feature planes together to form models:
    x_train7[move] = np.concatenate((planes_pos, planes_lib, plane_neighbors), axis=0)
    x_train6[move] = np.concatenate((planes_pos, planes_lib), axis=0)

x_train7 = np.reshape(x_train7, (number_of_moves, -1))
x_train6 = np.reshape(x_train6, (number_of_moves, -1))

# create target vector:
y_train = np.zeros((number_of_moves, 19*19))
y_train[np.arange(number_of_moves), 19*next_move[:,0]+next_move[:,1]] =1


# add models you want to compare to training data
training_data = [y_train, x_train8, x_train7, x_train6]

gc.collect()
"""

##### Generate Test Data #######################################################

# warning: after this statement old loaded data is discarded
number_of_moves, next_move, bit_board = util.load_data("./data/kgsgo-test.dat")

# list with number of feature planes of all models you want to compare
number_of_planes = [8, 7, 6]

gc.collect()

x_test8 = np.reshape(bit_board, (number_of_moves, -1))

x_test7 = np.zeros((number_of_moves, number_of_planes[1], 19, 19))
x_test6 = np.zeros((number_of_moves, number_of_planes[2], 19, 19))

for move in np.arange(number_of_moves):
    planes_pos = util.get_board(bit_board[move])
    planes_lib = util.get_liberties(bit_board[move])
    plane_neighbors = util.get_neighbors(bit_board[move], next_move[move-1])
    x_test7[move] = np.concatenate((planes_pos, planes_lib, plane_neighbors), axis=0)
    x_test6[move] = np.concatenate((planes_pos, planes_lib), axis=0)
x_test7= np.reshape(x_test7, (number_of_moves, -1))
x_test6= np.reshape(x_test6, (number_of_moves, -1))

y_test = np.zeros((number_of_moves, 19*19))
y_test[np.arange(number_of_moves), 19*next_move[:,0]+next_move[:,1]] =1

x_tests = [x_test8, x_test7, x_test6]

##### Train Softmax Network ####################################################
"""
# during training 5% of training_data is used for validation
models, histories = util.fit_models(training_data, number_of_epochs=10, batch_size=2048, plot=False)
"""

##### Model Stats ##############################################################
"""
# print stats to console
for history in histories:
    print("Categorical Accuracy:")
    print(history.history['categorical_accuracy'])
    print("Validation Categorical Accuracy:")
    print(history.history['val_categorical_accuracy'])
    print("Loss:")
    print(history.history['loss'])
    print("Validation Loss:")
    print(history.history['val_loss'])


# save models to reuse later
model_names = ['soft_100k_8l.h5', 'soft_100k_7l.h5', 'soft_100k_6l.h5']
for model,n in zip(models, np.arange(len(model_names))):
    name = model_names[n]
    model.save(name)
    print("save {0:d} successfull".format(int(n)))
"""

##### Load Models for Testing ##################################################

models = []
number_of_models = 3
model_names = ['soft_100k_8l.h5', 'soft_100k_7l.h5', 'soft_100k_6l.h5']

for m in np.arange(number_of_models):
    models.append(load_model(model_names[m]))


##### Test Model ###############################################################

for model, m in zip(models, np.arange(len(models))):
    y_pred = model.predict(x_tests[m] , batch_size=32, verbose=0)

    # coordinates of most likely move:
    which_move=np.argmax(y_pred, axis=1)
    # "probability" of this move:
    how_likely=np.amax(y_pred, axis=1)

    # accuracy: How much percent of moves is correctly predicted
    acc_test = np.mean(which_move == np.argmax(y_test, axis=1))
    print('test accuracy: {0}'.format(acc_test))

    """
    # print predicted moves and correct next moves
    for i in np.arange(which_move.shape[0]):
        print("Predicted move: ({0:d}, {1:d}) with prob: {2:4f}; Next move: ({3:d}, {4:d})".format(int(which_move[i]/19), int(which_move[i]%19), how_likely[i], int(np.argmax(y_test[i])/19), int(np.argmax(y_test[i])%19)))
    """
    
    # Visualize predictions of network
    """
    plt.figure(2)
    for i in np.arange(y_pred.shape[0]):
        plt.imshow(np.reshape(y_pred[i],(19,19)), interpolation='nearest') #board_positions[i]), cmap = "Greys")
        plt.pause(1)
        plt.imshow(np.reshape(y_test[i], (19,19)), interpolation='nearest') #board_positions[i]), cmap = "Greys_r")
        plt.pause(1)
        plt.clf()
    plt.show()
    """
