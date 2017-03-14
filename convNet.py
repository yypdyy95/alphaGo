import matplotlib.pyplot as plt
import numpy as np
#from sklearn.neural_network import MLPClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution1D, Convolution2D,Flatten
import utilities as util
import time
import gc
import keras.backend as K
K.set_image_dim_ordering('th')


##### Load Data ################################################################
#number_of_moves, next_move, bit_board = util.load_data("./data/KGS-2004-19-12106-train10k.dat", max_moves=100000)
number_of_moves, next_move, bit_board = util.load_data("./data/kgsgo-train10k.dat")


##### Generate Training Data ###################################################
number_of_planes = 3
x_train = np.zeros((number_of_moves, number_of_planes, 19, 19))
for move in np.arange(number_of_moves):
    x_train[move] = util.get_board(bit_board[move])

y_train = np.zeros((number_of_moves, 19*19))
y_train[np.arange(number_of_moves), 19*next_move[:,0]+next_move[:,1]] =1

print(x_train.shape)
del bit_board
del next_move
gc.collect()
##### Train Network ############################################################
model_name = './networks/CNN_pos_100k.h5'

model = Sequential()
model.add(Convolution2D(32, 3,3, input_shape=(3, 19, 19), border_mode='same', activation='relu'))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(361, init = 'uniform', activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, nb_epoch=10, batch_size=4096, validation_split = 0.05)

model.save(model_name)
