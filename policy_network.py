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
#number_of_moves, next_move, bit_board = util.load_data("./data/KGS-2004-19-12106-train10k.dat", max_moves=500000)
number_of_moves, next_move, bit_board = util.load_data("./data/kgsgo-test.dat", max_moves=10000)


##### Generate Training Data ###################################################

number_of_planes = 8 #3 #1 #8
x_train = util.get_training_data(bit_board, number_of_moves, next_move, number_of_planes = number_of_planes)

y_train = np.zeros((number_of_moves, 19*19))
y_train[np.arange(number_of_moves), 19*next_move[:,0]+next_move[:,1]] =1

print(x_train.shape)

del bit_board
del next_move
gc.collect()

##### Train Network ############################################################
# specify model name, to load/ save model
#model_name = './networks/C485_C325_C325_C323_C83_C11_S_.h5'#'./networks/CNN_7l_16f_380k.h5'

# this model was the most successfull in our tests:
print("\n +++++++++++++++++++++++++++++++  start training   ++++++++++++++++++++++++++++++++++++ \n")

model = Sequential()
model.add(Convolution2D(8, 5,5, input_shape=(number_of_planes, 19, 19), border_mode='same', activation='relu'))
#print(model.output_shape)
model.add(Convolution2D(32, 5,5, border_mode='same', activation='relu'))
model.add(Convolution2D(32, 5,5, border_mode='same', activation='relu'))
model.add(Convolution2D(32, 3,3, border_mode='same', activation='relu'))
#print(model.output_shape)
model.add(Convolution2D(8, 3,3, border_mode='same', activation='relu'))

model.add(Convolution2D(1, 1,1, input_shape=(number_of_planes, 19, 19),  border_mode='same', activation='relu'))

#print(model.output_shape)
model.add(Flatten())
model.add(Dense(361, init = 'uniform', activation = 'relu'))
#print(model.output_shape)
model.add(Dense(361, init = 'uniform', activation = 'softmax'))

print("Number of parameters of the model: ", model.count_params())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
history = model.fit(x_train, y_train, nb_epoch=2, batch_size=512, validation_split = 0.05)

#model.save(model_name)

util.plot_loss_acc([history])

print("Categorical Accuracy:")
print(history.history['categorical_accuracy'])
print("Validation Categorical Accuracy:")
print(history.history['val_categorical_accuracy'])
print("Loss:")
print(history.history['loss'])
print("Validation Loss:")
print(history.history['val_loss'])

print("\n ++++++++++++++++++++++++++++++++++ training finished +++++++++++++++++++++++++++++++++++ \n")
##### Test Network #############################################################
print("\n ++++++++++++++++++++++++++++++++++    test network   +++++++++++++++++++++++++++++++++++ \n")
print("Model to test: ", model_name, "\n")
model_name = './networks/CNN_pos_32f5_380k.h5'
model = load_model(model_name)


number_of_moves, next_move, bit_board = util.load_data("./data/kgsgo-test.dat")
gc.collect()

number_of_planes = 3

x_test = util.get_training_data(bit_board, number_of_moves, next_move, number_of_planes = number_of_planes)

y_test = np.zeros((number_of_moves, 19*19))
y_test[np.arange(number_of_moves), 19*next_move[:,0]+next_move[:,1]] =1

tic = time.time()
y_pred = model.predict(x_test , batch_size=32, verbose=0)
toc = time.time()
print("prediction time = {}".format((toc-tic)/number_of_moves))
# coordinates of most likely move:
which_move=np.argmax(y_pred, axis=1)
# "probability" of this move:
how_likely=np.amax(y_pred, axis=1)

# accuracy: How much percent of moves is correctly predicted
acc_test = np.mean(which_move == np.argmax(y_test, axis=1))
print('test accuracy: {0}'.format(acc_test))
