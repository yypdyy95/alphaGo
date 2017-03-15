import matplotlib.pyplot as plt
import numpy as np
#from sklearn.neural_network import MLPClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution1D, Convolution2D
from keras.layers.core import Reshape, Flatten
from matplotlib.colors import LinearSegmentedColormap
import time

##### Loading Data #############################################################
# either read all files seperatly, to use single years for training/ validation
# or simply use whole dataset or testset containing games of all years together
data2001 = './data/KGS-2001-19-2298-train10k.dat'
data2002 = './data/KGS-2002-19-3646-train10k.dat'
data2003 = './data/KGS-2003-19-7582-train10k.dat'
data2004 = './data/KGS-2004-19-12106-train10k.dat'
data2005 = './data/KGS-2005-19-13941-train10k.dat'
data2006 = './data/KGS-2006-19-10388-train10k.dat'
data2007 = './data/KGS-2007-19-11644-train10k.dat'

testset = open('./data/kgsgo-test.dat', "rb")
dataset = open('./data/kgsgo-test.dat', "rb")#'train10k.dat', "rb")

'''
goboard1 = open(data2001, "rb")
goboard2 = open(data2002, "rb")
goboard3 = open(data2003, "rb")
goboard4 = open(data2004, "rb")
goboard5 = open(data2005, "rb")
goboard6 = open(data2006, "rb")
goboard7 = open(data2007, "rb")

pos1 = list(goboard1.read())
pos2 = list(goboard2.read())
pos3 = list(goboard3.read())
pos4 = list(goboard4.read())
pos5 = list(goboard5.read())
pos6 = list(goboard6.read())
pos7 = list(goboard7.read())


goboard1.close()
goboard2.close()
goboard3.close()
goboard4.close()
goboard5.close()
goboard6.close()
goboard7.close()
'''
# transform data to numpy array
# for test data change filename here
pos = np.array(list(dataset.read()))

# single moves are stored as: 2 bytes GO, 2 bytes next move, 19*19 = 361 Bytes Board
bytes_per_move = 365
# there are actually 3 more elements in pos array ('E' 'N' 'D' at the end of a file)
# due to rounding the following gives correct number of moves
number_of_moves = int(pos.shape[0]/bytes_per_move)

print ("loaded %d board positions" % number_of_moves )
# array of all moves from file as 1D arrays:
go_game = np.zeros((number_of_moves, bytes_per_move))

for move in np.arange(number_of_moves):
    go_game[move] = pos[(move*bytes_per_move):((move+1)*bytes_per_move)]


# get coordinates of next move (stored in two bytes in front of every new board position)
next_move = (go_game[:, 2:4]).astype(int)

go_game = go_game[:, 4:] #discard first 4 entries with GO, label for next move

# properties of every field on go board are stored in single bits -> acces bits via unpackbits
go_game_bits = np.unpackbits(go_game.astype(np.uint8), axis=1)
# reshpaing training data as 19*19 array with 8 entries each -> needed to apply Convolution
td = go_game_bits.copy()
training_data_full = np.reshape(td.flatten(), (number_of_moves, 19 ,19, 8 ))

target_vectors = np.zeros((number_of_moves, 19 * 19))

target_vectors[np.arange(number_of_moves), 19 * next_move[:,0] + next_move[:,1]] = 1
labels = np.reshape(target_vectors, (number_of_moves, 19 * 19))
#target_vectors[np.arange(number_of_moves,number_of_moves), 19 * next_move[:,0] + next_move[:,1]] = 1

#########################################################################################
# Training:

# which model should be used? (all models are pretrained)
# options: 'CRS_fullinfo.h5' -> Convolutional(16 filter, 5*5), ReLu, softmax
'''
model.add(Convolution2D(4,5,border_mode = 'same', input_shape = (8,19,19)))
model.add(Flatten())
model.add(Dense(19*19, init = 'uniform', activation = 'softmax'))
model.add(Reshape((19,19)))
'''

#          'clark_et_all_fullinfo_network.h5' -> 4* Convolutional, 1 softmax (note: use full data for training)
#          'C32RS_fullinfo.h5' -> same as CRS, but 1 Filter instead of 4

used_model = './networks/CS_fullinfo.h5'#'./networks/C32S_fullinfo.h5'

model = load_model(used_model)

'''
model = Sequential()
model.add(Convolution2D(48,5,5,border_mode = 'same', input_shape = (19,19,8)))

model.add(Convolution2D(32,5,5,border_mode = 'same'))
model.add(Convolution2D(32,3,3,border_mode = 'same'))
model.add(Convolution2D(8,3,3,border_mode = 'same'))
model.add(Flatten())
#model.add(Dense(19*19, init = 'uniform', activation = 'relu'))
model.add(Dense(19*19, init = 'uniform', activation = 'softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print("Number of parameters: ", model.count_params())

hist = model.fit(training_data_full,labels, nb_epoch=25, batch_size=128*4, validation_split = 0.1)

# store training results in same file:
model.save(used_model)
########################################################################################
#plotting training history: (history is returned as dictionary by keras fit function)

fig = plt.figure(figsize = (8,8))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training History")
#plt.plot(hist.history['val_acc'], label ='validation accuracy')
plt.plot(hist.history['loss'], label ='training loss')
plt.plot(hist.history['val_loss'], label ='validation loss')
#plt.plot(hist.history['acc'], label ='training accuracy')

plt.legend(loc = 'best')
'''
###########################################################################################
# more evaluation:
# check number of correctly predicted moves on full data set
# stop time, to get time for one prediction

start_time = int(round(time.time() * 1000))
predictions = model.predict(training_data_full, verbose = 1)
sim_time = int(round(time.time() * 1000))-start_time
print("takes ", (sim_time/number_of_moves), "milliseconds to predict move" )
predictions.reshape(number_of_moves, 19,19)

acc = 0
for i in range(number_of_moves):
    # field with highest accuracy in result -> predicted move
    predicted_move = np.argmax(predictions[i])
    predicted_move = [int(predicted_move/19), predicted_move % 19]
    #print ("real move: ", next_move[i], "  predicted move: ", predicted_move)
    if next_move[i][0] == predicted_move[0] and predicted_move[1] == next_move[i][1]:
        acc +=1

print ( acc/number_of_moves*100 , "percent of moves predicted correctly on training and validation set" )
###########################################################################################
# same steps as beforr for testset to get accuracy on testset

test = np.array(list(testset.read()))

number_of_moves = int(test.shape[0]/bytes_per_move)
print(number_of_moves)
# array of all moves from file as 1D arrays:
go_game = np.zeros((number_of_moves, bytes_per_move))

for move in np.arange(number_of_moves):
    go_game[move] = test[(move*bytes_per_move):((move+1)*bytes_per_move)]

# get coordinates of next move (stored in two bytes in front of every new board position)
next_move = (go_game[:, 2:4]).astype(int)

go_game = go_game[:, 4:] #discard first 4 entries with GO, label for next move

# properties of every field on go board are stored in single bits -> acces bits via unpackbits
go_game_bits = np.unpackbits(go_game.astype(np.uint8), axis=1)
# reshpaing training data as 19*19 array with 8 entries each -> needed to apply Convolution
testdata = go_game_bits.copy()
training_data_full = np.reshape(testdata.flatten(), (number_of_moves, 19 ,19, 8 ))

predictions = np.array(model.predict(training_data_full, verbose = 1))

predictions.reshape(number_of_moves, 19,19)

acc = 0
for i in range(number_of_moves):
    # field with highest accuracy in result -> predicted move
    predicted_move = np.argmax(predictions[i])
    predicted_move = [int(predicted_move/19), predicted_move % 19]
    #print ("real move: ", next_move[i], "  predicted move: ", predicted_move)
    if next_move[i][0] == predicted_move[0] and predicted_move[1] == next_move[i][1]:
        acc +=1

print ( acc/number_of_moves*100 , "percent of moves predicted correctly on training and validation set" )

plt.show()
