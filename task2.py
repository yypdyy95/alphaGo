import matplotlib.pyplot as plt
import numpy as np
#from sklearn.neural_network import MLPClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution1D
from keras.layers.core import Reshape
from matplotlib.colors import LinearSegmentedColormap
import time

##### Loading Data #############################################################
# available data sets:
data2001 = './data/KGS-2001-19-2298-test.dat'
data2002 = './data/KGS-2002-19-3646-train10k.dat'
data2003 = './data/KGS-2003-19-7582-train10k.dat'
data2004 = './data/KGS-2004-19-12106-train10k.dat'
data2005 = './data/KGS-2005-19-13941-train10k.dat'
data2006 = './data/KGS-2006-19-10388-train10k.dat'
data2007 = './data/KGS-2007-19-11644-train10k.dat'

goboard = open(data2005, "rb")
goboard2 = open(data2004, "rb")
goboard3 = open(data2001, 'rb')
pos1 = list(goboard.read())
pos2 = list(goboard2.read())
pos3 = list(goboard3.read())
# transform data to numpy array
#pos = np.concatenate((pos1[:len(pos1)-3], pos2))
#pos = np.concatenate(pos1[:len(pos1)-3], pos2)
pos = np.array(pos3)
# single moves are stored as: 2 bytes GO, 2 bytes next move, 19*19 = 361 Bytes Board
bytes_per_move = 365
number_of_moves = int(pos.shape[0]/bytes_per_move)

# array of all moves from file:
go_game = np.zeros((number_of_moves, bytes_per_move))

for move in np.arange(number_of_moves):
    go_game[move] = pos[(move*bytes_per_move):((move+1)*bytes_per_move)]

# get coordinates of next move
next_move = (go_game[:, 2:4]).astype(int)

go_game = go_game[:, 4:] #discard first 4 entries with GO, label for next move

# on final (plotable) go board store data as follows : own stone 1, enemy stone 0,
#                                                      empty field 0.5
go_game_bits = np.unpackbits(go_game.astype(np.uint8), axis=1)
#print(len(go_game_bits), len(go_game_bits[0]))
training_data_full = np.reshape(go_game_bits.copy(), (number_of_moves, 19 * 19 * 8))
#print (len(training_data_full), len(training_data_full[0]), len(training_data_full[0][0]))
go_game_bits = np.reshape(go_game_bits, (go_game_bits.shape[0], -1, 8))

go_game_plot = np.zeros_like(go_game)
#go_game_plot +=

#check which bits are 1, depending on which bit might be 1 add/subtract 0.5
# if own stone on field one of bits 2-5 will be 1 -> add 0.5,
# enemy stone on field -> bit 6-8 will be 1 -> subtract 0.5
for i in np.arange(2,5):
    go_game_plot +=go_game_bits[:,:,i]
for i in np.arange(5,8):
    go_game_plot -= go_game_bits[:,:,i]

print ("loaded %d board positions" % number_of_moves )

go_game_plot = np.reshape(go_game_plot, (number_of_moves, 19, 19))

# split training and validation data
number_of_training_positions = int(0.01* number_of_moves)
training_data = go_game_plot[:number_of_training_positions]
validation_data = go_game_plot[number_of_training_positions:]

# target vectors "empty go boards" with 1 at position if next move
target_vectors = np.zeros((number_of_moves, 19 * 19))
target_vectors[np.arange(number_of_moves), 19 * next_move[:,0] + next_move[:,1]] = 1
# bring them in the right dimension
training_labels = np.reshape(target_vectors[:number_of_training_positions], (number_of_training_positions,19,19))
validation_labels = np.reshape(target_vectors[number_of_training_positions:], (number_of_moves - number_of_training_positions,19,19))

used_model = 'task2_test_network.h5'

#model = load_model(used_model)


model = Sequential()
#model.add(Convolution1D( 19, 5, border_mode='same', input_shape=(19,19*8) ))

model.add(Dense(19*19, init = 'uniform', input_dim = (19*19*8), activation = 'softmax'))
model.add(Reshape((19,19)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

hist = model.fit(training_data_full[:number_of_training_positions],training_labels, nb_epoch=1000, batch_size=1, validation_data = (training_data_full[number_of_training_positions:],validation_labels ))

model.save(used_model)

start_time = int(round(time.time()))
predictions = model.predict(training_data_full, verbose = 1)
sim_time = int(round(time.time() * 1000))-start_time
print("takes ", (sim_time/number_of_moves), "milliseconds to predict move" )

predictions.reshape(number_of_moves, 19,19)


acc = 0

for i in range(number_of_moves):
    predicted_move = np.argmax(predictions[i])
    predicted_move = [int(predicted_move/19), predicted_move % 19]
    if next_move[i][0] == predicted_move[0] and predicted_move[1] == next_move[i][1]:
        acc +=1

print ( acc/number_of_moves*100 , "percent of moves predicted correctly" )

fig = plt.figure(figsize = (8,8))
plt.xlabel("Epoch")
plt.title("Training History for rollout network")
#plt.plot(hist.history['val_acc'], label ='validation accuracy')
plt.plot(hist.history['loss'], label ='training loss')
plt.plot(hist.history['val_loss'], label ='validation loss')
#plt.plot(hist.history['acc'], label ='training accuracy')

plt.legend(loc = 'best')

plt.show()
