import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from keras.models import Sequential
from keras.layers import Dense, Activation
import operator
import gc

def load_data(path, max_moves=1e12):
    """
    takes dataset from path and returns number of moves, the bitwise board
    position and the next move for each postition

    Inputs:
    - path: String with path to datasets downloaded via Tool from
    https://github.com/hughperkins/kgsgo-dataset-preprocessor
    - max_moves: maximum number of moves to return, rest of dataset is discarded

    Returns:
    - number_of_moves: number of moves
    - next_move: np.array of shape (number_of_moves, 2) with next move to each
      board position
    - go_game_bits: np.array of shape (number_of_moves, 361, 8) where for each
      move and position on the field there are 8 binary features characterizing
      the stone (https://github.com/hughperkins/kgsgo-dataset-preprocessor
      for more info)

    """

    # write data in position list

    # NOTE: even if maximum number of moves is less then number of samples in file,
    #       whole file is read, should be changed to save some time
    data = open(path, "rb")
    position_list = list(data.read())
    data.close()
    position_list = np.array(position_list)
    bytes_per_move = 365 #19x19 + 2 for next move +2 for 'GO'
    number_of_moves = int(position_list.shape[0]/bytes_per_move)
    number_of_moves = min(number_of_moves, max_moves)

    # create numpy array with one row per move
    go_game = np.zeros((number_of_moves, bytes_per_move))
    for move in np.arange(number_of_moves):
        go_game[move] = position_list[(move*bytes_per_move):((move+1)*bytes_per_move)]

    # extract next move from array
    next_move = (go_game[:, 2:4]).astype(int)
    # discard first 4 entries with 'GO', label for next move
    go_game = go_game[:, 4:]

    # create 3d array with ints converted to bits
    go_game_bits = np.unpackbits(go_game.astype(np.uint8), axis=1)
    go_game_bits = np.reshape(go_game_bits, (number_of_moves, -1, 8))

    #clean up
    del go_game
    del position_list
    gc.collect()

    return number_of_moves, next_move, go_game_bits

def get_board(board_position):
    """
    Encode own stones, opponents stones and empty positions on separate planes

    Inputs:
    - board_position: np.array of shape (361, 8) encoding all binary features
      of one particular board_position

    Returns:
    - planes: np.array of shape (3, 19, 19) containing three planes:
        - 1st plane: binary plane with 1 at own stones
        - 2nd plane: binary plane with 1 at opponents stones
        - 3rd plane: binary plane with 1 at empty positions

    """

    planes = np.zeros((3, 19*19))
    for i in np.arange(5,8): # own stones
        planes[0, board_position[:,i] == 1] += 1   # own stone
    for i in np.arange(2,5): # own stones
        planes[1, board_position[:,i] == 1] += 1   # opponent stone
    mask = np.logical_and(planes[0,:]==0, planes[1,:]==0)
    planes[2, mask] = 1 # empty space
    planes = np.reshape(planes, (3, 19, 19))
    return planes

def get_liberties(board_position):
    """
    Encode 1,2 and 3+ liberties on separate planes

    Inputs:
    - board_position: np.array of shape (361, 8) encoding all binary features
      of one particular board_position

    Returns:
    - planes: np.array of shape (3, 19, 19) containing three planes:
        - 1st plane: 1 at stone with 1 liberty
        - 2nd plane: 1 at stone with 2 liberties
        - 3rd plane: 1 at stone with 3+ liberties

    """

    planes = np.zeros((3, 19*19))
    for i in [4,7]: # 1 liberty
        planes[0,board_position[:,i]==1] +=1
    for i in [3,6]: # 2 liberties
        planes[1,board_position[:,i]==1] +=1
    for i in [2,5]: # 3+ liberties
        planes[2,board_position[:,i]==1] +=1
    planes = np.reshape(planes, (3, 19, 19))
    return planes

def get_neighbors(board_position, prev_move):
    """
    Encodes neighbors of previous move on a planes

    Inputs:
    - board_position: np.array of shape (361, 8) encoding all binary features
      of one particular board_position
    - prev_move: np.array of shape (2,) containing x,y-coordinates to prev move

    Returns:
    - neighbors: np.array of shape (1, 19, 19) containing one plane with 1 at 8
      sourrounding stones of previous move

    """

    neighbors = np.zeros((19,19))
    # if prev move on border -> do not calc neighbors
    if max(prev_move[0], prev_move[1]) >17 or min(prev_move[0], prev_move[1]) <1:
        return np.reshape(neighbors, (1, 19,19))

    # create all permutations of x and y shifts:
    shift = [-1, 0, 1]
    x_shift, y_shift = np.meshgrid(shift, shift)
    x_shift = np.reshape(x_shift, (x_shift.shape[0]*x_shift.shape[1]))
    y_shift = np.reshape(y_shift, (y_shift.shape[0]*y_shift.shape[1]))
    for (dx,dy) in zip(x_shift, y_shift):
        neighbors[prev_move[0]+dx, prev_move[1]+dy] =1
    # set previous move back to 0
    neighbors[prev_move[0], prev_move[1]] = 0
    neighbors = np.reshape(neighbors, (1, 19, 19))

    return neighbors

def get_plotable(board_positions):
    '''
    gets go Board in the form of: black stone: -1, vacant field 0, white stone 1
    in one layer

    Inputs:
    - board_position: np.array of shape (361, 8) encoding all binary features
      of one particular board_position
    Returns:
    - plotable: np.array of shape (1,19,19) with above encoding of the board
    '''
    raw_board = np.zeros((19*19))
    # check which bits are 1, depending on which bit might be 1 add/subtract 0.5
    # if own stone on field one of bits 2-5 will be 1 -> add 0.5,
    # enemy stone on field -> bit 6-8 will be 1 -> subtract 0.5
    for i in np.arange(2,5):
        raw_board +=  board_positions[:,i]
    for i in np.arange(5,8):
        raw_board -=  board_positions[:,i]

    return np.reshape(raw_board,(19,19))

def get_training_data(board_positions, number_of_moves, next_move, number_of_planes):
    '''
    gets training data in desired representation (1,3,7 or 8 layers)

    Inputs:
    - board_position: np.array of shape (361, 8) encoding all binary features
      of one particular board_position
    - number_of_moves: number of board_positions
    - next move: array returned from load data, needed for encoding liberties
    Returns:
    - training data as numpy array with shape (number_of_moves, number_of_planes, 19, 19)
    '''
    x_train = np.zeros((number_of_moves, number_of_planes, 19, 19))
    print()
    if number_of_planes == 7:
        for move in np.arange(number_of_moves):
            planes_pos = get_board(board_positions[move])
            planes_lib = get_liberties(board_positions[move])
            plane_neighbors = get_neighbors(board_positions[move], next_move[move-1])
            x_train[move] = np.concatenate((planes_pos, planes_lib, plane_neighbors), axis=0)

    elif number_of_planes == 8:
        x_train = board_positions.reshape(number_of_moves,8,19,19)

    elif number_of_planes == 3:
        for move in np.arange(number_of_moves):
            x_train[move] = get_board(board_positions[move])
            #print(x_train[move,1,5:15,5:15])


    elif number_of_planes == 1:
        for move in np.arange(number_of_moves):
            x_train[move] = get_plotable(board_positions[move])

    return x_train


def plot_board(board_positions, next_move=None, ko=False, delay_time=0.2):
    """
    plots board_positions with delay of delay_time, if next_move is given, next
    move is highlighted blue, if ko=True, forbidden ko-fields will be marked
    yellow

    Inputs:
    - board_positions: np.array of shape (number_of_moves, 361, 8) encoding all binary features
      of all board_positions in dataset
    - next_move: np.array of shape (number_of_moves, 2),
    - ko: boolean, if ko=True, forbidden ko-fields are marked yellow
    - delay_time: float, timeintervall one board will be shown (in sec.)

    Returns:

    """

    #create custom colormaps: alternating with each move as color of 'own' and 'opponent' changes
    color_list = [(0.0, 'white'),(0.5, 'peru') ,(0.7, 'blue'),(0.8, "yellow"),(1, 'black')]
    color_list_inverted = [(0.0, 'black'),(0.5, 'peru') ,(0.7, 'blue'), (0.8, "yellow"), (1, 'white')]
    go_color = LinearSegmentedColormap.from_list('go_color', color_list )
    go_color_inverted = LinearSegmentedColormap.from_list('go_color_inv', color_list_inverted )

    number_of_moves = board_positions.shape[0]
    go_game_plot = np.zeros((number_of_moves,19*19))
    go_game_plot += 0.5
    for i in np.arange(2,5): #opponents stones
        go_game_plot +=0.5*board_positions[:,:,i]
    for i in np.arange(5,8): #own stones
        go_game_plot -=0.5*board_positions[:,:,i]

    if ko == True:
        # ko fields:
        mask = board_positions[:,:, 1]==1
        go_game_plot[mask] =0.8

    # reshape to format of GO-board
    go_game_plot = np.reshape(go_game_plot, (number_of_moves, 19, 19))

    # mark next move
    if next_move != None:
        go_game_plot[np.arange(number_of_moves), next_move[:,0], next_move[:,1]] = 0.7

    gc.collect()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # create lines of Go-Board
    major_ticks = np.array([0, 3, 9, 15, 18])
    minor_ticks = np.arange(0, 18)

    if number_of_moves > 1:
        for move in np.arange(number_of_moves):

            # set lines on go board
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)
            ax.grid(which='minor', linestyle='-', linewidth=0.3)
            ax.grid(which='major', linestyle='-', linewidth=1)

            # create dots on intersections of major lines
            intersec_x, intersec_y = np.meshgrid(major_ticks[1:4], major_ticks[1:4])
            ax.scatter(np.reshape(intersec_x, (intersec_x.shape[0]*intersec_x.shape[1])),\
            np.reshape(intersec_y, (intersec_x.shape[0]*intersec_x.shape[1])),marker='o',\
            edgecolors='none', facecolors= 'black', s=40 )

            if move%2==0:
                ax.imshow(go_game_plot[move], cmap=go_color, interpolation='nearest')
            else:
                ax.imshow(go_game_plot[move], cmap=go_color_inverted, interpolation='nearest')
            ax.set_title("Move %d" %move)
            plt.pause(delay_time)
            plt.cla() #delete previous images
        #fig.show()
    else:

        # set lines on go board
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', linestyle='-', linewidth=0.3)
        ax.grid(which='major', linestyle='-', linewidth=1)

        # create dots on intersections of major lines
        intersec_x, intersec_y = np.meshgrid(major_ticks[1:4], major_ticks[1:4])
        ax.scatter(np.reshape(intersec_x, (intersec_x.shape[0]*intersec_x.shape[1])),\
        np.reshape(intersec_y, (intersec_x.shape[0]*intersec_x.shape[1])),marker='o',\
        edgecolors='none', facecolors= 'black', s=40 )

        ax.imshow(go_game_plot[0], cmap=go_color, interpolation='nearest')
        plt.show()

def plot_liberties(board_positions, delay_time=1):
    """
    plots liberties for stones on board with delay_time between board positions

    Inputs:
    - board_positions: np.array of shape (number_of_moves, 361, 8) encoding all binary features
      of all board_positions in dataset
    - delay_time: float, timeintervall one board will be shown (in sec.)

    Returns:

    """
    number_of_moves = board_positions.shape[0]

    liberties = np.zeros((number_of_moves, 361))
    for i in np.arange(2,5):
        mask = board_positions[:,:,i] == 1
        liberties[mask] = 5-i
    for i in np.arange(5,8):
        mask = board_positions[:,:,i] == 1
        liberties[mask] = i-8

    liberties = np.reshape(liberties, (number_of_moves, 19, 19))
    gc.collect()

    # rescale liberties to [0,1]
    liberties += 3
    liberties*=1/6.0

    liberty_colors = [(0.0, 'darkgreen'),(0.1667, 'yellowgreen') ,(0.3334, 'lightgreen'),\
    (0.5, "white"),(0.6667, 'darksalmon'), (0.8334, 'red' ), (1, 'firebrick' )]
    liberty_colors_inverted = [(0.0, 'firebrick' ),(0.1667, 'red') ,(0.3334, 'darksalmon'),\
    (0.5, "white"),(0.6667,'lightgreen'), (0.8334,'yellowgreen'), (1, 'darkgreen')]
    liberty_colormap = LinearSegmentedColormap.from_list('liberty_colormap', liberty_colors)
    liberty_colormap_inverted = LinearSegmentedColormap.from_list('liberty_colormap_inv', liberty_colors_inverted )

    # plot board_positions:
    for move in np.arange(number_of_moves):
        if move%2==0:
            plt.imshow(liberties[move], cmap=liberty_colormap, interpolation='nearest')
        else:
            plt.imshow(liberties[move], cmap=liberty_colormap_inverted, interpolation='nearest')
        plt.title("Move %d" %move)
        plt.pause(delay_time)
        plt.clf() #delete previous images
    plt.show()

def get_occupied_positions(board_planes): #boardplanes.shape =  3x19x19
    """
    returns a numpy array with coordinates of all stones on a board_planes

    Inputs:
    - board_planes: np.array of shape (3, 19, 19), output of get board_planes

    Returns:
    - occupied_positions: np.array of shape (:,2),
    """
    #occupied_positions = []
    occupied = board_planes[2, :, :] == 0
    #print(occupied)
    occupied_positions = np.argwhere(occupied)
    return occupied_positions

def hash_patterns(occupied_positions, board_planes, liberty_planes):
    """
    returns numpy array of hashed 3x3 patterns arround occupied_positions
    Hash:   - #1 digit gives color of middle stone
            - #8 digits for positions:
            - #8 digits for liberties:


    Inputs:
    - occupied_positions: np.array of shape (:,2),
    - board_planes: output from get_board
    - liberty_planes: output from get_liberties

    Returns:
    - pattern_hash: np.array of shape (number_of_moves) with 17digit ints in
      each component
    """
    pattern_hash = np.zeros((occupied_positions.shape[0]), dtype=int)

    #roll all planes in every direction to create 3x3 environments

    black_env = np.empty((9, 19, 19))
    black_env[0] = board_planes[0]
    black_env[1] = np.roll(board_planes[0], 1, axis=1)
    black_env[2]= np.roll(board_planes[0], -1, axis=1)
    black_env[3] = np.roll(board_planes[0], -1, axis=0)
    black_env[4] = np.roll(board_planes[0], 1, axis=0)
    black_env[5] = np.roll(black_env[3], 1, axis=1)
    black_env[6] = np.roll(black_env[3], -1, axis=1)
    black_env[7] = np.roll(black_env[4], 1, axis=1)
    black_env[8] = np.roll(black_env[4], -1, axis=1)

    white_env = np.empty((9, 19, 19))
    white_env[0] = board_planes[1]
    white_env[1] = np.roll(board_planes[1], 1, axis=1)
    white_env[2] = np.roll(board_planes[1], -1, axis=1)
    white_env[3] = np.roll(board_planes[1], -1, axis=0)
    white_env[4] = np.roll(board_planes[1], 1, axis=0)
    white_env[5] = np.roll(white_env[3], 1, axis=1)
    white_env[6] = np.roll(white_env[3], -1, axis=1)
    white_env[7] = np.roll(white_env[4], 1, axis=1)
    white_env[8] = np.roll(white_env[4], -1, axis=1)

    one_env = np.empty((9, 19, 19))
    one_env[0] = np.roll(liberty_planes[0], 1, axis=1)
    one_env[1] = np.roll(liberty_planes[0], -1, axis=1)
    one_env[2] = np.roll(liberty_planes[0], -1, axis=0)
    one_env[3] = np.roll(liberty_planes[0], 1, axis=0)
    one_env[4] = np.roll(one_env[2], 1, axis=1)
    one_env[5] = np.roll(one_env[2], -1, axis=1)
    one_env[6] = np.roll(one_env[3], 1, axis=1)
    one_env[7] = np.roll(one_env[3], -1, axis=1)

    two_env = np.empty((9, 19, 19))
    two_env[0] = np.roll(liberty_planes[1], 1, axis=1)
    two_env[1] = np.roll(liberty_planes[1], -1, axis=1)
    two_env[2] = np.roll(liberty_planes[1], -1, axis=0)
    two_env[3] = np.roll(liberty_planes[1], 1, axis=0)
    two_env[4] = np.roll(two_env[2], 1, axis=1)
    two_env[5] = np.roll(two_env[2], -1, axis=1)
    two_env[6] = np.roll(two_env[3], 1, axis=1)
    two_env[7] = np.roll(two_env[3], -1, axis=1)

    three_env = np.empty((9, 19, 19))
    three_env[0] = np.roll(liberty_planes[2], 1, axis=1)
    three_env[1] = np.roll(liberty_planes[2], -1, axis=1)
    three_env[2] = np.roll(liberty_planes[2], -1, axis=0)
    three_env[3] = np.roll(liberty_planes[2], 1, axis=0)
    three_env[4] = np.roll(three_env[2], 1, axis=1)
    three_env[5] = np.roll(three_env[2], -1, axis=1)
    three_env[6] = np.roll(three_env[3], 1, axis=1)
    three_env[7] = np.roll(three_env[3], -1, axis=1)


    # has patterns
    color_multiplier = np.array([10000000000000000, 1000000000000000, 100000000000000, 10000000000000, 1000000000000, 100000000000, 10000000000, 1000000000, 100000000])
    liberty_multiplier = np.array([10000000, 1000000, 100000, 10000, 1000, 100, 10, 1])
    for i in np.arange(9):
        mask = black_env[i ,occupied_positions[:,0], occupied_positions[:,1]] ==1
        pattern_hash[mask] += 2*color_multiplier[i]
    for i in np.arange(9):
        mask = white_env[i ,occupied_positions[:,0], occupied_positions[:,1]] ==1
        pattern_hash[mask] += 1*color_multiplier[i]
    for i in np.arange(8):
        mask = one_env[i ,occupied_positions[:,0], occupied_positions[:,1]] ==1
        pattern_hash[mask] += 1*liberty_multiplier[i]
    for i in np.arange(8):
        mask = two_env[i ,occupied_positions[:,0], occupied_positions[:,1]] ==1
        pattern_hash[mask] += 2*liberty_multiplier[i]
    for i in np.arange(8):
        mask = three_env[i ,occupied_positions[:,0], occupied_positions[:,1]] ==1
        pattern_hash[mask] += 3*liberty_multiplier[i]

    return pattern_hash



def count_patterns(pattern_hash):

    """
    return dictionary with patterns as keys and number of occurences as values

    Inputs:
    - pattern_hash: see above

    Returns:
    - dictionary of patterns with number of occurences
    """

    patterns = {}
    for pattern in pattern_hash:
        if pattern in patterns:
            patterns[pattern] +=1
        else:
            patterns[pattern] = 1
    return patterns

def plot_loss_acc(histories):
    """
    plots validation and training accuracy and loss from histories

    Inputs:
    - histories: list of history objects returned from keras fit function
    """

    number_of_models = len(histories)
    fig = plt.figure(1)

    colors = ['red', 'blue', 'green', 'orange', 'turquoise', 'darkorchid']
    if number_of_models > len(colors):
        print("More models than specified colors!")
        return

    plt.subplot(2, 1, 1)
    for hist,n in zip(histories, np.arange(number_of_models)):
        plt.plot(hist.history['loss'], label='train{:d}'.format(n), color=colors[n], linestyle='--')
        plt.plot(hist.history['val_loss'], label='val{:d}'.format(n), color=colors[n],linestyle='-' )
    plt.title('Loss history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = 'best')


    plt.subplot(2, 1, 2)
    for hist,n in zip(histories, np.arange(number_of_models)):
        plt.plot(hist.history['categorical_accuracy'], label='train{:d}'.format(n), color=colors[n], linestyle='--')
        plt.plot(hist.history['val_categorical_accuracy'], label='val{:d}'.format(n), color=colors[n], linestyle='-')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.xlabel("Epoch")
    plt.title("Training History")
    plt.legend(loc = 'best')

    #plt.tight_layout()

    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.4   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    plt.show()




def fit_models(training_data, number_of_epochs=20, batch_size=512, plot=True):
    """
    trains one-layer networks with different inputs in training_data and
    and uses plot_loss_acc to compare grapfically if 'plot'=True
    Inputs:
    - training_data: list with first element 'y_train' followed by different feature vectors
    - number_of_epochs:
    - plot: boolean, if true plot accuracy and loss of models
    Returns:
    - list with models
    """
    number_of_models = len(training_data)-1
    histories = []
    models = []

    y_train = training_data[0]
    for m in np.arange(number_of_models):
        x_train = training_data[m+1]
        input_dim = x_train.shape[1]

        model = Sequential()
        model.add(Dense(361,input_dim=input_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        hist = model.fit(x_train, y_train, nb_epoch=number_of_epochs, batch_size=batch_size, validation_split=0.05)
        histories.append(hist)
        models.append(model)

        gc.collect()
    if plot==True:
        plot_loss_acc(histories)

    return models, histories
