import flask
import numpy as np
import pandas as pd
import copy
import csv
import ast
import random
import pickle
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def get_num(string):
    return int(string.split("loc")[-1])

class GP:
    def __init__(self):
        num = 0
        shape = []
        cost = 0
        spaces = 0
        buttons = 0
        
class player:
    def __init__(self):
        self.loc = 0
        self.buttons = 5
        self.board_buttons = 0
        self.board = []
        for i in range(9):
            self.board.append([0] * 9)
        self.plus_seven = 0
        self.next_but = 5
            
class shared_board:
    def __init__(self):
        self.tot_spaces = 53
        self.sq_spaces = [20,26,32,44,50]
        self.but_spaces = [5,11,17,23,29,35,41,47,53]
        self.next_sq= 20

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0.00
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate),  metrics=['accuracy'] )
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state, valid_moves):
        valid_moves = recalc_valid_moves(valid_moves)
        if np.random.rand() <= self.epsilon:
            act = random.choice(valid_moves)
            return act
        act_values = self.model.predict(state)
        huer_vals = value_moves(state)
        final_values = [a+b for a,b in zip(act_values[0],huer_vals)]
        final_values = [final_values[i] for i in valid_moves]
        return valid_moves[np.argmax(final_values)]  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                  target = reward + self.gamma * np.mean(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def save(self, name):
        self.model.save_weights(name)
        pickle.dump(self.memory, open("../pw_memory.pkl","wb"))
    def load(self, name):
        self.model.load_weights(name)
        self.memory = pickle.load(open("../pw_memory.pkl","rb"))

#import peices
def create_pieces():
    pieces = []
    with open('../Patchwork_Pieces.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='.')
        for row in spamreader:
            pieces.append(row)
    p_dict = {}
    p_list = []
    for i, p in enumerate(pieces):
        if i == 0:
            pass
        else:
            p_list.append(p[0])
            p_dict[p[0]] = GP()
            p_dict[p[0]].num = int(p[0])
            p_dict[p[0]].shape = ast.literal_eval(p[1])
            p_dict[p[0]].cost = int(p[2])
            p_dict[p[0]].spaces = int(p[3])
            p_dict[p[0]].buttons = int(p[4])

    init_pieces = []
    for key in p_list:
        init_pieces.append(p_dict[key]) 
    return init_pieces

def all_perms(shape):
    rotate0 = shape
    rotate1 = [list(i) for i in zip(*rotate0[::-1])]
    rotate1 = map_shape(map_slot(rotate1))
    rotate2 = [list(i) for i in zip(*rotate1[::-1])]
    rotate2 = map_shape(map_slot(rotate2))
    rotate3 = [list(i) for i in zip(*rotate2[::-1])]
    rotate3 = map_shape(map_slot(rotate3))
    trans0 = [list(i) for i in zip(*rotate0)]
    trans1 = [list(i) for i in zip(*rotate1)]
    trans2 = [list(i) for i in zip(*rotate2)]
    trans3 = [list(i) for i in zip(*rotate3)]
    perms = [rotate0, rotate1, rotate2, rotate3, trans0, trans1, trans2, trans3]
    i = 0
    j = 0
    while i < len(perms):
        j = i+1
        while j < len(perms):
            if perms[i] == perms[j]:
                perms.pop(j)
            else:
                j = j+1
        i = i+1
    return perms

def map_shape(slots):
    slots = top_right(slots)
    shape = []
    for i in range(5):
        shape.append([0] * 5)
    for slot in slots:
        shape[slot[1]][slot[0]] = 1
    return shape

def map_slot(shape):
    slots = []
    for i in range(len(shape)):
        for j in range(len(shape[0])):
            if shape[j][i] == 1:
                slots.append([i,j])
    return slots

def top_right(input_slots):
    y_min = min([input_slots[i][1] for i in range(len(input_slots))])
    x_min = min([input_slots[i][0] for i in range(len(input_slots))])

    # move piece top left
    moved = copy.deepcopy(input_slots)
    for slot in moved:
        slot[0] -= x_min
        slot[1] -= y_min 
    return moved

def piece_shape_list(state):
    init_pieces = create_pieces()

    with open('../keys.csv', 'r') as f:
        reader = csv.reader(f)
        key_list = list(reader)
    key_list = key_list[0]

    piece_list = []
    p_shapes = []
    p_cost = []
    p_spaces = []
    p_buttons = []
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list.append(index)
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list[loc] = get_num(key_list[index+172])
    for index, piece in enumerate(piece_list):
        for gp in init_pieces:
            if piece == gp.num:
                piece_list[index] = gp
    for shape in piece_list:
        p_shapes.append([item for sublist in shape.shape for item in sublist])
        p_cost.append(shape.cost)
        p_spaces.append(shape.spaces)
        p_buttons.append(shape.buttons)

    return p_shapes, p_cost, p_spaces, p_buttons

def piece_perms(state, p):
    init_pieces = create_pieces()

    with open('../keys.csv', 'r') as f:
        reader = csv.reader(f)
        key_list = list(reader)
    key_list = key_list[0]

    piece_list = []
    piece_shapes = []
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list.append(index)
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list[loc] = get_num(key_list[index+172])
    for index, piece in enumerate(piece_list):
        for gp in init_pieces:
            if piece == gp.num:
                piece_list[index] = gp
    perms = all_perms(piece_list[p].shape)
    a = []
    for perm in perms:
        a.append([item for sublist in perm for item in sublist])
    return a

def create_initial_state():
    #import keys
    import csv
    
    with open('../keys.csv', 'r') as f:
        reader = csv.reader(f)
        key_list = list(reader)
        
    key_list = key_list[0]
    
    #create pieces
    init_pieces = create_pieces()
    #piece_list = sorted(init_pieces[:len(init_pieces)-1], key=lambda k: random.random())
    #piece_list.append(init_pieces[-1])
    new_list = []
    fixed_list = [8,2,24,28,18,3,26,23,15,25,30,4,10,32,12,27,19,9,11,22,17,16,1,14,7,5,13,6,31,29,21,20,999]
    print(len(fixed_list))
    for i in fixed_list:
        for j in init_pieces:
            if j.num == i:
                new_list.append(j)
    piece_list = new_list

    #setup intial middle board
    sb = shared_board()

    #setup players
    player1 = player()
    player2 = player()
    player_list = [player1, player2]
    
    turn = 0
    counter = 0
    
    d = build_state(player_list,turn,piece_list, counter, init_pieces, sb)
    state = []
    for key in key_list:
        state.append(d[key])
    return state

def check_pass_button(loc, player):
    if player.loc >= player.next_but:
            return True
    return False

def check_pass_square(loc, n_s):
    if loc >= n_s:
        return True
    return False

def remove_and_reorder_pieces(p2r, piece_list):
    i = piece_list.index(p2r)
    k = piece_list[i+1:]
    k.extend(piece_list[:i])
    return k

def human_calc_piece(p2p, turn, shr_brd, player_list, pi_list, p_loc, p_perm):
    if player_list[turn].buttons >= p2p.cost:
        player_list[turn].board, place_sts = human_place_piece(p2p, player_list[turn].board, p_loc, p_perm)
    else:
        place_sts=False
    if place_sts == True:
        #move player forward x spaces
        player_list[turn].loc += p2p.spaces
        if player_list[turn].loc >= shr_brd.tot_spaces:
            player_list[turn].loc = shr_brd.tot_spaces
        #remove cost of buttons from player
        player_list[turn].buttons -= p2p.cost
        
        #add board buttons if there were any on piece
        player_list[turn].board_buttons += p2p.buttons
        
        #player has 7x7 board?
        player_list = check_plus7(player_list,turn)
            
        #reorder piece list
        return remove_and_reorder_pieces(p2p, pi_list), player_list, place_sts
    else:
        return pi_list, player_list, place_sts

def check_open(input_slots, board):
    for slot in input_slots:
        if board[slot[1]][slot[0]] == 1:
            #print('Cannot put piece in that location.')
            return False
    return True

#gameplay - placing pieces
####COMPUTER PLACEMENT
def human_place_piece(piece, board, p_loc, p_perm):
    a_perms = all_perms(piece.shape)
    u_perm = p_perm
    u_loc = p_loc
    
    #check if spots are open - if so, put the piece in!
    u_slot = map_slot(a_perms[u_perm])
    for slot in u_slot:
        slot[0] += u_loc[0]
        slot[1] += u_loc[1]

    if check_open(u_slot, board) == True:
        for slot in u_slot:
            board[slot[1]][slot[0]] = 1
    else:
        return (board, False)

    return (board, True)

def comp_place_square(board):
    max_edges = 0
    loc = [0,0]
    touch_slots = [[1,0],[-1,0],[0,-1],[0,1]]
    
    for index, i in enumerate(board):
        for jndex in range(len(board)):
            open_flag = True
            total_edges = 0
            if board[index][jndex] == 1:
                    open_flag = False
            if open_flag==True:
                for s in touch_slots:
                    if jndex+s[0]<0 or jndex+s[0]>=len(board):
                        total_edges += 1
                    elif index+s[1]<0 or index+s[1]>=len(board):
                        total_edges += 1
                    else:
                        total_edges += board[index+s[1]][jndex+s[0]]
            if total_edges > max_edges:
                max_edges = total_edges
                loc = [jndex,index]
            
    board[loc[1]][loc[0]] = 1
    return(board)

def check_plus7(players, turn):
    square_size = 7
    for p in players:
        if p.plus_seven == 1:
            return players
    
    for i in range(len(players[turn].board)-square_size+1):
        for j in range(len(players[turn].board)-square_size+1):
            l = []
            for k in range(i, i+square_size):
                for m in range(j, j+square_size):
                    l.append(players[turn].board[k][m])
            if l[0] == 1:
                if len(set(l)) == 1:
                    players[turn].plus_seven = 1
                    return players
    return players
    
def score_game(players):
    score = [0,0]
    for index, p in enumerate(players):
        #total buttons
        score[index] += p.buttons
        
        #total empty squares
        empty_sq = 0
        for i in p.board:
            for j in i:
                if j == 0:
                    empty_sq +=1
        score[index] += (empty_sq * -2)
        
        #plus 7
        score[index] += p.plus_seven*7
    
    return score

def check_winner(score):
    if score[0] > score[1]:
        return 1
    elif score[0] == score[1]:
        return 0
    else:
        return -1
def comp_place_piece(piece, board):
    loc, perm, valid = heuristic_loc(piece, board)
    a_perms = all_perms(piece.shape)
    u_perm = perm
    u_loc = loc
    
    #check if spots are open - if so, put the piece in!
    u_slot = map_slot(a_perms[u_perm])
    for slot in u_slot:
        slot[0] += u_loc[0]
        slot[1] += u_loc[1]

    if check_open(u_slot, board) == True and valid==True:
        for slot in u_slot:
            board[slot[1]][slot[0]] = 1
    else:
        return (board, False)

    return (board, True)

def comp_check_open(input_slots, board):
    total_edges = 0
    touch_slots = [[1,0],[-1,0],[0,-1],[0,1]]
    open_flag = True
    for slot in input_slots:
        if slot[1]>8 or slot[1]<0 or slot[0]>8 or slot[0]<0:
            open_flag=False
        try:
            if board[slot[1]][slot[0]] == 1:
                open_flag = False
        except:
            open_flag=False
    if open_flag==True:
        total_edges = 0
        for slot in input_slots:
            for s in touch_slots:
                if slot[0]+s[0]<0 or slot[0]+s[0]>=len(board):
                    total_edges += 1
                elif slot[1]+s[1]<0 or slot[1]+s[1]>=len(board):
                    total_edges += 1
                else:
                    total_edges += board[slot[1]+s[1]][slot[0]+s[0]]
    return total_edges, open_flag

def heuristic_loc(piece,board):
    max_edges = 0
    valid = False
    out_valid=False
    placement = [[0,0],0]
    a_perms = all_perms(piece.shape)
    for index, row in enumerate(board):
        for jndex, j in enumerate(row):
            for k, perm in enumerate(a_perms):
                u_slot = map_slot(a_perms[k])
                for slot in u_slot:
                    slot[0] += index
                    slot[1] += jndex
                    
                cur_edges, valid = comp_check_open(u_slot, board)
                if valid is True:
                    if cur_edges > max_edges:
                        max_edges = cur_edges
                        out_valid=True
                        placement[0][0] = index
                        placement[0][1] = jndex
                        placement[1] = k
    return placement[0], placement[1], out_valid

def check_place_piece(piece, board):
    loc, perm, valid = heuristic_loc(piece, board)
    a_perms = all_perms(piece.shape)
    u_perm = perm
    u_loc = loc
    
    #check if spots are open - if so, put the piece in!
    u_slot = map_slot(a_perms[u_perm])
    for slot in u_slot:
        slot[0] += u_loc[0]
        slot[1] += u_loc[1]

    if check_open(u_slot, board) == True and valid == True:
        return (True)
    else:
        return (False)

def recalc_valid_moves(valid_moves):
    out = []
    for index, val in enumerate(valid_moves):
        if val == 1:
            out.append(index)
    return out

def make_move_human(state, action, p_perm, p_loc):
    ###action is simply [0 to 3] - 0 is just move piece
    ###1-3 is pick up piece 1-3... hueristic determines placement, etc
    
    player1 = player()
    player2 = player()
    player_list = [player1, player2]
    
    ######take in state - assign necessary stuff to vars#####
    #read in keys for the state
    with open('../keys.csv', 'r') as f:
        reader = csv.reader(f)
        key_list = list(reader)
    key_list = key_list[0]
    
    #set turn
    turn = state[171]
    
    #build players
    for i in range(0,81):
        y = int(i/9)
        x = i % 9
        player_list[0].board[y][x] = state[i]
    player_list[0].loc = state[83]
    player_list[0].buttons = state[81]
    player_list[0].board_buttons = state[84]
    player_list[0].plus_seven = state[82]
    player_list[0].next_but = state[206]
    for i in range(85,166):
        k = i-85
        y = int(k/9)
        x = k%9
        player_list[1].board[y][x] = state[i]
    player_list[1].loc = state[168]
    player_list[1].buttons = state[166]
    player_list[1].board_buttons = state[169]
    player_list[1].plus_seven = state[167]
    player_list[1].next_but = state[207]
    
    #build shared board
    sb = shared_board()
    sb.next_sq = state[208]
        
    #create pieces
    init_pieces = create_pieces()
    
    #build piece order
    #there is a smarter way to do this
    piece_list = []
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list.append(index)
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list[loc] = get_num(key_list[index+172])
    for index, piece in enumerate(piece_list):
        for gp in init_pieces:
            if piece == gp.num:
                piece_list[index] = gp 
    #counter
    counter = state[205]
    
    ######make move######
    #action 0 is just moving past the other player
    if action == 0:
        play_status=True
        move_spaces = player_list[1-turn].loc - player_list[turn].loc + 1
        player_list[turn].loc += move_spaces
        player_list[turn].buttons += move_spaces
        if player_list[turn].loc >= sb.tot_spaces:
            player_list[turn].loc = sb.tot_spaces
        #check if user passes button collect on board, if so collect buttons
        if check_pass_button(player_list[turn].loc, player_list[turn]) == True:
            player_list[turn].buttons += player_list[turn].board_buttons
            for space in sb.but_spaces:
                if player_list[turn].next_but < space:
                    player_list[turn].next_but = space
                    break
        if check_pass_square(player_list[turn].loc,sb.next_sq) == True:
            player_list[turn].board = comp_place_square(player_list[turn].board) 
            end = True
            for space in sb.sq_spaces:
                if player_list[turn].loc < space:
                    sb.next_sq = space
                    end = False
                    break
            if end == True:
                sb.next_sq = 999
    else:
        p2p = piece_list[action-1]
        piece_list, player_list, play_status = human_calc_piece(p2p, 
                                                          turn, 
                                                          sb, 
                                                          player_list, 
                                                          piece_list, p_loc, p_perm)
        if check_pass_button(player_list[turn].loc, player_list[turn]) == True:
            player_list[turn].buttons += player_list[turn].board_buttons
            for space in sb.but_spaces:
                if player_list[turn].next_but < space:
                    player_list[turn].next_but = space
                    break
        if check_pass_square(player_list[turn].loc,sb.next_sq) == True:
            player_list[turn].board = comp_place_square(player_list[turn].board) 
            end=True
            for space in sb.sq_spaces:
                if player_list[turn].loc < space:
                    sb.next_sq = space
                    end=False
                    break
            if end == True:
                sb.next_sq = 999
    
    game_end = False
    #calculate reward
    #check if game is over - if not, reward is 0
    if play_status == True:
        if player_list[turn].loc + player_list[1-turn].loc <sb.tot_spaces*2:
            reward = 0
            game_end=False
        else:
            scores = score_game(player_list)
            reward = check_winner(scores)
            game_end=True
    else:
        game_end=False
        reward = -999
        
    if player_list[turn].loc > player_list[1-turn].loc:
        turn = 1-turn
    counter += 1

    #convert to new state
    d = build_state(player_list,turn,piece_list,counter,init_pieces, sb)
    state = []
    for key in key_list:
        state.append(d[key])
    return state, reward, game_end

def comp_calc_piece(p2p, turn, shr_brd, player_list, pi_list):
    if player_list[turn].buttons >= p2p.cost:
        player_list[turn].board, place_sts = comp_place_piece(p2p, player_list[turn].board)
    else:
        place_sts=False
    if place_sts == True:
        #move player forward x spaces
        player_list[turn].loc += p2p.spaces
        if player_list[turn].loc >= shr_brd.tot_spaces:
            player_list[turn].loc = shr_brd.tot_spaces
        #remove cost of buttons from player
        player_list[turn].buttons -= p2p.cost
        
        #add board buttons if there were any on piece
        player_list[turn].board_buttons += p2p.buttons
        
        #player has 7x7 board?
        player_list = check_plus7(player_list,turn)
            
        #reorder piece list
        return remove_and_reorder_pieces(p2p, pi_list), player_list, place_sts
    else:
        return pi_list, player_list, place_sts

def make_move(state, action):
    ###action is simply [0 to 3] - 0 is just move piece
    ###1-3 is pick up piece 1-3... hueristic determines placement, etc
    
    player1 = player()
    player2 = player()
    player_list = [player1, player2]
    
    ######take in state - assign necessary stuff to vars#####
    #read in keys for the state
    with open('../keys.csv', 'r') as f:
        reader = csv.reader(f)
        key_list = list(reader)
    key_list = key_list[0]
    
    #set turn
    turn = state[171]
    
    #build players
    for i in range(0,81):
        y = int(i/9)
        x = i % 9
        player_list[0].board[y][x] = state[i]
    player_list[0].loc = state[83]
    player_list[0].buttons = state[81]
    player_list[0].board_buttons = state[84]
    player_list[0].plus_seven = state[82]
    player_list[0].next_but = state[206]
    for i in range(85,166):
        k = i-85
        y = int(k/9)
        x = k%9
        player_list[1].board[y][x] = state[i]
    player_list[1].loc = state[168]
    player_list[1].buttons = state[166]
    player_list[1].board_buttons = state[169]
    player_list[1].plus_seven = state[167]
    player_list[1].next_but = state[207]
    
    #build shared board
    sb = shared_board()
    sb.next_sq = state[208]
        
    #create pieces
    init_pieces = create_pieces()
    
    #build piece order
    #there is a smarter way to do this
    piece_list = []
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list.append(index)
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list[loc] = get_num(key_list[index+172])
    for index, piece in enumerate(piece_list):
        for gp in init_pieces:
            if piece == gp.num:
                piece_list[index] = gp 
    #counter
    counter = state[205]
    
    ######make move######
    #action 0 is just moving past the other player
    if action == 0:
        play_status=True
        move_spaces = player_list[1-turn].loc - player_list[turn].loc + 1
        player_list[turn].loc += move_spaces
        player_list[turn].buttons += move_spaces
        if player_list[turn].loc >= sb.tot_spaces:
            player_list[turn].loc = sb.tot_spaces
        
        #check if user passes button collect on board, if so collect buttons
        if check_pass_button(player_list[turn].loc, player_list[turn]) == True:
            player_list[turn].buttons += player_list[turn].board_buttons
            for space in sb.but_spaces:
                if player_list[turn].next_but < space:
                    player_list[turn].next_but = space
                    break
        if check_pass_square(player_list[turn].loc,sb.next_sq) == True:
            player_list[turn].board = comp_place_square(player_list[turn].board) 
            end = True
            for space in sb.sq_spaces:
                if player_list[turn].loc < space:
                    sb.next_sq = space
                    end = False
                    break
            if end == True:
                sb.next_sq = 999
    else:
        p2p = piece_list[action-1]
        piece_list, player_list, play_status = comp_calc_piece(p2p, 
                                                          turn, 
                                                          sb, 
                                                          player_list, 
                                                          piece_list)
        if check_pass_button(player_list[turn].loc, player_list[turn]) == True:
            player_list[turn].buttons += player_list[turn].board_buttons
            for space in sb.but_spaces:
                if player_list[turn].next_but < space:
                    player_list[turn].next_but = space
                    break
        if check_pass_square(player_list[turn].loc,sb.next_sq) == True:
            player_list[turn].board = comp_place_square(player_list[turn].board) 
            end=True
            for space in sb.sq_spaces:
                if player_list[turn].loc < space:
                    sb.next_sq = space
                    end=False
                    break
            if end == True:
                sb.next_sq = 999
    
    game_end = False
    #calculate reward
    #check if game is over - if not, reward is 0
    if play_status == True:
        if player_list[turn].loc + player_list[1-turn].loc <sb.tot_spaces*2:
            reward = 0
            game_end=False
        else:
            scores = score_game(player_list)
            reward = check_winner(scores)
            game_end=True
    else:
        game_end=False
        reward = 0
        
    if player_list[turn].loc > player_list[1-turn].loc:
        turn = 1-turn
    counter += 1

    #convert to new state
    d = build_state(player_list,turn,piece_list,counter,init_pieces, sb)
    state = []
    for key in key_list:
        state.append(d[key])
    return state, reward, game_end

def value_moves(state):
    button_spaces = [5,11,17,23,29,35,41,47,53]
    
    with open('../keys.csv', 'r') as f:
        reader = csv.reader(f)
        key_list = list(reader)
    key_list = key_list[0]
    
    next_three = []
    i = 0
    while len(next_three)<3:
        for index, loc in enumerate(state[0][172:205]):
            if loc == i:
                next_three.append(get_num(key_list[index+172]))
                i = i+1
                break
                
    p = create_pieces()
    rec = lambda x: sum(map(rec, x)) if isinstance(x, list) else x
    
    turn = state[0][171]
    
    if turn == 0:
        loc = state[0][83]
    if turn == 1:
        loc = state[0][168]
        
    num_paydays = len([k for k in button_spaces if k>loc])
    
    values = [1]
    
    for i in next_three:
        for pi in p:
            if pi.num == i:
                values.append((2 * (rec(pi.shape)) - pi.cost + pi.buttons*num_paydays)/pi.spaces)
                
    for index, val in enumerate(values):
        if index !=0:
            if loc < 35:
                if val < 2.5:
                    values[index] = .5
                
    return values
  
def check_valid_moves(player_list, turn, piece_list):
    valid_moves = [1]
    for i in range(3):
        if player_list[turn].buttons >= piece_list[i].cost:
            val = check_place_piece(piece_list[i],player_list[turn].board)
            if val == True:
                valid_moves.append(1)
            else:
                valid_moves.append(-999)
        else:
            valid_moves.append(-999)
    return valid_moves

def make_move_computer(state):
    #load agent, weights
    agent = DQNAgent(209,4)
    agent.load("../patchwork.h5")
    # determine valid moves - need to get info from state
    with open('../keys.csv', 'r') as f:
        reader = csv.reader(f)
        key_list = list(reader)
    key_list = key_list[0]
    player1 = player()
    player2 = player()
    player_list = [player1, player2]
    turn = state[171]
    player_list[0].loc = state[83]
    player_list[0].buttons = state[81]
    player_list[0].board_buttons = state[84]
    player_list[0].plus_seven = state[82]
    player_list[0].next_but = state[206]
    player_list[1].loc = state[168]
    player_list[1].buttons = state[166]
    player_list[1].board_buttons = state[169]
    player_list[1].plus_seven = state[167]
    player_list[1].next_but = state[207]
    for i in range(0,81):
        y = int(i/9)
        x = i % 9
        player_list[0].board[y][x] = state[i]
    for i in range(85,166):
        k = i-85
        y = int(k/9)
        x = k%9
        player_list[1].board[y][x] = state[i]

    #create pieces
    init_pieces = create_pieces()
    #build piece order
    #there is a smarter way to do this
    piece_list = []
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list.append(index)
    for index, loc in enumerate(state[172:205]):
        if loc != -1:
            piece_list[loc] = get_num(key_list[index+172])
    for index, piece in enumerate(piece_list):
        for gp in init_pieces:
            if piece == gp.num:
                piece_list[index] = gp

    valid_moves = check_valid_moves(player_list, turn, piece_list)    
    # determine action
    action = agent.act(np.reshape(np.array(state), [1, 209]), valid_moves)
    ## update state
    state = np.reshape(np.array(state), [1, 209])
    next_state, reward, done = make_move(state.tolist()[0], action)   
    next_state = np.reshape(np.array(next_state), [1, 209])
    agent.remember(state, action, reward, next_state, done)
    state = next_state[0]

    return(state.tolist(), reward, done)

def build_state(players, turn, pi_list, counter, init_pieces, sb):
    d_t={}
    for index, p in enumerate(players):
        #setting board states
        for i in range(len(p.board)):
            for j in range(len(p.board)):
                d_t['player' + str(index) + 'pos' + str(i) + "-" + str(j)] = p.board[i][j]
        #setting player states(buttons, locations, plus7, board_buttons)
        d_t['player' + str(index) + 'bank_buttons'] = p.buttons
        d_t['player' + str(index) + 'p7'] = p.plus_seven
        d_t['player' + str(index) + 'loc'] = p.loc
        d_t['player' + str(index) + 'board_buttons'] = p.board_buttons
        d_t['player' + str(index) + 'next_but'] = p.next_but
                        
    #setting player turn
    d_t['player0_turn'] = 1 - turn
    d_t['player1_turn'] = turn
                
    #setting piece order
    for r in init_pieces:
        active_nums = [i.num for i in pi_list]
        try:
            d_t['piece_loc' + str(r.num)] = active_nums.index(r.num)
        except:
            d_t['piece_loc' + str(r.num)] = -1

    d_t['counter'] = counter
    
    d_t['next_sq'] = sb.next_sq
    
    return d_t

# Initialize the app
app = flask.Flask(__name__)

@app.route("/")
def viz_page():
    with open("Patchwork.html", 'r') as viz_file:
        return viz_file.read()

@app.route("/update", methods=["POST"])
def update():
    data = flask.request.json
    new_game = data['new_game']
    state = data['state']
    if data['action'] == "Move":
        action = 0
    else:
        action = int(data['action']) + 1
    p_perm = int(data['perm'])
    p_loc = data['loc']
    p_loc = [int(x) for x in p_loc]

    if new_game == 1:
        state = create_initial_state()
        reward = 0
        status = False
    else:
        turn = state[171]
        p1_type = data['p1_type']
        p2_type = data['p2_type']

        if turn == 0:
            p_type = p1_type
        else:
            p_type = p2_type
        if p_type == 'Human':
            state, reward, status = make_move_human(state, action, p_perm, p_loc)
        else:
            state, reward, status = make_move_computer(state)

    board1 = state[:81]
    board2 = state[85:166]
    piece_shapes, piece_cost, piece_spaces, piece_buttons = piece_shape_list(state)
    return flask.jsonify({'state': state, 'board1': board1, 'board2': board2, 'pieces': piece_shapes,
                            'costs': piece_cost, 'spaces': piece_spaces, 'buttons': piece_buttons,
                            'reward': reward, 'status': status})

@app.route("/perm", methods=["POST"])
def perm():
    data = flask.request.json
    state = data['state']
    p = int(data['p'])

    perms = piece_perms(state,p)
    return flask.jsonify({'perms': perms})

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port=5002)