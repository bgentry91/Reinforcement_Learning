import copy
import numpy as np
import curses

def make_blank_board():
	#empty state
	return [-1,-1,-1,-1,-1,-1,-1,-1,-1,0], [0,1,2,3,4,5,6,7,8]

def print_board(board):
	#pretty print board
	l = []
	for s in board[0]:
		if s == -1:
			s = '-'
		if s == 0:
			s = 'X'
		if s == 1:
			s = 'O'
		l.append(s)
		if len(l) % 3 == 0:
			print(*l)
			l = []
	print('*******')

def find_blanks(board):
	#identify possible moves
	blanks = []
	for index, loc in enumerate(board[0][0:9]):
		if loc == -1:
			blanks.append(index)
	return blanks

def det_winner(state):
	#determine game winner
	if len(set([state[0],state[1],state[2]])) == 1 and set([state[0],state[1],state[2]]) != {-1}:
		return state[0]
	if len(set([state[3], state[4], state[5]])) == 1 and set([state[3],state[4], state[5]]) != {-1}:
		return state[3]
	if len(set([state[6], state[7], state[8]])) == 1 and set([state[6], state[7], state[8]]) != {-1}:
		return state[6]
	if len(set([state[0], state[3], state[6]])) == 1 and set([state[0],state[3], state[6]]) != {-1}:
		return state[0]
	if len(set([state[1], state[4], state[7]])) == 1 and set([state[1],state[4], state[7]]) != {-1}:
		return state[1]
	if len(set([state[2], state[5], state[8]])) == 1 and set([state[2],state[5], state[8]]) != {-1}:
		return state[2]
	if len(set([state[0], state[4], state[8]])) == 1 and set([state[0],state[4], state[8]]) != {-1}:
		return state[0]
	if len(set([state[2], state[4], state[6]])) == 1 and set([state[2],state[4], state[6]]) != {-1}:
		return state[2]
	for i in state[0:9]:
		if i == -1:
			return ('-')
	return "Draw"

def check_win(state):
	#check if game is over & return winner
	winner = det_winner(state[0])
	if winner == 0:
		return 1, True
	if winner == 1:
		return -1, True
	if winner == 'Draw':
		return 0, True
	return 0, False

def update_state(board,player,d):
	#update dictionary (backprop probabilities)
	#return probabilities for future states
	alpha = .95
	pot_moves = find_blanks(board)
	next_probs = []
	for a in pot_moves:
		t_board = trial(board,player,a)
		if bool(d[tuple(t_board[0])]) == False:
			set_default_state(t_board,d)
		d[tuple(board[0])] = d[tuple(board[0])] + alpha*(d[tuple(t_board[0])] - d[tuple(board[0])])
		next_probs.append(d[tuple(t_board[0])])
	return next_probs

def set_default_state(board, d):
	#if record doesn't exist in weights dict, make it
	if bool(d[tuple(board[0])]) == False:
		d[tuple(board[0])] = check_win(board)[0]

def place_random(board, player):
	#place piece randomly
	blanks = find_blanks(board)
	p = blanks[np.random.randint(0,len(blanks))]
	return (p)

def place_move(board,player,move):
	#place piece
	board[0][move] = copy.deepcopy(player)
	return board

def trial(board, player, loc):
	#create future state
	test_board = copy.deepcopy(board)
	test_board[0][loc] = player
	return test_board

def check_move(board, loc):
	#check if user's move is valid
	if board[0][loc] != -1:
		print('Not an open location')
		return False
	return True

def update_state_dqn(state, action):
    player = state[9]
    new_state = copy.deepcopy(state)
    new_state[action] = player
    
    new_state[9] = 1-int(new_state[9])
    
    valid_moves = find_blanks(new_state)
    
    reward, done = check_win(new_state)

    return new_state, reward, done, valid_moves

