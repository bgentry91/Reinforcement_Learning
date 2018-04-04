import copy


def make_blank_board():
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1,0], [0,1,2,3,4,5,6,7,8]

def print_board(board):
    l = []
    for s in board:
        if s == -1:
            s = '-'
        l.append(s)
        if len(l) % 3 == 0:
            print(*l)
            l = []
    print('*******')

def find_blanks(board):
    blanks = []
    for index, loc in enumerate(board[0:9]):
        if loc == -1:
            blanks.append(index)
    return blanks

def det_winner(state):
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
    winner = det_winner(state)
    if winner == 0:
        return 1, True
    if winner == 1:
        return -1, True
    if winner == 'Draw':
        return 0, True
    return 0, False

def check_move(board, loc):
	#check if user's move is valid
	if loc > 8 or loc < 0:
		print('Not a valid position')
		return False
	if board[0][loc] != -1:
		print('Not an open location')
		return False
	return True
		

def update_state(state, action):
    player = state[9]
    new_state = copy.deepcopy(state)
    new_state[action] = player
    
    new_state[9] = 1-int(new_state[9])
    
    valid_moves = find_blanks(new_state)
    
    reward, done = check_win(new_state)

    return new_state, reward, done, valid_moves