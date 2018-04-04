import TTT_Functions as ttt
import random
from collections import defaultdict
import pickle

def det_move(board, player, d, beta):
	#computer determine next move
	ttt.set_default_state(board,d)
	p = ttt.update_state(board,player,d)
	best_prob = 0
	pot_moves = ttt.find_blanks(board)
	exploratory_probs = []
	for index, prob in enumerate(p):
		if prob == 1:
			return pot_moves[index]
		if prob <= beta:
			exploratory_probs.append(pot_moves[index])
		if prob > beta:
			if prob > best_prob:
				best_prob = prob
				best_move = pot_moves[index]
	if best_prob > 0:
		return best_move
	return random.choice(exploratory_probs)

def train_model(itrs):
	#train model
	#beta (decision parameter) is high for initial training
	beta = .1
	d = defaultdict(int)
	for i in range(itrs):
		board = ttt.make_blank_board()
		move = 0
		player = 0
		z = False
		while move < 9:
			if player == 0:
				next_move = det_move(board, player, d, beta)
				board = ttt.place_move(board,player,next_move)
			else:
				next_move = ttt.place_random(board,player)
				ttt.set_default_state(board,d)
				ttt.update_state(board,player,d)
				board = ttt.place_move(board,player,next_move)
			win = ttt.check_win(board)[0]
			if win == -1:
				z = True
			elif win == 1:
				z = True
			if z == True:
				break
			player = 1 - player
			move += 1
		print('Game %d/%d trained.' %(i+1, itrs), end = '\r')
	print('Game %d/%d trained.' %(i+1, itrs))
	return d

def play_game(d):
	#play the game against human using trained AI
	#beta (decision parameter is zero)
	beta = 0
	board = ttt.make_blank_board()
	move = 0
	player = 0
	ttt.print_board(board)
	z=False
	while move < 9:
		if player == 0:
			next_move = det_move(board, player, d, beta)
			board = ttt.place_move(board,player,next_move)
			ttt.print_board(board)
		else:
			check = False
			while check == False:
				print('Input Move Location (0 to 8): ')
				next_move = int(input())
				check = ttt.check_move (board, next_move)
			print('*******')
			ttt.set_default_state(board, d)
			ttt.update_state(board,player, d)
			board = ttt.place_move(board,player,next_move)
			ttt.print_board(board)
		win = ttt.check_win(board)[0]
		if win == -1:
			print('O Wins')
			z = True
		elif win == 1:
			print('X Wins')
			z = True
		if z == True:
			break
		player = 1 - player
		move += 1
	return d

#Script to play - train/play/save

print('Would you like to train the model from scratch?[y/n]:')
train_r = input()

if train_r == 'y':
	print('For how many iterations?')
	itrs = int(input())
	d = train_model(itrs)
else:
	print('Loading in weights...')
	try:
		with open('TTT_Weights.pkl', 'rb') as f:
			pickle.load(f)
	except:
		print('No weights found.')

play = 'y'
while play == 'y':
	print('Would you like to play the trained model?[y/n]:')
	play = input()
	if play =='y':
		d = play_game(d)

print('Would you like to save the trained weights?[y/n]:')
save = input()
if save == 'y':
	try:
		with open('TTT_Weights.pkl', 'wb') as f:
			pickle.dump(d,f)
	except:
		print('No weights to save.')


