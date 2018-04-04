import numpy as np
from collections import defaultdict
import copy
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Nadam
import TTT_Functions_DQN as ttt
import pickle

# Deep Q-learning Agent - https://keon.io/deep-q-learning/
class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.9995
		self.learning_rate = 0.0001
		self.model = self._build_model()
	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(100, input_dim=self.state_size, activation='relu'))
		model.add(Dense(75, activation='elu'))
		model.add(Dense(50, activation='elu'))
		model.add(Dense(25, activation='elu'))
		model.add(Dense(10, activation='elu'))
		model.add(Dense(self.action_size, activation='softmax'))
		model.compile(loss='categorical_crossentropy',
					  optimizer=Nadam(lr=self.learning_rate))
		return model
	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
	def act(self, state, valid_moves):
		#both players attempting to win game
		if state[0][9] == 1:
			#if epsilon has not decayed far enough, choose random move
			if np.random.rand() <= self.epsilon:
				return random.choice(valid_moves)
			#otherwise, choose best move for player 2 to win
			act_values = self.model.predict(state)
			act_values = [act_values[0][i] for i in valid_moves]
			return valid_moves[np.argmin(act_values)]  # returns action
		else:
			#same as above, except chose best move for player1 to win
			if np.random.rand() <= self.epsilon:
				return random.choice(valid_moves)
			act_values = self.model.predict(state)
			act_values = [act_values[0][i] for i in valid_moves]
			return valid_moves[np.argmax(act_values)]  # returns action
	def replay(self, batch_size):
		#create minibatch from memory
		minibatch = random.sample(self.memory, batch_size)
		#update weights for each record in memory
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				#if game isnt over, set target of future action based on gamma & prediction
				target = reward + self.gamma * np.amax(self.model.predict(next_state))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			#refit model to update weights
			self.model.fit(state, target_f, epochs=1, verbose=0)
		#decay epsilon as necessary
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
	def save(self, name):
		self.model.save_weights(name)
		pickle.dump(self.memory, open("memory.pkl","wb"))
	def load(self, name):
		self.model.load_weights(name)
		self.memory = pickle.load(open("memory.pkl","rb"))

def train_model(episodes):
	agent = DQNAgent(10,9)
	try:
		agent.load("./TTT.h5")
	except:
		pass
	# Iterate the game - set number of episodes
	for e in range(episodes):
		# reset state in the beginning of each game
		state, valid_moves = ttt.make_blank_board()
		state = np.reshape(state, [1, 10])
		done=False
		# iterate through until game is over
		while done==False:
			# Decide action
			action = agent.act(state, valid_moves)
			# take action
			next_state, reward, done, valid_moves = ttt.update_state(state[0],action)
			next_state = np.reshape(next_state, [1, 10])
			# Remember the previous state, action, reward, and done
			agent.remember(state, action, reward, next_state, done)
			# next state becomes current state
			state = next_state
			if done:
				# print the score and break out of the loop
				print("\r Game: {}/{}, score: {}"
					  .format(e+1, episodes, reward))
				break
		# train the agent
		try:
			agent.replay(32)
		except:
			agent.replay(3)
	agent.save("./TTT.h5")

def play_game():
	agent = DQNAgent(10,9)
	try:
		agent.load("./TTT.h5")
	except:
		pass
	state, valid_moves = ttt.make_blank_board()
	state = np.reshape(state, [1, 10])
	done=False
	# iterate through until game is over
	while done==False:
		# Decide action
		if state[0][9] == 0:
			action = agent.act(state, valid_moves)
		else:
			check = False
			while check == False:
				print('Input Move Location (0 to 8): ')
				action = int(input())
				try:
					check = ttt.check_move(state, action)
				except:
					print('Not a valid input')
			print('*******')
		# take action
		next_state, reward, done, valid_moves = ttt.update_state(state[0],action)
		next_state = np.reshape(next_state, [1, 10])
		# Remember the previous state, action, reward, and done
		agent.remember(state, action, reward, next_state, done)
		# next state becomes current state
		state = next_state
		ttt.print_board(state[0])
		if done:
			if reward == 1:
				print('Player 1 Wins!')
			if reward == -1:
				print('Player 2 Wins!')	
			if reward == 0:
				print ('Draw :-(')
	# train the agent
	try:
		agent.replay(32)
	except:
		agent.replay(3)
	agent.save("./TTT.h5")

#Script to play - train/play/save
print('Would you like to add more training data to the model?[y/n]:')
train_r = input()

if train_r == 'y':
	print('For how many iterations?')
	itrs = int(input())
	d = train_model(itrs)


play = 'y'
while play == 'y':
	print('Would you like to play the trained model?[y/n]:')
	play = input()
	if play =='y':
		d = play_game()

print('Thank you for playing. Your trained model has been saved.')
