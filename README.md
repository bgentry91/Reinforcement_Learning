### Reinforcement Learning Repo

This repo contains two different reinforcement learning games. The first is TicTacToe, which includes a game using standard Q-Learning per [Sutton and Barto's Reinforcement Learning](http://incompleteideas.net/book/bookdraft2017nov5.pdf) introduction. After about 5000 games, this model wins very reliably.
In addition there is an implementation of a Deep Q-Learning Network (DQN) in TicTacToe based on the code outlined on [Keon Kim's post on DQNs](https://keon.io/deep-q-learning/). After a large number of iterations, I was unable to get the DQN model to win the game reliably on it's own.

The second is an implementation of a combination heuristic and DQN used to train and play the board game [Patchwork](https://boardgamegeek.com/boardgame/163412/patchwork). This was presented as my final project at Metis and includes an associated Flask app to play the game.
