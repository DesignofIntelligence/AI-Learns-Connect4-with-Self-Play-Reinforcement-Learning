# AI-Learns-Connect4-with-Self-Play-Reinforcement-Learning
A reinforcement Learning project, where 3 neural networks play against each other, by choosing the best possible move which is decided based on the value function. This value function takes in a board position and outputs a win probability, so each neural network chooses the move which yields the highest win probability.

The training process occur by collecting the data, mirroring it (as connect4 configuration could be mirrored), and feeding it to the neural network, with the rewards as labels, if the game is won then the reward is +1, if its lost then the reward is 0. A Network of the 3 networks gets evaluated against a Monte-Carlo Tree Search Algorithm every X iterations.


The 3 networks then are used as an ensemble to play against the human player, it have reached a level as good as a Monte-Carlo 2000 Rollout. and it can defeat a professional human player.


Run

```
cd AI-Learns-Connect4-with-Self-Play-Reinforcement-Learning
python main.py
```
