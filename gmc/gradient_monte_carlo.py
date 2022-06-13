import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
import numpy as np
import random

class ValueNetworkCNN:
    def __init__(self, input_shape, lr):
        self.model = Sequential()
        self.model.add(tf.keras.layers.Conv2D(128, (4, 4), input_shape=input_shape))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(64))
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.Dense(64))
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(loss="mse",
                           optimizer=Adam(lr=lr))

    def forward_prop(self, obs):
        x = tf.convert_to_tensor([obs], dtype=None, dtype_hint=None, name=None)
        #x = tf.reshape(x, (1, 6, 7, 1))
        y = self.model(x)
        return y

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)

    def get_model(self):
        return self.model


class GradientMonteCarlo:
    def __init__(self, actions, step_size):
        self.value_net = ValueNetworkCNN((6, 7, 1), step_size)
        self.boards = []
        self.rewards = []
        self.batch_size = 128
        self.win_rate = 0
    def fit(self):
        c = list(zip(self.rewards, self.boards))
        random.shuffle(c)
        self.rewards, self.boards = zip(*c)
        self.value_net.model.fit(np.array(self.boards), np.array(self.rewards), verbose=0,batch_size=self.batch_size)
        self.boards = []
        self.rewards = []
    def play(self, game,epsilon,p2_turn=False):
        list_of_actions = game.get_available_actions()
        list_of_values = []  # values of the board for each action taken
        for action in list_of_actions:
            game_clone = game.clone()
            board = game_clone.drop(action,game_clone.board)
            if p2_turn:
                board = board * -1
            value = self.value_net.forward_prop(board)
            list_of_values.append(value)
        self.win_rate = max(list_of_values)
        if np.random.uniform(0,1)<epsilon:
            return np.random.choice(list_of_actions)
        else:
            return list_of_actions[np.argmax(list_of_values)]
