import os.path
import seaborn as sns
import matplotlib.pyplot as plt


import sys

sys.path.append('../../')

from connect_four_project.mcts.mcts import MCTS
import random
import numpy as np
import pickle


class Tournament:
    def __init__(self, players, GameClass):
        self.number_of_games_collect_train = 100
        self.number_of_iterations = 10000
        self.evalute_each_X_iterations = 100
        self.players = players
        self.GameClass = GameClass

        # load evaluation if exists
        try:
            with open('../trained-models/evaluation_against_MCTS_1000', 'rb') as f:
                self.evaluation_results = pickle.load(f)
                # MTCS 50, evaluation each 10 iterations.
                # [1, 2, 2, 3, 2, 4, 1, 1, 3, 3, 1, 1, 6, 1, 4, 0, 4, 3, 3, 1, 0, 1, 2, 1, 5, 6, 2, 0, 1, 8, 2, 3, 3, 3, 3, 5, 3, 4, 2, 3, 4, 2, 2, 4, 4, 2, 1, 2, 2, 3, 3, 2, 3, 5, 4, 2, 4, 5, 3, 4, 4, 1, 1, 4, 6, 3, 1, 6, 2, 3, 4, 4, 3, 4, 5, 4, 3, 5, 2, 6, 6, 2, 3, 4, 2, 5, 4, 6, 2, 4, 3, 4, 8, 4, 5, 5, 5, 8, 6, 5, 6, 4, 5, 6, 5, 5, 5, 4, 3, 6, 5, 6, 6, 3, 7, 4, 5, 4, 4, 6, 7, 4, 7, 5, 4, 6, 5, 6, 3, 5, 3, 2, 9, 6, 6, 6, 5, 6, 4, 7, 5, 5, 8, 9, 8, 7, 7, 5, 3, 6, 6, 10]
                # MCTS 1000, evaluation each 50 iteration, double board games.
                # [0, 0, 0, 0, 2, 1, 2, 0, 0, 2, 1, 1, 2, 2, 3, 2, 0, 0, 2, 1, 1, 1, 2, 1, 1, 2, 0, 2, 1, 1, 0, 1, 3, 2, 2, 2, 1, 1, 3, 0, 0, 1, 0, 0, 4, 1, 4, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 5, 3, 4, 1, 1, 0, 3, 4, 4, 1, 5, 2, 2, 3, 1, 2, 0, 1, 1, 3, 3, 2, 1, 3, 3, 2, 2, 3, 3, 2, 0, 4, 3, 0, 3, 2, 2, 2, 1, 1, 1, 3, 2, 4, 4, 1, 4, 3, 3, 2, 5, 4, 5, 2, 1, 5, 6, 5, 4, 3, 8, 6, 3, 5, 4, 4, 1, 2, 6, 2, 3, 6, 2, 4, 3, 3, 5, 5, 5, 7, 5, 4, 4, 3, 5, 5, 7, 3, 7, 4, 6, 6, 6, 4, 6, 5, 4, 7, 10, 7, 4, 5, 7, 7, 7, 6, 8, 10]

        except:
            self.evaluation_results = []

    def start_training(self):
        for iteration in range(self.number_of_iterations):
            # play some games
            for _ in range(self.number_of_games_collect_train):
                game = self.GameClass(7, 6)  # create game
                players = random.sample(self.players, 2)  # create players
                game.win, p1_turn, game.tie, p1_boards, p2_boards = self.play(players[0], players[1], game)
                self.store(game.win, p1_turn, game.tie, p1_boards, p2_boards, players[0], players[1])
            # update networks
            for i, plyer in enumerate(self.players):
                plyer.fit()

            # evaluate networks
            if iteration % self.evalute_each_X_iterations == 0 and iteration != 0:
                self.evaluation_results.append(self.evaluate(self.players[0]))
                self.plot()
                for i, plyer in enumerate(self.players):
                    plyer.value_net.save('trained-models/player ' + str(i))

                with open('../trained-models/evaluation_against_MCTS_1000', 'wb') as f:
                    pickle.dump(self.evaluation_results, f)

    def play(self, p1, p2, game, epsilon=0.1):
        done = False
        p1_turn = True

        p1_boards = []
        p2_boards = []
        while not done:
            if p1_turn:
                action = p1.play(game, epsilon, p2_turn=False)
                status = game.step(action)
                p1_boards.append(np.array(game.board))
            else:
                action = p2.play(game, epsilon, p2_turn=True)
                status = game.step(action)
                p2_boards.append(game.board * -1)

            if game.win or game.tie:
                return game.win, p1_turn, game.tie, p1_boards, p2_boards

            p1_turn = not p1_turn

    def store(self, win, p1_turn, tie, p1_boards, p2_boards, p1, p2):
        discount = 1
        if tie:
            # discounted
            p1.rewards += [0.1 * (discount ** x) for x in range(len(p1_boards))][::-1]
            p2.rewards += [0.1 * (discount ** x) for x in range(len(p2_boards))][::-1]
            p1.rewards += [0.1 * (discount ** x) for x in range(len(p1_boards))][::-1]
            p2.rewards += [0.1 * (discount ** x) for x in range(len(p2_boards))][::-1]
        if win:
            # discounted
            p1.rewards += [int(p1_turn) * (discount ** x) for x in range(len(p1_boards))][::-1]
            p2.rewards += [int(not p1_turn) * (discount ** x) for x in range(len(p2_boards))][::-1]
            p1.rewards += [int(p1_turn) * (discount ** x) for x in range(len(p1_boards))][::-1]
            p2.rewards += [int(not p1_turn) * (discount ** x) for x in range(len(p2_boards))][::-1]
        p1.boards += p1_boards
        p2.boards += p2_boards

        # exploiting horizontal symmetry
        p1_reversed = [np.fliplr(board) for board in p1_boards]
        p2_reversed = [np.fliplr(board) for board in p2_boards]
        p1.boards += p1_reversed
        p2.boards += p2_reversed

    def evaluate(self, p1):
        GMC_wins = []
        for game_number in range(10):
            game = self.GameClass(7, 6)
            if game_number > 4:
                win, p1_turn, tie, _, _ = self.play(MCTS(2000, np.sqrt(2)), self.players[0], game, epsilon=0)
                if win and p1_turn:
                    GMC_wins.append(-1)
                if win and not p1_turn:
                    GMC_wins.append(1)

            else:
                win, p1_turn, tie, _, _ = self.play(self.players[0], MCTS(2000, np.sqrt(2)), game, epsilon=0)
                if win and p1_turn:
                    GMC_wins.append(1)
                if win and not p1_turn:
                    GMC_wins.append(-1)
            if tie:
                GMC_wins.append(0)
        return GMC_wins.count(1)

    def plot(self):
        sns.set_style('darkgrid')
        plt.plot(np.array(self.evaluation_results))
        plt.show()

    def play_against_human(self, computer_red=True):
        p1, p2, p3 = self.players
        done = False
        game = self.GameClass(7, 6)

        if not computer_red:
            game.render()
            action = int(input())
            status = game.step(action)

        while not done:
            game.render()

            # computer turn
            actions = [p1.play(game, epsilon=0, p2_turn=not computer_red),
                       p2.play(game, epsilon=0, p2_turn=not computer_red),
                       p3.play(game, epsilon=0, p2_turn=not computer_red)]
            action = max(set(actions), key=actions.count)
            print("Probability of winning against you is {}%".format(float(p1.win_rate+p2.win_rate+p3.win_rate)*100/3))
            status = game.step(action)
            game.render()
            if game.win or game.tie:
                return game.win, game.tie

            # player turn
            action = int(input())
            status = game.step(action)

            if game.win or game.tie:
                return game.win, game.tie


