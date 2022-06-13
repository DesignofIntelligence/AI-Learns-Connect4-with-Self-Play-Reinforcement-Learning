import numpy as np
from connect_four_project.game.connectfour_game import ConnectFour


class Node:
    def __init__(self, parent, action, player):
        self.visits = 0
        self.value = None
        self.children = []
        self.parent = parent
        self.action = action
        self.player = player
        self.terminal = False


class MCTS:
    def __init__(self, num_iter,c):
        self.num_iter = num_iter
        self.c = c

    def play(self, game,epsilon=0 ,p2_turn=False):
        root = Node(None, None, -1 * game.current_turn)
        current_node = root
        for _ in range(self.num_iter):

            game_clone = game.clone()

            while len(current_node.children):  # selection
                current_node = self.choose_highest_UCB(current_node.children)
                status = game_clone.step(current_node.action)

            if current_node.terminal:
                #only node remaining, even though high low ucb
                break # the node will keep getting picked because of ucb, break

            if current_node.value == None:
                self.rollout(game_clone)
            else:
                for action in game_clone.get_available_actions():  # expansion
                    current_node.children.append(Node(current_node, action, -current_node.player))
                if not len(current_node.children):
                    print("HI")
                current_node = np.random.choice(current_node.children)

                status = game_clone.step(current_node.action)

                if status != "On Going":
                    current_node.terminal = True
                    current_node.visits = 100000
                else:
                    self.rollout(game_clone)

            value = 0
            if game_clone.win:  # win
                value = 1 if game_clone.current_turn == current_node.player else 0
            if game_clone.tie:  # tie
                value = 0.5 # because if tie is equal to 0, at the end of the game that has all future outcomes
                # ties, the root wont expand as the value will always be 0

            # backpropagation
            current_node.value = value
            current_node.visits += 1
            while current_node.parent is not None:
                current_node = current_node.parent
                value = 1 - value  # reverse value for the previous player
               #  value *= -1
                current_node.value += value
                current_node.visits += 1


        root.children.sort(key=lambda x: x.value, reverse=True)
        # print("Their coress values"+ str([x.value for x in root.children]))
        # print("Top Actions are " + str([x.action for x in root.children]))
        # print("their visits " + str([x.visits for x in root.children]))
        return root.children[0].action

    def rollout(self, game):
        while not (game.win or game.tie):
            game.step(np.random.choice(game.get_available_actions()))

    def choose_highest_UCB(self, children):  # children are nodes
        UCBs = []
        for child in children:
            ucb_child = 0
            if child.visits == 0:
                ucb_child = np.inf
            else:
                ucb_child = child.value/child.visits + self.c * np.sqrt(np.log(child.parent.visits) / child.visits)

            UCBs.append(ucb_child)

        maximum = max(UCBs)
        all_max_indices = [i for i, x in enumerate(UCBs) if x == maximum]  # get all max UCBs
        index_max = np.random.choice(all_max_indices)  # tie break randomly
        return children[index_max]


if __name__ == '__main__':
    env = ConnectFour(7, 6)
    env.render()
    turn = False
    mcts_explore = MCTS(3000,np.sqrt(2)) # explorer
    while True:
        print("----------------------------------")
        #turn = "Yellow" if env.current_turn == -1 else "Red"
        print("Player {} turn".format(turn))
        if turn:
            action = mcts_explore.play(env)
        else:
            action = int(input())
            #action = mcts_explore.play(env)
        env.step(action)
        env.render()


        if env.win:
            print(env.current_turn, "Win")
            env = ConnectFour(7, 6)
        if env.tie:
            print("Tie")
            env = ConnectFour(7, 6)
        turn = not turn

