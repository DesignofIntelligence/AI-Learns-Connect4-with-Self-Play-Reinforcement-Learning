import numpy as np
from scipy.signal import convolve2d

RED = 1
YELLOW = -1
NOTHING = 0


class ConnectFour:
    def __init__(self, width, height):
        self.board_width = width
        self.board_height = height
        self.connect_x = 4  # connect 4
        self.current_turn = RED
        self.board = np.zeros((height, width))
        self.win = False
        self.tie = False

    def get_available_actions(self):
        actions = []
        for i in range(self.board_width):
            if self.is_action_legal(i):
                actions.append(i)
        return actions

    def clone(self):
        connect = ConnectFour(self.board_width,self.board_height)
        connect.current_turn = self.current_turn
        connect.board = np.array(self.board)
        return connect

    def is_action_legal(self,action):
        if action < 0 or action > self.board_width - 1 or (not self.board[0, action] == NOTHING):
            return False
        return True

    def drop(self, column, board):
        row = 0
        while row < 5:
            if board[row + 1, column] == 0:
                row += 1
            else:
                break
        board[row, column] = self.current_turn
        return board

    def step(self, action):
        if not self.is_action_legal(action):
            return -1
        self.board = self.drop(action, self.board)  # returns row and column of piece dropped
        if self._check_for_win():
            self.win = True
            return 1
        if np.all(self.board):  # check if there are no zeros in the board
            self.tie = True
            return 0
        self.current_turn *= -1  # change turns
        return "On Going"

    def render(self):
        board_animate = self.board.tolist()
        board_animate = [["🥵" if x == RED else x for x in row] for row in board_animate]
        board_animate = [["🟡" if x == YELLOW else x for x in row] for row in board_animate]
        board_animate = [["⬛️" if x == NOTHING else x for x in row] for row in board_animate]
        for row in board_animate:
            print("".join(row))
        print("0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣")

    def _check_for_win(self):  # convolve the board with 4 kernels, if there is a 4 in result then return true
        horizontal_kernel = np.array([[1] * self.connect_x])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(self.connect_x, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        for kernel in detection_kernels:
            if (convolve2d(self.board == self.current_turn, kernel, mode="valid") == self.connect_x).any():
                return True
        return False


if __name__ == '__main__':
    env = ConnectFour(7, 6)
    env.render()
    while True:
        print("----------------------------------")
        turn = "Yellow" if env.current_turn == -1 else "Red"
        print("Player {} turn".format(turn))
        env.get_available_actions()
        action = int(input())
        status = env.step(action)
        env.render()
        if status == 1:
            print(env.current_turn, "Win")
            env = ConnectFour(7, 6)
        if status == 0:
            print("Tie")
            env = ConnectFour(7, 6)