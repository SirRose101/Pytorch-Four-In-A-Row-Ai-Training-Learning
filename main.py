import typing


def declare_winner(win_sum: int) -> None:
    if win_sum == 3:
        raise ZeroDivisionError
    if win_sum == -3:
        raise TypeError


# main game class
class Game:
    def __init__(self):
        self.board: typing.List[int] = [0, 0, 0,
                                   0, 0, 0,
                                   0, 0, 0,]
        self.X_turn = True
        self.turn = 0

    def index_to_symbol(self, index: int) -> str:
        if self.board[index] == 0:
            return " "
        if self.board[index] == 1:
            return "X"
        else:
            return "O"

    def print_board(self):
        for i in range(3):
            for j in range(3):
                print(self.index_to_symbol(i * 3 + j), end="")
                if j != 2:
                    print("|", end="")
            print("")
            if i != 2:
                print("------")

    def insert(self, index):
        if self.board[index] == 0:
            self.turn += 1
            if self.X_turn:
                self.board[index] = 1
            else:
                self.board[index] = -1
            if self.turn >= 5:
                self.check_win()
                if self.turn == 9:
                    raise  AttributeError
        self.X_turn = not self.X_turn

    def check_win(self):
        win_sum = 0
        for i in range(3):
            for j in self.board[3 * i:3 * i + 3]:
                j = j
                win_sum += j
            declare_winner(win_sum)
            win_sum = 0
        for i in range(3):
            for j in self.board[i::3]:
                win_sum += j
            declare_winner(win_sum)
            win_sum = 0
        win_sum = self.board[0] + self.board[4] + self.board[8]
        declare_winner(win_sum)
        win_sum = self.board[2] + self.board[4] + self.board[6]
        declare_winner(win_sum)

    def reset(self):
        self.board = [0, 0, 0,
                      0, 0, 0,
                      0, 0, 0]
        self.turn = 0
        self.X_turn = True


game = Game()
try:
    while True:
        game.print_board()
        game.insert(int(input("which square do you want?\n>>>\t")) - 1)
except ZeroDivisionError:
    print("X won!")
except TypeError:
    print("0 won!")
except AttributeError:
    print("Tie...")
game.print_board()