import typing


class Slot:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.color: str = "empty"

    def __str__(self):
        if self.color == "empty":
            return "0"
        if self.color == "red":
            return "\033[31m1\033[0m"
        return "\033[34m1\033[0m"

    def is_empty(self):
        return self.color == "empty"

    def other_color(self):
        if self.color == "empty":
            return "empty"
        if self.color == "red":
            return "blue"
        return "red"


def list_of_slots_and_color_to_array(array: typing.List[Slot], color: str) -> typing.List[int]:
    return_array = []
    for slot in array:
        if slot.color == color:
            return_array.append(1)
        else:
            return_array.append(0)
    return return_array

class Game:
    def __init__(self):
        self.running = True
        rows: int = 6
        cols: int = 7
        self.turn = 0
        self.col: typing.List[typing.List[Slot]] = [[] for _ in range(cols)]
        self.rows: typing.List[typing.List[Slot]] = []
        for i in range(rows):
            row: typing.List[Slot] = []
            for j in range(cols):
                slot = Slot(i, j)
                row.append(slot)
                self.col[j].append(slot)
            self.rows.append(row)
        self.red_turn: bool = True

    def insert(self, column_inserted):
        if column_inserted<0:
            raise ValueError("Try again, that column is full or invalid.")
        column = self.col[column_inserted]
        if not column[0].is_empty():
            raise ValueError("Column is full")
        column_curr = column[0]
        for i in range(len(column) - 1):
            if not column[i + 1].is_empty():
                column_curr.color = self.return_turn_color()
                self.red_turn = not self.red_turn
                return i
            column_curr = column[i + 1]
        column[-1].color = self.return_turn_color()
        self.red_turn = not self.red_turn
        return len(column) - 1

    def print_state(self):
        for row in self.rows:
            for slot in row:
                print(slot, end="")
            print()

    def return_turn_color(self):
        return "red" if self.red_turn else "blue"

    def check_win(self, row, col):
        color = self.rows[row][col].color
        num_rows = len(self.rows)
        num_cols = len(self.rows[0])
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < num_rows and 0 <= c < num_cols and self.rows[r][c].color == color:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def gameloop(self):
        while self.running:
            self.print_state()
            while True:
                try:
                    col = int(input("What column do you pick? "))
                    row = self.insert(col)
                    break
                except ValueError:
                    print("Try again, that column is full or invalid.")
            self.turn += 1
            if self.turn >= 7 and self.check_win(row, col):
                self.print_state()
                winner = "Blue" if self.red_turn else "Red"
                print(f"{winner} wins!")
                self.running = False

    def training_loop(self, col):
        try:
            row = self.insert(col)
        except ValueError:
            raise ValueError("Try again, that column is full or invalid.")
        return row

    def game_to_array(self):
        red, empty, blue = [], [], []
        for row in self.rows:
            red = red + list_of_slots_and_color_to_array(row, "red")
        for row in self.rows:
            empty = empty + list_of_slots_and_color_to_array(row, "empty")
        for row in self.rows:
            blue = blue + list_of_slots_and_color_to_array(row, "blue")
        if not self.red_turn:
            return blue, empty, red
        return red, empty, blue

    def reset(self):
        self.running = True
        rows: int = 6
        cols: int = 7
        self.turn = 0
        self.col: typing.List[typing.List[Slot]] = [[] for _ in range(cols)]
        self.rows: typing.List[typing.List[Slot]] = []
        for i in range(rows):
            row: typing.List[Slot] = []
            for j in range(cols):
                slot = Slot(i, j)
                row.append(slot)
                self.col[j].append(slot)
            self.rows.append(row)
        self.red_turn: bool = True
