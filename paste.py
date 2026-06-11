import typing

# ── Game logic (unchanged from original) ─────────────────────────────────────

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
        if column_inserted < 0:
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

    def undo(self, col):
        """Remove the top piece from a column, used by minimax to backtrack."""
        column = self.col[col]
        for i in range(len(column)):
            if not column[i].is_empty():
                column[i].color = "empty"
                self.red_turn = not self.red_turn
                return i
        raise ValueError("Column is already empty.")

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

    def is_draw(self):
        return all(not self.col[c][0].is_empty() for c in range(len(self.col)))

    def legal_cols(self):
        return [c for c in range(len(self.col)) if self.col[c][0].is_empty()]


# ── Minimax heuristic ─────────────────────────────────────────────────────────

def score_window(window: typing.List[str], color: str) -> int:
    """
    Score a window of 4 cells for one color.
    Higher = better for `color`.
    """
    opp = "blue" if color == "red" else "red"
    mine  = window.count(color)
    empty = window.count("empty")
    theirs = window.count(opp)

    if mine == 4:
        return 100
    if mine == 3 and empty == 1:
        return 5
    if mine == 2 and empty == 2:
        return 2
    if theirs == 3 and empty == 1:
        return -4   # block opponent's three-in-a-row
    return 0


def heuristic(game: Game, color: str) -> int:
    """
    Estimate how good the board is for `color` without searching further.
    Scores every horizontal, vertical, and diagonal window of 4.
    """
    num_rows = len(game.rows)
    num_cols = len(game.rows[0])
    score = 0

    # Prefer the centre column (more winning lines pass through it)
    center_col = num_cols // 2
    center = [game.rows[r][center_col].color for r in range(num_rows)]
    score += center.count(color) * 3

    # Horizontal windows
    for r in range(num_rows):
        for c in range(num_cols - 3):
            window = [game.rows[r][c + i].color for i in range(4)]
            score += score_window(window, color)

    # Vertical windows
    for c in range(num_cols):
        for r in range(num_rows - 3):
            window = [game.rows[r + i][c].color for i in range(4)]
            score += score_window(window, color)

    # Diagonal (down-right)
    for r in range(num_rows - 3):
        for c in range(num_cols - 3):
            window = [game.rows[r + i][c + i].color for i in range(4)]
            score += score_window(window, color)

    # Diagonal (down-left)
    for r in range(num_rows - 3):
        for c in range(3, num_cols):
            window = [game.rows[r + i][c - i].color for i in range(4)]
            score += score_window(window, color)

    return score


# ── Minimax with alpha-beta pruning ───────────────────────────────────────────

# Column order: search centre columns first — they tend to be stronger moves,
# which makes alpha-beta pruning cut off far more branches early.
MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]


def minimax(game: Game, depth: int, alpha: int, beta: int,
            maximising: bool, ai_color: str) -> int:
    """
    Returns the heuristic value of the current board position.

    maximising=True  → it's the AI's turn (wants to maximise score)
    maximising=False → it's the human's turn (wants to minimise score)
    alpha-beta pruning cuts branches that can't affect the final decision,
    making the search fast enough for depth 6 in real time.
    """
    opp_color = "blue" if ai_color == "red" else "red"
    legal = [c for c in MOVE_ORDER if c in game.legal_cols()]

    # ── Terminal checks ───────────────────────────────────────────────────
    # Check if the previous move was a win by seeing if any non-empty cell
    # at the top of a column produced a win. We check all columns that
    # could have been the last move.
    for c in range(len(game.col)):
        column = game.col[c]
        for slot in column:
            if not slot.is_empty():
                if game.check_win(slot.row, slot.col):
                    if slot.color == ai_color:
                        return 100_000 + depth   # win sooner = better
                    else:
                        return -100_000 - depth  # lose later = better
                break  # only the top-most piece matters per column

    if not legal or game.is_draw():
        return 0

    if depth == 0:
        return heuristic(game, ai_color)

    # ── Recursive search ──────────────────────────────────────────────────
    if maximising:
        best = -10**9
        for col in legal:
            game.insert(col)
            best = max(best, minimax(game, depth - 1, alpha, beta, False, ai_color))
            game.undo(col)
            alpha = max(alpha, best)
            if alpha >= beta:
                break   # β cut-off — opponent would never allow this branch
        return best
    else:
        best = 10**9
        for col in legal:
            game.insert(col)
            best = min(best, minimax(game, depth - 1, alpha, beta, True, ai_color))
            game.undo(col)
            beta = min(beta, best)
            if alpha >= beta:
                break   # α cut-off — we would never allow this branch
        return best


def best_move(game: Game, ai_color: str, depth: int = 6) -> int:
    """
    Returns the column the AI should play.
    depth=6 is strong and fast; increase to 7-8 for a harder challenge
    (noticeably slower on CPU).
    """
    best_score = -10**9
    best_col   = game.legal_cols()[0]

    for col in [c for c in MOVE_ORDER if c in game.legal_cols()]:
        game.insert(col)
        score = minimax(game, depth - 1, -10**9, 10**9, False, ai_color)
        game.undo(col)
        if score > best_score:
            best_score = score
            best_col   = col

    return best_col


# ── Game loop ─────────────────────────────────────────────────────────────────

def gameloop():
    game = Game()

    # Let the player choose their colour
    while True:
        choice = input("Do you want to be red or blue? (red goes first) ").strip().lower()
        if choice in ("red", "blue"):
            human_color = choice
            ai_color    = "blue" if human_color == "red" else "red"
            break
        print("Please type 'red' or 'blue'.")

    print(f"\nYou are {human_color}. AI is {ai_color}. Red goes first.\n")

    while game.running:
        game.print_state()
        current_color = game.return_turn_color()

        if current_color == human_color:
            # ── Human turn ────────────────────────────────────────────────
            while True:
                try:
                    col = int(input("Your move (0-6): "))
                    if col < 0 or col >= 7:
                        raise ValueError
                    row = game.insert(col)
                    break
                except ValueError:
                    print("Invalid column — try again.")
        else:
            # ── AI turn ───────────────────────────────────────────────────
            print("AI is thinking...")
            col = best_move(game, ai_color)
            row = game.insert(col)
            print(f"AI plays column {col}.")

        game.turn += 1

        if game.check_win(row, col):
            game.print_state()
            winner = "You win!" if current_color == human_color else "AI wins!"
            print(winner)
            game.running = False
        elif game.is_draw():
            game.print_state()
            print("It's a draw!")
            game.running = False


if __name__ == "__main__":
    gameloop()