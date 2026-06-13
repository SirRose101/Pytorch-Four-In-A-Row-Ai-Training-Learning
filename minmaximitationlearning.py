import typing
import random
import torch
import torch.nn as nn

# ── Game logic ────────────────────────────────────────────────────────────────

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
            raise ValueError("Invalid column.")
        column = self.col[column_inserted]
        if not column[0].is_empty():
            raise ValueError("Column is full.")
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
        column = self.col[col]
        for slot in column:
            if not slot.is_empty():
                slot.color = "empty"
                self.red_turn = not self.red_turn
                return
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

    def game_to_array(self):
        """
        Returns board from the current player's perspective:
        channel 0 = my pieces, channel 1 = empty, channel 2 = opponent pieces.
        """
        my_color  = self.return_turn_color()
        opp_color = "blue" if my_color == "red" else "red"
        mine, empty, opp = [], [], []
        for row in self.rows:
            for slot in row:
                mine.append(1 if slot.color == my_color  else 0)
                empty.append(1 if slot.color == "empty"  else 0)
                opp.append(1  if slot.color == opp_color else 0)
        return mine, empty, opp


# ── Minimax teacher ───────────────────────────────────────────────────────────

MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]


def score_window(window: typing.List[str], color: str) -> int:
    opp   = "blue" if color == "red" else "red"
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
        return -4
    return 0


def heuristic(game: Game, color: str) -> int:
    num_rows = len(game.rows)
    num_cols = len(game.rows[0])
    score = 0
    center_col = num_cols // 2
    center = [game.rows[r][center_col].color for r in range(num_rows)]
    score += center.count(color) * 3
    for r in range(num_rows):
        for c in range(num_cols - 3):
            window = [game.rows[r][c + i].color for i in range(4)]
            score += score_window(window, color)
    for c in range(num_cols):
        for r in range(num_rows - 3):
            window = [game.rows[r + i][c].color for i in range(4)]
            score += score_window(window, color)
    for r in range(num_rows - 3):
        for c in range(num_cols - 3):
            window = [game.rows[r + i][c + i].color for i in range(4)]
            score += score_window(window, color)
    for r in range(num_rows - 3):
        for c in range(3, num_cols):
            window = [game.rows[r + i][c - i].color for i in range(4)]
            score += score_window(window, color)
    return score


def minimax(game: Game, depth: int, alpha: int, beta: int,
            maximising: bool, ai_color: str) -> int:
    legal = [c for c in MOVE_ORDER if c in game.legal_cols()]

    for c in range(len(game.col)):
        for slot in game.col[c]:
            if not slot.is_empty():
                if game.check_win(slot.row, slot.col):
                    if slot.color == ai_color:
                        return 100_000 + depth
                    else:
                        return -100_000 - depth
                break

    if not legal or game.is_draw():
        return 0
    if depth == 0:
        return heuristic(game, ai_color)

    if maximising:
        best = -10**9
        for col in legal:
            game.insert(col)
            best = max(best, minimax(game, depth - 1, alpha, beta, False, ai_color))
            game.undo(col)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best
    else:
        best = 10**9
        for col in legal:
            game.insert(col)
            best = min(best, minimax(game, depth - 1, alpha, beta, True, ai_color))
            game.undo(col)
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best


def best_move(game: Game, ai_color: str, depth: int = 5) -> int:
    """Returns the best column according to minimax."""
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


# ── Neural network ────────────────────────────────────────────────────────────

class FourInARowAi(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(64 * 6 * 7, 7)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)  # raw logits over 7 columns


# ── Tensor utility ────────────────────────────────────────────────────────────

def game_to_tensor(game: Game, device) -> torch.Tensor:
    mine, empty, opp = game.game_to_array()
    return torch.tensor(
        [mine, empty, opp], dtype=torch.float32, device=device
    ).reshape(3, 6, 7)


# ── Data generation ───────────────────────────────────────────────────────────

def generate_game(teacher_depth: int = 5):
    """
    Play one full game where both sides use minimax, but with a small chance
    of a random move each turn. This injects variety so the dataset covers
    many different board positions, not just the single line of perfect play.

    Returns a list of (board_array, best_col) training pairs — one per move.
    """
    game   = Game()
    samples = []
    RANDOM_MOVE_PROB = 0.15   # 15 % of moves are random for diversity

    while True:
        color  = game.return_turn_color()
        legal  = game.legal_cols()

        # Record the board state BEFORE the move (from current player's view)
        board = game.game_to_array()

        # Ask minimax for the best move regardless — this is our label
        label_col = best_move(game, color, depth=teacher_depth)

        # Occasionally play a random move instead to create varied positions,
        # but always label it with what minimax WOULD have played.
        if random.random() < RANDOM_MOVE_PROB:
            play_col = random.choice(legal)
        else:
            play_col = label_col

        samples.append((board, label_col))

        row = game.insert(play_col)

        if game.check_win(row, play_col) or game.is_draw():
            break

    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    num_games: int     = 10_000,  # number of self-play games to generate
    teacher_depth: int = 5,       # minimax depth used as the teacher
    batch_size: int    = 256,
    lr: float          = 1e-3,
    report_every: int  = 100,     # print progress every N games
):
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    print(f"Generating {num_games} games at minimax depth {teacher_depth}...\n")

    model     = FourInARowAi().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Dataset ───────────────────────────────────────────────────────────
    # We generate all games up front and store them in a buffer.
    # Each entry is (3×6×7 tensor, label_column).
    dataset: typing.List[typing.Tuple[torch.Tensor, int]] = []

    gen_start = time.time()

    for game_idx in range(num_games):
        samples = generate_game(teacher_depth)
        for board, label in samples:
            mine, empty, opp = board
            tensor = torch.tensor(
                [mine, empty, opp], dtype=torch.float32
            ).reshape(3, 6, 7)
            dataset.append((tensor, label))

        if (game_idx + 1) % report_every == 0:
            elapsed    = time.time() - gen_start
            pace       = elapsed / (game_idx + 1)
            remaining  = pace * (num_games - game_idx - 1)
            mins, secs = divmod(int(remaining), 60)
            print(f"  [{game_idx + 1:5d} / {num_games}] "
                  f"{len(dataset):6d} positions | "
                  f"{pace:.2f}s/game | "
                  f"ETA {mins}m {secs:02d}s")

    print(f"\nDataset ready: {len(dataset)} positions.")
    print("Starting supervised training...\n")

    # ── Training loop ─────────────────────────────────────────────────────
    # We train for multiple passes (epochs) over the dataset.
    # CrossEntropyLoss is the right loss here: we're doing 7-class
    # classification (which column is best?), not regression.
    loss_fn = nn.CrossEntropyLoss()
    epochs  = 20

    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss    = 0.0
        total_correct = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            states  = torch.stack([s for s, _ in batch]).to(device)
            labels  = torch.tensor([l for _, l in batch],
                                   dtype=torch.long, device=device)

            logits = model(states)
            loss   = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * len(batch)
            total_correct += (logits.argmax(1) == labels).sum().item()

        avg_loss = total_loss    / len(dataset)
        accuracy = total_correct / len(dataset) * 100
        print(f"Epoch {epoch + 1:2d}/{epochs} | "
              f"loss {avg_loss:.4f} | "
              f"accuracy {accuracy:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, "model_supervised.pth")
    print("\nTraining complete. Model saved to model_supervised.pth")


# ── Play against the trained model ────────────────────────────────────────────

def play(model_path: str = "model_supervised.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FourInARowAi().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    game = Game()

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
            with torch.no_grad():
                state  = game_to_tensor(game, device)
                logits = model(state).squeeze(0)
                # Mask illegal columns so the AI never picks a full one
                legal  = game.legal_cols()
                mask   = torch.full((7,), float('-inf'), device=device)
                mask[legal] = 0
                col    = torch.argmax(logits + mask).item()
            row = game.insert(col)
            print(f"AI plays column {col}.")

        game.turn += 1

        if game.check_win(row, col):
            game.print_state()
            print("You win!" if current_color == human_color else "AI wins!")
            game.running = False
        elif game.is_draw():
            game.print_state()
            print("It's a draw!")
            game.running = False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play()
    else:
        train(teacher_depth=5)