# Our Ai imports:
import torch
import torch.nn as nn

# Importing the other class:
import fourinarow as fourinarow

# Utility imports:
import random
from collections import deque


# Ai Evaluation functions:
def count_line(game: fourinarow.Game, row, col, dr, dc, color):
    """
    This function counts the max number of a connected slots with the same color and how many open ends it has.
    """
    num_rows = len(game.rows)
    num_cols = len(game.rows[0])
    count = 1

    forward_open = False
    fr, fc = row + dr, col + dc
    while 0 <= fr < num_rows and 0 <= fc < num_cols and game.rows[fr][fc].color == color:
        count += 1
        fr += dr
        fc += dc
    if 0 <= fr < num_rows and 0 <= fc < num_cols and game.rows[fr][fc].color == "empty":
        forward_open = True

    backward_open = False
    br, bc = row - dr, col - dc
    while 0 <= br < num_rows and 0 <= bc < num_cols and game.rows[br][bc].color == color:
        count += 1
        br -= dr
        bc -= dc
    if 0 <= br < num_rows and 0 <= bc < num_cols and game.rows[br][bc].color == "empty":
        backward_open = True

    open_ends = int(forward_open) + int(backward_open)
    return count, open_ends


# General
def evaluate_board(game: fourinarow.Game, row, col):
    """
    This function evaluates the last move of an AI and estimates how good it was.
    Rewards are capped at 0.99 so they never exceed the win signal.
    """
    color = game.rows[row][col].color
    opponent_color = "blue" if color == "red" else "red"

    num_rows = len(game.rows)
    num_cols = len(game.rows[0])
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    if game.check_win(row, col):
        return 1.0

    total_reward = 0.0

    center_col = num_cols // 2
    if col == center_col:
        total_reward += 0.05

    my_threats = 0
    opp_threats = 0

    for dr, dc in directions:
        my_count, my_open = count_line(game, row, col, dr, dc, color)
        if my_count >= 3:
            if my_open == 2:
                total_reward += 0.5
            elif my_open == 1:
                total_reward += 0.3
            if my_open >= 1:
                my_threats += 1
        elif my_count == 2:
            if my_open == 2:
                total_reward += 0.15
            elif my_open == 1:
                total_reward += 0.05

        for sign in (1, -1):
            nr, nc = row + sign * dr, col + sign * dc
            if not (0 <= nr < num_rows and 0 <= nc < num_cols):
                continue
            if game.rows[nr][nc].color != opponent_color:
                continue
            opp_count = 1
            fr, fc = nr + sign * dr, nc + sign * dc
            while 0 <= fr < num_rows and 0 <= fc < num_cols and game.rows[fr][fc].color == opponent_color:
                opp_count += 1
                fr += sign * dr
                fc += sign * dc
            opp_forward_open = (
                    0 <= fr < num_rows and 0 <= fc < num_cols
                    and game.rows[fr][fc].color == "empty"
            )
            opp_open = int(opp_forward_open)  # backward end is blocked by our piece

            if opp_count >= 3:
                if opp_open == 2:
                    total_reward += 0.45
                elif opp_open == 1:
                    total_reward += 0.3
                opp_threats += 1
            elif opp_count == 2:
                if opp_open == 2:
                    total_reward += 0.1
                elif opp_open == 1:
                    total_reward += 0.05

    if my_threats >= 2:
        total_reward += 0.8
    if opp_threats >= 2:
        total_reward += 0.45

    return min(total_reward, 0.99)


# ── Model ─────────────────────────────────────────────────────────────────────

class FourInARowAi(nn.Module):
    def __init__(self):
        super().__init__()

        # FIX: halved channel counts — plenty for a 6x7 board, much faster
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
        return self.head(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── Utility function ──────────────────────────────────────────────────────────

def array_to_tensor(arr, device):
    red, empty, blue = arr
    # FIX: use torch.as_tensor instead of torch.tensor to avoid an unnecessary
    # data copy when the input is already a numpy array
    return torch.as_tensor(
        [red, blue, empty],
        dtype=torch.float32,
        device=device,
    ).reshape(3, 6, 7)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(episodes=100_000):
    # FIX: use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = FourInARowAi().to(device)
    target_model = FourInARowAi().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer()

    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.9995
    TARGET_UPDATE = 200
    LEARN_EVERY = 8  # FIX: only run a gradient update every 4 steps

    step = 0  # FIX: global step counter for LEARN_EVERY

    for episode in range(episodes):
        game = fourinarow.Game()
        done = False
        pending = {}  # color -> (state, action)

        while not done:
            current_color = "red" if game.red_turn else "blue"
            state = array_to_tensor(game.game_to_array(), device)

            # ── Pick action (epsilon-greedy) ──────────────────────────────
            legal = [c for c in range(7) if game.col[c][0].is_empty()]
            if random.random() < EPSILON:
                action = random.choice(legal)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    mask = torch.full((7,), float('-inf'), device=device)
                    mask[legal] = 0
                    action = torch.argmax(q_values + mask).item()

            # ── Apply action ──────────────────────────────────────────────
            try:
                row = game.insert(action)
            except ValueError:
                buffer.push(state, action, -0.1, state, False)
                continue

            # ── Calculate reward ──────────────────────────────────────────
            won = game.check_win(row, action)
            draw = not won and all(not game.col[c][0].is_empty() for c in range(7))
            done = won or draw

            reward = evaluate_board(game, row, action)
            next_state = array_to_tensor(game.game_to_array(), device)

            buffer.push(state, action, reward, next_state, done)

            if won:
                other_color = "blue" if current_color == "red" else "red"
                if other_color in pending:
                    opp_state, opp_action = pending[other_color]
                    buffer.push(opp_state, opp_action, -1.0, next_state, True)

            pending[current_color] = (state, action)
            state = next_state
            step += 1

            # ── Learn from replay buffer ──────────────────────────────────
            # FIX: only learn every LEARN_EVERY steps instead of every step
            if len(buffer) >= BATCH_SIZE and step % LEARN_EVERY == 0:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

                # FIX: move sampled batch to device
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_model(next_states).max(1).values
                    target_q = rewards + GAMMA * next_q * (1 - dones)

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        # ── Decay exploration rate ────────────────────────────────────────
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        # ── Sync target network ───────────────────────────────────────────
        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())
            print(f"Episode {episode}, epsilon: {EPSILON:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "episode": episodes,
        "epsilon": EPSILON,
    }, "model.pth")
    print("Training complete. Model saved to model.pth")


if __name__ == "__main__":
    train()

# ── Evaluation ────────────────────────────────────────────────────────────────

# model.eval()
# score = 0
# game = fourinarow.Game()
#
# with torch.no_grad():
#     for _ in range(500):
#         num_1 = random.randint(0, 15)
#         num_2 = random.randint(0, 15)
#
#         state = torch.tensor(num_to_binary(num_1) + num_to_binary(num_2), dtype=torch.float32)
#         logits = model(state)
#         prediction = torch.argmax(logits).item()  # pick the highest scoring answer
#
#         if prediction == num_1 ^ num_2:
#             score += 1





# print(f"Final score: {score}/500 ({score/5:.1f}%)")