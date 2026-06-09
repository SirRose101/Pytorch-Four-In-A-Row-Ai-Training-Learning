import random
import torch
import fourinarow
from main import FourInARowAi


def load_model(path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FourInARowAi().to(device)  # FIX: move model to correct device
    checkpoint = torch.load(path, map_location=device)  # FIX: load weights onto the right device
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


def array_to_tensor(arr, device):
    red, empty, blue = arr
    return torch.as_tensor(
        [red, blue, empty],
        dtype=torch.float32,
        device=device,
    ).reshape(3, 6, 7)


def ai_pick(model, game, device):  # FIX: accept device as parameter
    state = array_to_tensor(game.game_to_array(), device)  # FIX: use array_to_tensor for correct shape
    with torch.no_grad():
        q_values = model(state).squeeze(0)  # FIX: squeeze to shape (7,) so indexing works correctly
    for col in range(7):
        if not game.col[col][0].is_empty():
            q_values[col] = float('-inf')
    return torch.argmax(q_values).item()


def play():
    model, device = load_model()  # FIX: unpack device from load_model
    game = fourinarow.Game()

    human_color = random.choice(["red", "blue"])
    ai_color = "blue" if human_color == "red" else "red"
    print(f"You are {human_color}. AI is {ai_color}.")
    print("Columns are numbered 0-6 left to right.\n")

    while game.running:
        game.print_state()
        current_color = game.return_turn_color()

        if current_color == human_color:
            # ── Human turn ────────────────────────────────────────────
            while True:
                try:
                    col = int(input("Your move (0-6): "))
                    row = game.insert(col)
                    break
                except ValueError:
                    print("That column is full or invalid, try again.")
        else:
            # ── AI turn ───────────────────────────────────────────────
            col = ai_pick(model, game, device)  # FIX: pass device through
            row = game.insert(col)
            print(f"AI played column {col}.")

        if game.check_win(row, col):
            game.print_state()
            if current_color == human_color:
                print("You win!")
            else:
                print("AI wins!")
            game.running = False

        elif all(not game.col[c][0].is_empty() for c in range(7)):
            game.print_state()
            print("It's a draw!")
            game.running = False


if __name__ == "__main__":
    play()