# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
# import random # Already imported above
import math

# --- Provided Game2048Env Class ---
# (Assuming the Game2048Env class code provided in the prompt is here)
# Color mapping for rendering (Optional, but needed if render() is called)
COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32" # Higher tiles use a dark background
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", # Dark text for light tiles
    8: "#f9f6f2", 16: "#f9f6f2", 32: "#f9f6f2", 64: "#f9f6f2", # Light text for colored tiles
    128: "#f9f6f2", 256: "#f9f6f2", 512: "#f9f6f2", 1024: "#f9f6f2",
    2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        # Return the state (board) consistent with gym interface
        return self.board.copy() # Return a copy to prevent external modification

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        reward_increase = 0 # Track score increase for this merge step
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                # self.score += row[i] # Score update should happen based on the move's total effect
                reward_increase += row[i] # Accumulate reward locally
        return row, reward_increase

    def move_left(self):
        """Move the board left"""
        moved = False
        current_move_reward = 0
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row, reward1 = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            current_move_reward += reward1 # Accumulate reward from merging
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved, current_move_reward

    def move_right(self):
        """Move the board right"""
        moved = False
        current_move_reward = 0
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row, reward1 = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            current_move_reward += reward1
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved, current_move_reward

    def move_up(self):
        """Move the board up"""
        moved = False
        current_move_reward = 0
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col, reward1 = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            current_move_reward += reward1
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved, current_move_reward

    def move_down(self):
        """Move the board down"""
        moved = False
        current_move_reward = 0
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col, reward1 = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            current_move_reward += reward1
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved, current_move_reward

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally for possible merges
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically for possible merges
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        # No empty cells and no possible merges means game over
        return True

    def step(self, action):
        """
        Execute one action.
        Returns:
            observation (np.array): The new board state.
            reward (float): The reward obtained from this step (score increase).
            done (bool): Whether the game is over.
            info (dict): Auxiliary information (empty in this case).
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Store score before move to calculate reward
        score_before_move = self.score
        
        moved = False
        reward = 0

        if action == 0: # Up
            moved, reward = self.move_up()
        elif action == 1: # Down
            moved, reward = self.move_down()
        elif action == 2: # Left
            moved, reward = self.move_left()
        elif action == 3: # Right
            moved, reward = self.move_right()

        self.last_move_valid = moved  # Record if the move was valid
        self.score += reward # Update total score

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        # Return observation (state), reward, done, info tuple
        return self.board.copy(), reward, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None and action < len(self.actions):
             title += f" | action: {self.actions[action]}"
        elif action is not None:
             title += f" | action: {action} (invalid)" # Handle potential invalid action display

        plt.title(title)
        plt.gca().invert_yaxis() # Make (0,0) top-left corner
        plt.show()

    def simulate_row_move(self, row):
        """
        Simulate a left move for a single row without changing score or board.
        Used only for checking if a move is legal.
        Returns the potentially changed row.
        """
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """
        Check if the specified move is legal (i.e., changes the board state).
        Does *not* modify the actual board or score.
        """
        # Create a temporary copy of the board to simulate the move
        temp_board = self.board.copy()
        original_board_snapshot = self.board.copy() # Keep original state safe

        if action == 0:  # Simulate Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Simulate Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Simulate Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Simulate Move right
            for i in range(self.size):
                # Reverse the row, simulate, then reverse back
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            # This case should not be reached if action comes from action_space
            return False # Or raise an error, but returning False is safer

        # Restore original board just in case (good practice)
        # self.board = original_board_snapshot # Not strictly needed as we used temp_board

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(original_board_snapshot, temp_board)

# --- Student Agent Implementation ---

def get_random_action(state):
    """
    Gets a random *legal* action for the given state.
    Args:
        state (np.array): The current board state.
    Returns:
        int: A random legal action (0, 1, 2, or 3).
             Returns a random action among all if no legal move is found (game over state).
    """
    # We need an environment instance to use the is_move_legal method.
    # Create a temporary instance and set its state.
    temp_env = Game2048Env()
    temp_env.board = state.copy() # Use a copy to avoid modifying the original state

    legal_moves = [action for action in range(temp_env.action_space.n) if temp_env.is_move_legal(action)]

    if legal_moves:
        return random.choice(legal_moves)
    else:
        # If no moves are legal, the game is likely over or stuck.
        # Returning a random action (e.g., 0) is a fallback.
        # The environment's step function should handle the game over state correctly.
        # print("Warning: No legal moves found. Returning random action 0.") # Optional debug print
        return random.choice([0, 1, 2, 3]) # Fallback: choose any action if none are legal

def get_action(state, score):
    """
    The main function for the student agent.
    Currently, it returns a random legal action, ignoring the score.

    Args:
        state (np.array): The current board state (observation).
        score (int): The current game score.

    Returns:
        int: The action chosen by the agent.
    """
    # For now, just call the random action function.
    # The score parameter is ignored in this simple random agent.
    return get_random_action(state)

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    env = Game2048Env()
    state = env.reset()
    done = False
    total_reward = -1
    step_count = 0

    print("Starting random agent test...")
    env.render() # Show initial board

    while not done:
        # Get action from our agent function
        action = get_action(state, env.score)

        # Perform the action in the environment
        next_state, reward, done, info = env.step(action)

        # Update state and total reward
        state = next_state
        total_reward += reward
        step_count += 1

        # Render the board (optional, can be slow)
        # print(f"Step: {step_count}, Action: {env.actions[action]}, Reward: {reward}, Total Score: {env.score}")
        # env.render(action=action) # Show board after action

        if done:
            print("\nGame Over!")
            print(f"Final Score: {env.score}")
            print(f"Total Steps: {step_count}")
            env.render() # Show final board

    print("Random agent test finished.")

