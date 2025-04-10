# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import random
import math
import os # For saving/loading LUTs
from collections import defaultdict # For easier LUT implementation

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left (vectorized)."""
        mask = row != 0
        new_row = row[mask]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row (vectorized)."""
        for i in range(self.size - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left (vectorized)."""
        original_board = self.board.copy()
        for i in range(self.size):
            self.board[i] = self.compress(self.merge(self.compress(self.board[i])))
        return not np.array_equal(original_board, self.board)

    def move_right(self):
        """Move the board right (vectorized)."""
        original_board = self.board.copy()
        self.board = np.fliplr(self.board)
        for i in range(self.size):
            self.board[i] = self.compress(self.merge(self.compress(self.board[i])))
        self.board = np.fliplr(self.board)
        return not np.array_equal(original_board, self.board)

    def move_up(self):
        """Move the board up (vectorized)."""
        original_board = self.board.copy()
        self.board = np.transpose(self.board)
        for i in range(self.size):
            self.board[i] = self.compress(self.merge(self.compress(self.board[i])))
        self.board = np.transpose(self.board)
        return not np.array_equal(original_board, self.board)

    def move_down(self):
        """Move the board down (vectorized)."""
        original_board = self.board.copy()
        self.board = np.transpose(np.flipud(self.board))
        for i in range(self.size):
            self.board[i] = self.compress(self.merge(self.compress(self.board[i])))
        self.board = np.flipud(np.transpose(self.board))
        return not np.array_equal(original_board, self.board)

    def is_game_over(self):
        """Check if there are no legal moves left (optimized)."""
        if np.any(self.board == 0):
            return False
        # Check horizontally
        if np.any(np.diff(self.board, axis=1) == 0):
            return False

        # Check vertically
        if np.any(np.diff(self.board, axis=0) == 0):
            return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        afterstate = self.board.copy()
        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, afterstate

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def _can_compress(self, row):
        """Check if a row can be compressed in a way that changes the board."""
        # Find the first zero
        first_zero_index = -1
        for i in range(self.size):
            if row[i] == 0:
                first_zero_index = i
                break
        
        if first_zero_index == -1:
            return False

        # Check if there is a non-zero number after the first zero
        for i in range(first_zero_index + 1, self.size):
            if row[i] != 0:
                return True
        return False

    def _can_merge(self, row):
        """Check if a row can be merged (has adjacent equal numbers)."""
        for i in range(self.size - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                return True
        return False

    def is_move_legal(self, action):
        """
        Check if the specified move is legal (i.e., changes the board) - Optimized.
        This version only checks if the move *would* change the board, without
        actually performing the full simulation.
        """
        if action == 0:  # Move up
            # Check if any column can be compressed or merged
            for j in range(self.size):
                col = self.board[:, j]
                if self._can_compress(col) or self._can_merge(col):
                    return True
            return False
        elif action == 1:  # Move down
            # Check if any column can be compressed or merged (reversed)
            for j in range(self.size):
                col = self.board[:, j][::-1]
                if self._can_compress(col) or self._can_merge(col):
                    return True
            return False
        elif action == 2:  # Move left
            # Check if any row can be compressed or merged
            for i in range(self.size):
                row = self.board[i]
                if self._can_compress(row) or self._can_merge(row):
                    return True
            return False
        elif action == 3:  # Move right
            # Check if any row can be compressed or merged (reversed)
            for i in range(self.size):
                row = self.board[i][::-1]
                if self._can_compress(row) or self._can_merge(row):
                    return True
            return False
        else:
            raise ValueError("Invalid action")


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(float) for _ in patterns]
        self.symmetries = [self.generate_symmetries(pattern) for pattern in self.patterns]
        try:
            self.load_luts()
            print("Loaded LUTs from ntuple_luts.pkl")
        except:
            print("Failed to load LUTs. Starting with empty LUTs.")

    def generate_symmetries(self, pattern):
        symmetries = [pattern]
        coords = np.array(pattern)

        # Rotations
        for _ in range(3):
            coords = np.array([(y, self.board_size - 1 - x) for x, y in coords])
            symmetries.append(tuple(map(tuple, coords)))

        # Horizontal flip
        coords = np.array([(x, self.board_size - 1 - y) for x, y in pattern])
        symmetries.append(tuple(map(tuple, coords)))

        # Vertical flip
        coords = np.array([(self.board_size - 1 - x, y) for x, y in pattern])
        symmetries.append(tuple(map(tuple, coords)))

        # Horizontal flip + rotation
        coords = np.array([(y, self.board_size - 1 - x) for x, y in np.array([(x, self.board_size - 1 - y) for x, y in pattern])])
        symmetries.append(tuple(map(tuple, coords)))

        # Vertical flip + rotation
        coords = np.array([(y, self.board_size - 1 - x) for x, y in np.array([(self.board_size - 1 - x, y) for x, y in pattern])])
        symmetries.append(tuple(map(tuple, coords)))

        return list(set(symmetries))

    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, pattern):  # Pass in pattern, not coords
        feature = tuple(self.tile_to_index(board[y, x]) for x, y in pattern)
        return feature

    def value(self, board):
        total_value = 0
        for i, pattern in enumerate(self.patterns): # Iterate through original patterns
            symmetries = self.symmetries[i] # Generate symmetries
            for sym_pattern in symmetries:
                feature = self.get_feature(board, sym_pattern) # Use symmetric pattern
                total_value += self.weights[i][feature] # Access weight with original pattern index
        return total_value

    def update(self, board, delta, alpha):
        for i, pattern in enumerate(self.patterns):
            symmetries = self.symmetries[i]
            for sym_pattern in symmetries:
                feature = self.get_feature(board, sym_pattern)
                self.weights[i][feature] += alpha * delta
    
    def save_luts(self, filename="ntuple_luts.pkl"):
        """ Save the learned LUTs to a file. """
        # Convert defaultdicts to regular dicts for pickling if needed (usually works fine)
        save_data = [dict(lut) for lut in self.weights]
        try:
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"LUTs saved to {filename}")
        except Exception as e:
            print(f"Error saving LUTs: {e}")

    def load_luts(self, filename="ntuple_luts.pkl"):
        """ Load LUTs from a file. """
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    loaded_data = pickle.load(f)
                # Convert loaded dicts back to defaultdicts
                self.weights = []
                for d in loaded_data:
                    lut = defaultdict(float)
                    lut.update(d)
                    self.weights.append(lut)
            except Exception as e:
                print(f"Error loading LUTs: {e}. Starting with empty LUTs.")
                # Fallback to default empty LUTs if loading fails
                self.luts = [defaultdict(float) for _ in range(self.num_tuples)]
        else:
            print(f"LUT file '{filename}' not found. Starting with empty LUTs.")
            self.luts = [defaultdict(float) for _ in range(self.num_tuples)]


def td_learning(env: Game2048Env, approximator, num_episodes=50000, alpha=0.01):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []
    td_errors = []

    for episode in range(num_episodes):
        state = env.reset()
        afterstate = state.copy()
        previous_score = 0
        done = False
        max_tile = np.max(state)
        single_episode_td_error = []
        trajectory = []

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            scores = []
            current_state = env.board.copy()
            current_score = env.score

            for action in legal_moves:
                next_state, new_score, done, next_afterstate = env.step(action)
                if done:
                    scores.append(0)
                else:
                    scores.append(new_score + approximator.value(next_afterstate))
                env.board = current_state.copy()
                env.score = current_score
                env.last_move_valid = True
                
            action = legal_moves[np.argmax(scores)]

            next_state, new_score, done, next_afterstate = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            trajectory.append((afterstate.copy(), incremental_reward, next_afterstate.copy()))

            state = next_state
            afterstate = next_afterstate

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        for t in trajectory:
            afterstate, incremental_reward, next_afterstate = t
            expected = approximator.value(afterstate)
            delta = incremental_reward + approximator.value(next_afterstate) - expected
            approximator.update(afterstate, delta, alpha)
            single_episode_td_error.append(abs(delta))
        
        td_errors.append(np.mean(single_episode_td_error))

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.mean(success_flags[-100:])
            td_error = np.mean(td_errors[-100:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | TD Error: {td_error:.2f}")

        if (episode + 1) % 1000 == 0:
            approximator.save_luts(f"ntuple_luts_{episode+1}.pkl")
            approximator.save_luts(f"ntuple_luts.pkl")
            print(f"Saved LUTs to ntuple_luts_{episode+1}.pkl")

    return final_scores


# TODO: Define your own n-tuple patterns
patterns = [
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)),
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)),
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
final_scores = td_learning(env, approximator, num_episodes=100000, alpha=0.01)
plt.plot(final_scores)
plt.xlabel('Episode')
plt.ylabel('Final Score')
plt.title('Final Score vs. Episode')
plt.show()