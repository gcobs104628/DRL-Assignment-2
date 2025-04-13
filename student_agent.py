# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math
import os

# Import our custom modules
from n_tuple_network import NTupleNetwork
from mcts_search import MCTS
from mcts_node import StateNode, AfterStateNode
from game_env import Game2048Env  # Import the Game2048Env class from game_env.py

# --- Student Agent Implementation ---

# MCTS Agent for 2048 game
class MCTSAgent:
    def __init__(self, env, weights_file=None, iterations=50, exploration_weight=0.1, v_norm=400000, time_limit=None):
        """
        Initialize the MCTS Agent.

        Args:
            env: Game environment
            weights_file: Path to the trained weights file
            iterations: Number of MCTS iterations to run
            exploration_weight: Exploration constant for UCB1 formula
            v_norm: Normalization constant for value estimates
            time_limit: Optional time limit for MCTS search in seconds
        """
        self.env = env
        self.network = NTupleNetwork()
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.v_norm = v_norm
        self.time_limit = time_limit

        # Load weights if provided
        if weights_file and os.path.exists(weights_file):
            self.network.load_weights(weights_file)
            print(f"Loaded weights from {weights_file}")
        else:
            print("No weights file provided or file not found. Using default weights.")

        # Initialize MCTS
        self.mcts = MCTS(self.network, self.env, iterations, exploration_weight, v_norm)

    def get_action(self, state):
        """
        Get the best action for the current state using MCTS.

        Args:
            state: The current game state

        Returns:
            Best action according to MCTS
        """
        # Set the environment state for MCTS search
        # This ensures MCTS operates on the correct state without modifying the global env
        self.mcts.env.board = state.copy()

        # Run MCTS search with optional time limit
        return self.mcts.search(state, time_limit=self.time_limit)

# Global variables to store the loaded model
mcts_agent = None

def initialize_agent():
    """
    Initialize the agent with all necessary components.
    """
    global mcts_agent

    # Initialize the environment
    env = Game2048Env()

    # Initialize the MCTS agent

    # 在 initialize_agent() 函數中
    # 使用相對於當前文件的路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用第5階段的模型權重，因為它是最完整的訓練結果
    weights_file = os.path.join(current_dir, "weights", "stage5_weights.pkl")

    # Check if weights file exists, if not, create a dummy one
    if not os.path.exists(weights_file):
        print(f"Weights file {weights_file} not found. Creating a dummy weights file.")
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        dummy_weights = {
            'luts': [{} for _ in range(8)],
            'large_tile_lut': {},
            'empty_cells_lut': {},
            'tile_types_lut': {},
            'mergeable_pairs_lut': {},
            'v_2v_pairs_lut': {}
        }
        with open(weights_file, 'wb') as f:
            pickle.dump(dummy_weights, f)

    # Create the MCTS agent with configurable parameters
    mcts_agent = MCTSAgent(
        env=env,
        weights_file=weights_file,
        iterations=50,
        exploration_weight=0.1,
        v_norm=400000,
        time_limit=None  # Set to a value like 0.5 to limit search time to 0.5 seconds
    )

# Initialize the agent
initialize_agent()

def get_action(state, score):
    """
    Main agent function to decide the next action.

    Args:
        state (np.array): Current game state
        score (int): Current score (not used but kept for API compatibility)

    Returns:
        int: Selected action (0: up, 1: down, 2: left, 3: right)
    """
    global mcts_agent

    # If the agent is not initialized, initialize it
    if mcts_agent is None:
        initialize_agent()

    # Use MCTS to find the best action
    # The agent's get_action method now handles setting the environment state
    action = mcts_agent.get_action(state)

    return action

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
    temp_env.board = state.copy()  # Use a copy to avoid modifying the original state

    # Find all legal moves
    legal_moves = []
    for action in range(temp_env.action_space.n):
        if temp_env.is_move_legal(action):
            legal_moves.append(action)

    if legal_moves:
        return random.choice(legal_moves)
    else:
        # If no moves are legal, the game is likely over or stuck.
        # Returning a random action is a fallback.
        # The environment's step function should handle the game over state correctly.
        print("Warning: No legal moves found. Returning random action.")
        return random.choice([0, 1, 2, 3])  # Fallback: choose any action if none are legal

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    env = Game2048Env()
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print("Starting MCTS agent test...")
    # env.render()  # Show initial board (disabled for headless environment)

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
            print(f"Highest Tile: {np.max(env.board)}")
            # env.render()  # Show final board (disabled for headless environment)

    print("MCTS agent test finished.")

