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
import time
import argparse

# Import our custom modules
from n_tuple_network import NTupleNetwork
from mcts_search import MCTS
from mcts_node import StateNode, AfterStateNode
from game_env import Game2048Env  # Import the Game2048Env class from game_env.py

# --- Student Agent Implementation ---

# MCTS Agent for 2048 game with Multi-Stage weights
class MCTSAgent:
    def __init__(self, env, weights_dir=None, iterations=50, exploration_weight=1.41, v_norm=400000, time_limit=None):
        """
        Initialize the MCTS Agent with multi-stage weights support.

        Args:
            env: Game environment
            weights_dir: Directory containing stage weights files
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
        self.weights_dir = weights_dir
        self.current_stage = 1  # Default to stage 1
        self.stage_weights = {}  # Dictionary to store loaded weights for each stage
        self.last_stage_change_time = 0  # To prevent too frequent stage changes

        # Stage conditions based on tile values - only 2 stages now
        self.stage_conditions = [
            lambda s: self._has_tile(s, 1024),   # Stage 1 -> 2: First 1024 tile
        ]

        # Load all stage weights if directory is provided
        if weights_dir and os.path.exists(weights_dir):
            self._load_all_stage_weights()
            # Initially load stage 1 weights
            self._set_weights_for_stage(1)
            print(f"Loaded multi-stage weights from {weights_dir}")
        else:
            print("No weights directory provided or directory not found. Using default weights.")

        # Initialize MCTS
        self.mcts = MCTS(self.network, self.env, iterations, exploration_weight, v_norm)

    def _load_all_stage_weights(self):
        """Load only Stage 1 weights into memory"""
        # Only load Stage 1 weights
        weights_file = os.path.join(self.weights_dir, "stage1_weights.pkl")
        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'rb') as f:
                    self.stage_weights[1] = pickle.load(f)
                print("Loaded stage 1 weights")
            except Exception as e:
                print(f"Error loading stage 1 weights: {e}")
        else:
            print(f"Stage 1 weights file not found at {weights_file}")

        # Create dummy weights for stage 1 if not found
        if 1 not in self.stage_weights:
            print("Creating dummy weights for stage 1")
            self.stage_weights[1] = {
                'luts': [{} for _ in range(8)],
                'large_tile_lut': {},
                'empty_cells_lut': {},
                'tile_types_lut': {},
                'mergeable_pairs_lut': {},
                'v_2v_pairs_lut': {}
            }

    def _has_tile(self, state, value):
        """Check if the state has a tile with the specified value"""
        has_tile = np.any(state == value)
        if has_tile:
            print(f"Found tile with value {value}")
        return has_tile

    def _determine_stage(self, state):
        """Always return stage 1 regardless of the state"""
        # Always use stage 1 weights
        return 1

    def _set_weights_for_stage(self, stage):
        """Set the network weights for the specified stage"""
        if stage in self.stage_weights:
            # Apply the weights directly to the network
            self.network.luts = self.stage_weights[stage]['luts']
            self.network.large_tile_lut = self.stage_weights[stage]['large_tile_lut']
            self.network.empty_cells_lut = self.stage_weights[stage]['empty_cells_lut']
            self.network.tile_types_lut = self.stage_weights[stage]['tile_types_lut']
            self.network.mergeable_pairs_lut = self.stage_weights[stage]['mergeable_pairs_lut']
            self.network.v_2v_pairs_lut = self.stage_weights[stage]['v_2v_pairs_lut']

            self.current_stage = stage
            print(f"Switched to stage {stage} weights")
            return True
        else:
            print(f"No weights available for stage {stage}")
            return False

    def get_action(self, state):
        """
        Get the best action for the current state using MCTS with Stage 1 weights only.

        Args:
            state: The current game state

        Returns:
            Best action according to MCTS
        """
        # Always use stage 1 weights
        if self.current_stage != 1:
            self._set_weights_for_stage(1)
            self.current_stage = 1

        # Set the environment state for MCTS search
        # This ensures MCTS operates on the correct state without modifying the global env
        self.mcts.env.board = state.copy()

        # Run MCTS search with optional time limit
        return self.mcts.search(state, time_limit=self.time_limit)

# Global variables to store the loaded model
mcts_agent = None

def initialize_agent(use_dummy_weights=False):
    """
    Initialize the agent with all necessary components.

    Args:
        use_dummy_weights (bool): If True, use dummy weights instead of trained weights
    """
    global mcts_agent

    # Initialize the environment
    env = Game2048Env()

    # Initialize the MCTS agent

    # 使用相對於當前文件的路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, "weights")

    # Ensure weights directory exists
    os.makedirs(weights_dir, exist_ok=True)

    # Only handle Stage 1 weights
    weights_file = os.path.join(weights_dir, "stage1_weights.pkl")
    backup_file = weights_file + ".backup"

    if use_dummy_weights:
        print("Using dummy weights for Stage 1")
        # Create dummy weights for Stage 1
        dummy_weights = {
            'luts': [{} for _ in range(8)],
            'large_tile_lut': {},
            'empty_cells_lut': {},
            'tile_types_lut': {},
            'mergeable_pairs_lut': {},
            'v_2v_pairs_lut': {}
        }
        # Backup original weights if they exist and not already backed up
        if os.path.exists(weights_file) and not os.path.exists(backup_file):
            try:
                with open(weights_file, 'rb') as f:
                    original_weights = pickle.load(f)
                with open(backup_file, 'wb') as f:
                    pickle.dump(original_weights, f)
                print("Backed up original Stage 1 weights")
            except Exception as e:
                print(f"Error backing up Stage 1 weights: {e}")
        # Write dummy weights
        with open(weights_file, 'wb') as f:
            pickle.dump(dummy_weights, f)
    else:
        # Check if Stage 1 weights file exists, if not, create dummy one
        # If backup exists and we're not using dummy weights, restore from backup
        if os.path.exists(backup_file):
            try:
                with open(backup_file, 'rb') as f:
                    backup_weights = pickle.load(f)
                with open(weights_file, 'wb') as f:
                    pickle.dump(backup_weights, f)
                print("Restored original Stage 1 weights from backup")
            except Exception as e:
                print(f"Error restoring Stage 1 weights: {e}")
        elif not os.path.exists(weights_file):
            print(f"Stage 1 weights file not found. Creating a dummy weights file.")
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
        weights_dir=weights_dir,
        iterations=50,
        exploration_weight=1.41,
        v_norm=400000,
        time_limit=None  # Set to a value like 0.5 to limit search time to 0.5 seconds
    )

# Agent will be initialized on first call to get_action

def get_action(state, score):
    """
    Main agent function to decide the next action using multi-stage weights.

    Args:
        state (np.array): Current game state
        score (int): Current score (not used but kept for API compatibility)

    Returns:
        int: Selected action (0: up, 1: down, 2: left, 3: right)
    """
    global mcts_agent

    # If the agent is not initialized, initialize it
    if mcts_agent is None:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='2048 MCTS Agent')
        parser.add_argument('--dummy', action='store_true', help='Use dummy weights instead of trained weights')
        args, unknown = parser.parse_known_args()

        initialize_agent(use_dummy_weights=args.dummy)

    # Use MCTS to find the best action with dynamic stage selection
    # The agent's get_action method now handles setting the environment state and selecting the appropriate stage
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='2048 MCTS Agent')
    parser.add_argument('--dummy', action='store_true', help='Use dummy weights instead of trained weights')
    args = parser.parse_args()

    # Initialize the agent with the specified weights option
    initialize_agent(use_dummy_weights=args.dummy)

    env = Game2048Env()
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print("Starting MCTS agent test...")
    # env.render()  # Show initial board (disabled for headless environment)

    # Print initial board state
    print("Initial board state:")
    for row in state:
        print(row)
    print("")

    while not done:
        # Get action from our agent function
        action = get_action(state, env.score)

        # Perform the action in the environment
        next_state, reward, done, info = env.step(action)

        # Update state and total reward
        state = next_state
        total_reward += reward
        step_count += 1

        # Print board state and move information
        print(f"Step: {step_count}, Action: {env.actions[action]}, Reward: {reward}, Total Score: {env.score}")
        print("Current board state:")
        for row in state:
            print(row)
        print("")

        if done:
            print("\nGame Over!")
            print(f"Final Score: {env.score}")
            print(f"Total Steps: {step_count}")
            print(f"Highest Tile: {np.max(env.board)}")
            # env.render()  # Show final board (disabled for headless environment)

    print("MCTS agent test finished.")

