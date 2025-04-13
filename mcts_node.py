import numpy as np
import math

# Action constants for better readability
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
ACTION_NAMES = ["up", "down", "left", "right"]

class MCTSNode:
    """Base class for MCTS nodes"""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}  # Maps actions/tiles to child nodes
        self.visit_count = 0
        self.value_sum = 0.0
        self.value = 0.0  # Average value (Q)

    def update(self, value):
        """Update node statistics with a new value"""
        self.visit_count += 1
        self.value_sum += value
        self.value = self.value_sum / self.visit_count

    def is_fully_expanded(self):
        """Check if all possible children have been expanded"""
        raise NotImplementedError("Subclasses must implement this method")

    def select_child(self):
        """Select a child node according to the selection policy"""
        raise NotImplementedError("Subclasses must implement this method")

    def expand(self):
        """Expand by adding a new child node"""
        raise NotImplementedError("Subclasses must implement this method")

    def is_terminal(self):
        """Check if this node represents a terminal state"""
        raise NotImplementedError("Subclasses must implement this method")

class StateNode(MCTSNode):
    """
    Represents a state node (max node) in MCTS.
    This is a regular game state where the agent chooses an action.
    """
    def __init__(self, state, env, parent=None, cumulative_reward=0):
        super().__init__(parent)
        self.state = state.copy()  # The game state (4x4 numpy array)
        self.env = env  # Store reference to the environment
        self.cumulative_reward = cumulative_reward  # Accumulated reward from root to this node
        self.available_actions = None  # Will be set during expansion

    def is_fully_expanded(self):
        """Check if all legal actions have been tried"""
        if self.available_actions is None:
            return False
        return len(self.children) == len(self.available_actions)

    def select_child(self, exploration_weight=1.0):
        """
        Select a child using UCB1 formula.

        Args:
            exploration_weight: Controls exploration vs exploitation (c in UCB1)

        Returns:
            (action, child_node) tuple
        """
        # If not all children are expanded, expand a new one
        if not self.is_fully_expanded():
            return self.expand()

        # UCB1 formula: Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        log_visits = math.log(self.visit_count) if self.visit_count > 0 else 0

        def ucb_score(action_child):
            _, child = action_child  # action is unpacked but not used in the formula
            # Avoid division by zero
            if child.visit_count == 0:
                return float('inf')
            # UCB1 formula
            exploitation = child.value
            exploration = exploration_weight * math.sqrt(log_visits / child.visit_count)
            return exploitation + exploration

        # Return the action with the highest UCB score
        return max(self.children.items(), key=ucb_score)

    def expand(self):
        """
        Expand by adding a new afterstate node for an untried action.

        Returns:
            (action, new_child_node) tuple
        """
        # If available actions not yet determined, compute them
        if self.available_actions is None:
            # Set the environment state
            self.env.board = self.state.copy()

            # Find all legal actions
            self.available_actions = []
            for action in ACTIONS:  # Use action constants
                if self.env.is_move_legal(action):
                    self.available_actions.append(action)

            # If no legal actions, mark as terminal
            if not self.available_actions:
                return None  # No actions available

        # Find an untried action
        for action in self.available_actions:
            if action not in self.children:
                # Simulate the action to get the afterstate
                self.env.board = self.state.copy()
                afterstate, reward = self.env.get_afterstate(self.state, action)

                # Check if the action is valid (afterstate is not None)
                if afterstate is None:
                    continue  # Skip invalid actions

                # Create a new afterstate node
                new_child = AfterStateNode(afterstate, self.env, self, self.cumulative_reward + reward)
                self.children[action] = new_child

                return action, new_child

        # Should not reach here if is_fully_expanded is implemented correctly
        raise RuntimeError("No untried actions available")

    def is_terminal(self):
        """Check if this state is terminal (game over)"""
        self.env.board = self.state.copy()
        return self.env.is_game_over()

class AfterStateNode(MCTSNode):
    """
    Represents an afterstate node (chance node) in MCTS.
    This is the state after the agent's action but before the environment's random tile placement.
    """
    def __init__(self, state, env, parent=None, cumulative_reward=0):
        super().__init__(parent)
        self.state = state.copy() if state is not None else None  # The afterstate (4x4 numpy array)
        self.env = env  # Store reference to the environment
        self.cumulative_reward = cumulative_reward  # Accumulated reward from root to this node
        self.empty_cells = None  # Will be set during expansion

    def is_fully_expanded(self):
        """Check if all possible tile placements have been tried"""
        if self.empty_cells is None:
            if self.state is None:
                self.empty_cells = []
            else:
                self.empty_cells = list(zip(*np.where(self.state == 0)))

        # For each empty cell, we can place a 2 or a 4
        return len(self.children) == len(self.empty_cells) * 2

    def select_child(self, exploration_weight=None):
        """
        Select a child randomly based on the environment dynamics.
        For afterstates, we don't use UCB1 but instead sample according to probabilities.

        Args:
            exploration_weight: Not used for chance nodes, kept for API consistency

        Returns:
            (tile_info, child_node) tuple where tile_info is (position, value)
            or None if no children are available
        """
        # Ignore exploration_weight as it's not used for chance nodes
        # If not all children are expanded, expand a new one
        if not self.is_fully_expanded():
            return self.expand()

        # Ensure empty_cells is populated
        if self.empty_cells is None:
            if self.state is None:
                self.empty_cells = []
            else:
                self.empty_cells = list(zip(*np.where(self.state == 0)))

        # Check for zero empty cells (edge case: board is full after player's move)
        if not self.empty_cells:
            # This is an edge case: board is full after player's move
            # The game likely ends here
            print("Warning: Trying to select child from AfterStateNode with no empty cells.")
            return None  # Indicate no valid child selection possible

        # Calculate probabilities for each child
        # 90% chance for a 2, 10% chance for a 4, uniform distribution over empty cells
        child_items = list(self.children.items())
        probabilities = []
        num_empty = len(self.empty_cells)  # Use a variable for clarity

        for (pos, value), _ in child_items:
            # Probability calculation safe now due to the check above
            prob = (0.9 / num_empty) if value == 2 else (0.1 / num_empty)
            probabilities.append(prob)

        # Normalize probabilities to ensure they sum to 1 (handle floating point issues)
        probabilities = np.array(probabilities)
        sum_prob = np.sum(probabilities)
        if sum_prob > 0:
            probabilities = probabilities / sum_prob

        # Sample a child according to probabilities
        try:
            idx = np.random.choice(len(child_items), p=probabilities)
            return child_items[idx]
        except ValueError as e:
            # This might happen if probabilities don't sum to 1 exactly due to float issues
            print(f"Error during np.random.choice: {e}. Probabilities: {probabilities}")
            # Fallback: choose uniformly
            idx = np.random.choice(len(child_items))
            return child_items[idx]

    def expand(self):
        """
        Expand by adding a new state node for an untried tile placement.

        Returns:
            (tile_info, new_child_node) tuple where tile_info is (position, value)
            or None if no expansion is possible
        """
        # If empty cells not yet determined, compute them
        if self.empty_cells is None:
            if self.state is None:
                self.empty_cells = []
            else:
                # Find all empty cells
                self.empty_cells = list(zip(*np.where(self.state == 0)))

        # Check if there are any empty cells
        if not self.empty_cells:
            return None  # No expansion possible

        # Try placing a 2 or 4 in each empty cell
        for pos in self.empty_cells:
            for value in [2, 4]:
                tile_info = (pos, value)

                if tile_info not in self.children:
                    # Create a new state by placing the tile
                    new_state = self.state.copy()
                    new_state[pos] = value

                    # Create a new state node
                    new_child = StateNode(new_state, self.env, self, self.cumulative_reward)
                    self.children[tile_info] = new_child

                    return tile_info, new_child

        # If we've tried all possible tile placements and none are available
        if len(self.children) == len(self.empty_cells) * 2:
            return None  # Fully expanded, no more expansions possible

        # Should not reach here if is_fully_expanded is implemented correctly
        raise RuntimeError("No untried tile placements available but empty cells exist")

    def is_terminal(self):
        """Afterstates are never terminal as we still need to place a random tile"""
        return False
