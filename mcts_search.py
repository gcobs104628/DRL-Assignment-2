import numpy as np
import time
from mcts_node import StateNode, ACTIONS, ACTION_UP

class MCTS:
    """Monte Carlo Tree Search for 2048 game"""
    def __init__(self, n_tuple_network, env, iterations=50, exploration_weight=0.1, v_norm=400000):
        """
        Initialize the MCTS algorithm.

        Args:
            n_tuple_network: Trained N-Tuple Network for state evaluation
            env: Game environment
            iterations: Number of MCTS iterations to run
            exploration_weight: Exploration constant for UCB1 formula
            v_norm: Normalization constant for value estimates
        """
        self.network = n_tuple_network
        self.env = env
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.v_norm = v_norm

    def search(self, state, time_limit=None):
        """
        Run MCTS search from the given state.

        Args:
            state: The current game state
            time_limit: Optional time limit in seconds

        Returns:
            Best action according to MCTS
        """
        # Create root node with environment reference
        root = StateNode(state, self.env)

        # Run iterations
        start_time = time.time()
        iteration = 0

        while iteration < self.iterations:
            # Check time limit if specified
            if time_limit and time.time() - start_time > time_limit:
                break

            # Run one iteration of MCTS
            self._execute_round(root)
            iteration += 1

        # Choose the action with the highest visit count
        if not root.children:
            # If no children (should not happen if legal moves exist), return a random action
            self.env.board = state.copy()
            legal_actions = []
            for action in ACTIONS:  # Use action constants
                if self.env.is_move_legal(action):
                    legal_actions.append(action)
            return np.random.choice(legal_actions) if legal_actions else ACTION_UP  # Default to UP if no legal moves

        return self._select_best_action(root)

    def _execute_round(self, root):
        """
        Execute one round of MCTS: selection, expansion, evaluation, backpropagation.

        Args:
            root: Root node of the search tree
        """
        # 1. Selection: Traverse the tree to find a node to expand
        node = root
        path = [node]

        # Traverse until we reach a node that is not fully expanded or is terminal
        while not node.is_terminal() and node.is_fully_expanded():
            # Select child according to selection policy
            if isinstance(node, StateNode):
                result = node.select_child(self.exploration_weight)
                if result is None:
                    # No valid children to select, break the loop
                    break
                _, node = result  # Unpack but ignore action/tile_info
            else:  # AfterStateNode
                result = node.select_child()
                if result is None:
                    # No valid children to select, break the loop
                    break
                _, node = result  # Unpack but ignore action/tile_info

            path.append(node)

        # 2. Expansion: If the node is not terminal, expand it
        if not node.is_terminal() and not node.is_fully_expanded():
            if isinstance(node, StateNode):
                result = node.expand()
                if result is not None:
                    _, node = result  # Unpack but ignore action/tile_info
                    path.append(node)  # Only append if expansion was successful
            else:  # AfterStateNode
                result = node.expand()
                if result is not None:
                    _, node = result  # Unpack but ignore action/tile_info
                    path.append(node)  # Only append if expansion was successful

        # 3. Evaluation: Use the N-Tuple Network to evaluate the leaf node
        value = self._evaluate(node)

        # 4. Backpropagation: Update statistics for all nodes in the path
        for node in reversed(path):
            node.update(value)

    def _evaluate(self, node):
        """
        Evaluate a node using the N-Tuple Network.

        Args:
            node: Node to evaluate (typically the leaf node found)

        Returns:
            Normalized value estimate of the node's state
        """
        # If the node represents a terminal state, its value is 0 future reward
        if node.is_terminal():
            return 0.0

        # Handle potential None state (e.g., failed expansion)
        if node.state is None:
            return 0.0

        # Evaluate the state directly using the N-Tuple Network
        raw_value = self.network.evaluate(node.state)

        # Normalize the value estimate from the network
        # The network estimates FUTURE value from this state
        normalized_value = raw_value / self.v_norm

        # Optionally clamp the value to a reasonable range if needed
        # normalized_value = max(0.0, min(1.0, normalized_value))

        return normalized_value

    def _select_best_action(self, root):
        """
        Select the best action based on visit counts.

        Args:
            root: Root node of the search tree

        Returns:
            Best action
        """
        # Choose the action with the highest visit count
        visit_counts = {action: child.visit_count for action, child in root.children.items()}

        # If all visit counts are 0, choose randomly among legal actions
        if all(count == 0 for count in visit_counts.values()):
            # Use pre-computed legal actions if available, otherwise compute
            if hasattr(root, 'available_actions') and root.available_actions:
                # If actions are available but no visits, choose randomly from them
                return np.random.choice(root.available_actions)
            else:
                # Fallback to re-computing if root never determined actions
                self.env.board = root.state.copy()  # Ensure env state is correct
                legal_actions = [action for action in ACTIONS if self.env.is_move_legal(action)]
                return np.random.choice(legal_actions) if legal_actions else ACTION_UP  # Default to UP if no legal moves

        return max(visit_counts.items(), key=lambda x: x[1])[0]
