import numpy as np
from game_env import Game2048Env

class NStepTDLearner:
    """
    N-Step Temporal Difference Learning for 2048 game using N-Tuple Network.
    This class implements n-step TD learning with a focus on afterstate evaluation.
    """
    def __init__(self, n_tuple_network, n_step=5):
        """
        Initialize the N-Step TD Learner.

        Args:
            n_tuple_network: The N-Tuple Network for state evaluation
            n_step: Number of steps to look ahead for n-step TD (default: 5)
        """
        self.network = n_tuple_network
        self.n_step = n_step  # n-step for TD(n)
        self.env = Game2048Env()

    def compute_td_target(self, rewards, next_state_value, gamma=1.0):
        """
        Compute the n-step TD target using the n-step return formula.

        Args:
            rewards: List of rewards for each step [r_1, r_2, ..., r_n]
            next_state_value: Value of the state after n steps
            gamma: Discount factor (usually 1.0 for 2048)

        Returns:
            The n-step TD target value
        """
        n = len(rewards)

        if n == 0:
            return next_state_value

        # Calculate n-step return
        n_step_return = 0
        for i in range(n):
            n_step_return += (gamma ** i) * rewards[i]

        # Add the discounted value of the final state
        n_step_return += (gamma ** n) * next_state_value

        return n_step_return

    def compute_td_error(self, state, td_target):
        """
        Compute the TD error between the current state value and the TD target.

        Args:
            state: The current state
            td_target: The TD target value

        Returns:
            The TD error
        """
        current_value = self.network.evaluate(state)
        return td_target - current_value

    def collect_episode_data(self, max_steps=1000):
        """
        Collect data from a complete episode.

        Args:
            max_steps: Maximum number of steps to run

        Returns:
            A list of (state, action, reward, next_state, afterstate, done) tuples
        """
        state = self.env.reset()
        done = False
        episode_data = []
        step_count = 0

        while not done and step_count < max_steps:
            # Choose action (greedy policy based on current network)
            action = self.choose_action(state)

            # Get afterstate
            afterstate, _ = self.env.get_afterstate(state, action)  # Ignore afterstate_reward

            # Take action
            next_state, reward, done, _ = self.env.step(action)  # Ignore info

            # Store data including the done flag
            episode_data.append((state, action, reward, next_state, afterstate, done))

            # Update state
            state = next_state
            step_count += 1

        return episode_data

    def choose_action(self, state):
        """
        Choose an action using a greedy policy based on the current network.

        Args:
            state: The current state

        Returns:
            The chosen action or None if no valid actions are available
        """
        best_action = None
        best_value = float('-inf')

        # Save current environment state to restore later
        current_board_state = self.env.board.copy()

        # Set environment state to the input state for action evaluation
        self.env.board = state.copy()

        for action in range(4):
            afterstate, _ = self.env.get_afterstate(state, action)

            if afterstate is None:  # Invalid move
                continue

            value = self.network.evaluate(afterstate)

            if value > best_value:
                best_value = value
                best_action = action

        # If no valid action found, check if any actions are legal
        if best_action is None:
            valid_actions = []
            for action in range(4):
                if self.env.is_move_legal(action):
                    valid_actions.append(action)

            if valid_actions:
                best_action = np.random.choice(valid_actions)
            else:
                # No valid actions means terminal state
                # Return None to indicate no action is possible
                best_action = None

        # Restore original environment state
        self.env.board = current_board_state

        return best_action

    def train_step(self, episode_data, alpha, gamma=1.0):
        """
        Perform a training step using n-step TD learning on the collected episode data.

        Args:
            episode_data: List of (state, action, reward, next_state, afterstate, done) tuples
            alpha: Learning rate
            gamma: Discount factor (default: 1.0 for 2048)

        Returns:
            Average TD error
        """
        total_td_error = 0.0
        n = len(episode_data)

        for i in range(n):
            # Get current afterstate
            _, _, _, _, afterstate, _ = episode_data[i]

            if afterstate is None:
                continue

            # Collect rewards and track if we hit a terminal state
            rewards = []
            final_step_index = i  # Track the index of the last step used for the return
            reached_terminal = False

            # Collect rewards for up to n steps or until terminal state
            for j in range(i, min(i + self.n_step, n)):
                rewards.append(episode_data[j][2])  # Reward
                final_step_index = j

                # Check if this transition led to a terminal state
                if j + 1 < n and episode_data[j][5]:  # done flag
                    reached_terminal = True
                    break

            # Compute the value of the final afterstate (0 if terminal)
            next_state_value = 0.0
            if not reached_terminal and final_step_index + 1 < n:
                next_afterstate = episode_data[final_step_index + 1][4]  # Next afterstate
                if next_afterstate is not None:
                    next_state_value = self.network.evaluate(next_afterstate)

            # Compute TD target using the collected rewards and the next_state_value
            td_target = self.compute_td_target(rewards, next_state_value, gamma)

            # Compute TD error for the current afterstate
            td_error = self.compute_td_error(afterstate, td_target)
            total_td_error += abs(td_error)

            # Update weights for the current afterstate
            self.network.update(afterstate, td_error, alpha)

        return total_td_error / n if n > 0 else 0.0
