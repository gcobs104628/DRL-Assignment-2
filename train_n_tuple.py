import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from game_env import Game2048Env
from n_tuple_network import NTupleNetwork

class OTDTrainer:
    """Optimistic Temporal Difference Trainer for N-Tuple Network"""
    def __init__(self, v_init=320000, alpha=0.0025, min_alpha=0.00025, gamma=1.0):
        """
        Initialize the OTD Trainer.
        
        Args:
            v_init: Initial value for optimistic initialization
            alpha: Learning rate
            min_alpha: Minimum learning rate after decay
            gamma: Discount factor
        """
        self.v_init = v_init
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.gamma = gamma
        self.env = Game2048Env()
        self.network = NTupleNetwork(v_init=v_init)
        
        # Create weights directory if it doesn't exist
        os.makedirs("DRL/DRL_HW2/DRL-Assignment-2/weights", exist_ok=True)
    
    def train(self, num_episodes=100000, eval_interval=1000, save_interval=10000):
        """
        Train the N-Tuple Network using OTD.
        
        Args:
            num_episodes: Number of episodes to train
            eval_interval: Interval for evaluation
            save_interval: Interval for saving weights
            
        Returns:
            Trained network
        """
        print(f"Starting training with {num_episodes} episodes...")
        
        # Statistics
        scores = []
        avg_scores = []
        max_scores = []
        td_errors = []
        
        # Training loop
        for episode in tqdm(range(num_episodes)):
            # Decay learning rate
            if episode > num_episodes / 2:
                self.alpha = max(self.min_alpha, self.alpha * 0.9999)
            
            # Run episode
            episode_score, avg_td_error = self._run_episode()
            
            # Record statistics
            scores.append(episode_score)
            td_errors.append(avg_td_error)
            
            # Evaluate and print statistics
            if (episode + 1) % eval_interval == 0:
                avg_score = np.mean(scores[-eval_interval:])
                max_score = np.max(scores[-eval_interval:])
                avg_td_error = np.mean(td_errors[-eval_interval:])
                
                avg_scores.append(avg_score)
                max_scores.append(max_score)
                
                print(f"Episode {episode+1}/{num_episodes}, Avg Score: {avg_score:.2f}, "
                      f"Max Score: {max_score}, Avg TD Error: {avg_td_error:.6f}, Alpha: {self.alpha:.6f}")
            
            # Save weights
            if (episode + 1) % save_interval == 0:
                self.network.save_weights(f"DRL/DRL_HW2/DRL-Assignment-2/weights/checkpoint_{episode+1}.pkl")
        
        # Save final weights
        self.network.save_weights("DRL/DRL_HW2/DRL-Assignment-2/weights/trained_weights.pkl")
        
        # Plot learning curves
        self._plot_learning_curves(avg_scores, max_scores, td_errors[-len(avg_scores):])
        
        return self.network
    
    def _run_episode(self):
        """
        Run a single episode of training.
        
        Returns:
            Tuple of (episode_score, average_td_error)
        """
        state = self.env.reset()
        done = False
        total_reward = 0
        td_errors = []
        
        while not done:
            # Choose action (greedy policy)
            action = self._choose_action(state)
            
            # Get afterstate
            afterstate, afterstate_reward = self.env.get_afterstate(state, action)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Calculate TD error
            if done:
                # Terminal state
                td_target = reward
            else:
                # Non-terminal state
                next_afterstate, next_afterstate_reward = self._get_best_afterstate(next_state)
                if next_afterstate is None:
                    # No legal moves from next state
                    td_target = reward
                else:
                    # TD(0) target
                    td_target = reward + self.gamma * (next_afterstate_reward + self.network.evaluate(next_afterstate))
            
            # Current value
            current_value = self.network.evaluate(afterstate)
            
            # TD error
            td_error = td_target - current_value
            td_errors.append(abs(td_error))
            
            # Update weights
            self.network.update(afterstate, td_error, self.alpha)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
        
        return total_reward, np.mean(td_errors) if td_errors else 0.0
    
    def _choose_action(self, state):
        """
        Choose an action using a greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Chosen action
        """
        best_action = None
        best_value = float('-inf')
        
        for action in range(4):
            afterstate, reward = self.env.get_afterstate(state, action)
            
            if afterstate is None:  # Invalid move
                continue
            
            value = reward + self.network.evaluate(afterstate)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        # If no valid action found, choose randomly
        if best_action is None:
            valid_actions = []
            for action in range(4):
                if self.env.is_move_legal(action):
                    valid_actions.append(action)
            
            if valid_actions:
                best_action = np.random.choice(valid_actions)
            else:
                best_action = np.random.randint(0, 4)
        
        return best_action
    
    def _get_best_afterstate(self, state):
        """
        Get the best afterstate from the current state.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (best_afterstate, reward)
        """
        best_afterstate = None
        best_value = float('-inf')
        best_reward = 0
        
        for action in range(4):
            afterstate, reward = self.env.get_afterstate(state, action)
            
            if afterstate is None:  # Invalid move
                continue
            
            value = reward + self.network.evaluate(afterstate)
            
            if value > best_value:
                best_value = value
                best_afterstate = afterstate
                best_reward = reward
        
        return best_afterstate, best_reward
    
    def _plot_learning_curves(self, avg_scores, max_scores, td_errors):
        """Plot learning curves"""
        plt.figure(figsize=(15, 5))
        
        # Plot average scores
        plt.subplot(1, 3, 1)
        plt.plot(avg_scores)
        plt.title('Average Scores')
        plt.xlabel('Evaluation (x1000 episodes)')
        plt.ylabel('Average Score')
        
        # Plot max scores
        plt.subplot(1, 3, 2)
        plt.plot(max_scores)
        plt.title('Max Scores')
        plt.xlabel('Evaluation (x1000 episodes)')
        plt.ylabel('Max Score')
        
        # Plot TD errors
        plt.subplot(1, 3, 3)
        plt.plot(td_errors)
        plt.title('TD Errors')
        plt.xlabel('Evaluation (x1000 episodes)')
        plt.ylabel('Average TD Error')
        
        plt.tight_layout()
        plt.savefig("DRL/DRL_HW2/DRL-Assignment-2/weights/learning_curves.png")
        plt.close()

if __name__ == "__main__":
    # For testing with fewer episodes
    test_episodes = 1000  # Set to a small number for testing
    
    # Create trainer
    trainer = OTDTrainer(v_init=320000, alpha=0.0025, min_alpha=0.00025)
    
    # Train network
    network = trainer.train(num_episodes=test_episodes, eval_interval=100, save_interval=500)
    
    print("Training complete. Weights saved to weights/trained_weights.pkl")
