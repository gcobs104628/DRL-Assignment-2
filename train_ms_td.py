import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from game_env import Game2048Env
from n_tuple_network import NTupleNetwork
from td_learner import NStepTDLearner

class MSTDTrainer:
    def __init__(self):
        self.env = Game2048Env()
        # 使用相對於當前文件的路徑
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = os.path.join(current_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

        # Stage conditions
        self.stage_conditions = [
            lambda s: self._has_tile(s, 1024),   # Stage 1 -> 2: First 1024 tile
            lambda s: self._has_tile(s, 2048),   # Stage 2 -> 3: First 2048 tile
            lambda s: self._has_tile(s, 4096),   # Stage 3 -> 4: First 4096 tile
            lambda s: self._has_tile(s, 8192),   # Stage 4 -> 5: First 8192 tile
            # Stage 5 -> ?: Maybe target 16384 or a combination
            lambda s: self._has_tile(s, 16384),
        ]

    def _has_tile(self, state, value):
        """Check if the state has a tile with the specified value"""
        return np.any(state == value)

    def _demonstrate_gameplay(self, network, max_steps=float('inf')):
        """Demonstrate the gameplay using the current network"""
        env = Game2048Env()
        state = env.reset()
        done = False
        step = 0
        total_score = 0

        # 顯示初始狀態
        print("\nInitial state:")
        self._print_board(env.board)

        while not done and step < max_steps:
            # 使用網絡選擇動作
            best_action = None
            best_value = float('-inf')

            for action in range(4):
                afterstate, _ = env.get_afterstate(state, action)

                if afterstate is None:  # 無效動作
                    continue

                value = network.evaluate(afterstate)

                if value > best_value:
                    best_value = value
                    best_action = action

            # 如果沒有有效動作，離開循環
            if best_action is None:
                print("No valid actions available. Game over.")
                break

            # 執行動作
            next_state, reward, done, _ = env.step(best_action)

            # 顯示動作和結果
            action_names = ["Up", "Down", "Left", "Right"]
            print(f"\nStep {step+1}: Action = {action_names[best_action]}, Reward = {reward}")
            self._print_board(env.board)

            state = next_state
            total_score += reward
            step += 1

        print(f"\nDemo finished. Total score: {total_score}, Max tile: {np.max(env.board)}")

    def _print_board(self, board):
        """Print the game board in a readable format"""
        # 創建一個格式化的板子字符串
        board_str = ""
        for row in board:
            board_str += "+------+------+------+------+\n|"
            for cell in row:
                if cell == 0:
                    board_str += "      |"
                else:
                    board_str += f"{cell:^6}|"
            board_str += "\n"
        board_str += "+------+------+------+------+"
        print(board_str)

    def train_stage(self, stage, network=None, start_data=None, num_episodes=5000000, initial_alpha=0.1,
                    alpha_decay_episodes=1000000, min_alpha=0.00025, eval_interval=10000, n_step=5, gamma=1.0):
        # 列印參數信息以進行調試
        print(f"\nIn train_stage method: stage={stage}, num_episodes={num_episodes}, eval_interval={eval_interval}")
        """
        Train a network for a specific stage.

        Args:
            stage: Stage number (1-5)
            network: Pre-trained network to continue training (None to create a new one)
            start_data: List of (state, score) tuples to start from (None for stage 1)
            num_episodes: Number of episodes to train
            initial_alpha: Initial learning rate
            alpha_decay_episodes: Number of episodes after which to reach min_alpha
            min_alpha: Minimum learning rate
            eval_interval: Interval for evaluation
            n_step: Number of steps for n-step TD learning
            gamma: Discount factor (usually 1.0 for 2048)

        Returns:
            Trained network and collected data for the next stage
        """
        print(f"Training Stage {stage}...")

        # Initialize network (create new if none provided)
        if network is None:
            network = NTupleNetwork()

        # Initialize TD learner with the network
        td_learner = NStepTDLearner(network, n_step=n_step)

        # Initialize learning rate
        alpha = initial_alpha

        # Initialize statistics
        scores = []
        avg_scores = []
        max_scores = []
        td_errors = []
        avg_td_errors_plot = []  # For plotting (one per eval_interval)
        td_errors_current_interval = []  # Temporary storage for current interval

        # Training loop
        for episode in tqdm(range(num_episodes)):
            # Linear alpha decay from initial_alpha to min_alpha over alpha_decay_episodes
            if episode < alpha_decay_episodes:
                decay_fraction = episode / alpha_decay_episodes
                alpha = initial_alpha - (initial_alpha - min_alpha) * decay_fraction
            else:
                # After decay period, keep alpha at min_alpha
                alpha = min_alpha

            # Run episode
            if start_data is None:
                # For stage 1, start from the beginning
                episode_data = td_learner.collect_episode_data()
                episode_score = td_learner.env.score
                steps = len(episode_data)
                max_tile = np.max(td_learner.env.board)
            else:
                # For later stages, start from collected data
                if not start_data:  # 如果 start_data 為空
                    # 嘗試載入更前一個階段的數據
                    prev_stage = stage - 1
                    while prev_stage > 0 and not start_data:
                        prev_stage_data_file = f"{self.weights_dir}/stage{prev_stage}_start_data.pkl"
                        if os.path.exists(prev_stage_data_file):
                            with open(prev_stage_data_file, 'rb') as f:
                                start_data = pickle.load(f)
                            print(f"Using data from stage {prev_stage-1} instead. Loaded {len(start_data)} data points.")
                        else:
                            prev_stage -= 1

                    if not start_data:  # 如果仍然沒有數據
                        print("Warning: No data available from any previous stage. Starting from a fresh game.")
                        episode_data = td_learner.collect_episode_data()
                        episode_score = td_learner.env.score
                        steps = len(episode_data)
                        max_tile = np.max(td_learner.env.board)
                        continue  # 繼續下一個迭代

                start_idx = random.randint(0, len(start_data) - 1)
                start_state, start_score = start_data[start_idx]

                # Set environment state
                td_learner.env.board = start_state.copy()
                td_learner.env.score = start_score

                # Collect episode data from this starting point
                episode_data = []
                state = start_state.copy()
                done = False
                step_count = 0

                # Run until done (no step limit)
                while not done:
                    action = td_learner.choose_action(state)
                    if action is None:  # No valid actions available
                        done = True
                        break

                    # Get afterstate before the step
                    afterstate, _ = td_learner.env.get_afterstate(state, action)

                    # Take action
                    next_state, reward, done, _ = td_learner.env.step(action)

                    # Store data with the done flag
                    episode_data.append((state, action, reward, next_state, afterstate, done))

                    state = next_state
                    step_count += 1

                episode_score = td_learner.env.score - start_score
                steps = step_count
                max_tile = np.max(td_learner.env.board)
            # Train on episode data
            avg_td_error = td_learner.train_step(episode_data, alpha, gamma)

            # Record statistics
            scores.append(episode_score)
            td_errors.append(avg_td_error)
            td_errors_current_interval.append(avg_td_error)

            # Print progress for every episode
            print(f"Episode {episode+1}/{num_episodes}, Steps: {steps}, Score: {episode_score}, Max Tile: {max_tile} TD Error: {avg_td_error:.6f} Alpha: {alpha:.6f}")

            # Demonstrate gameplay every eval_interval episodes
            if (episode + 1) % eval_interval == 0:
                # 每 eval_interval 個 episode 顯示一次訓練過程
                print("\n=== Demonstrating training process for episode", episode+1, "===")
                self._demonstrate_gameplay(td_learner.network)

            # Evaluate and print statistics
            if (episode + 1) % eval_interval == 0:
                avg_score = np.mean(scores[-eval_interval:])
                max_score = np.max(scores[-eval_interval:])
                avg_td_error_interval = np.mean(td_errors_current_interval)

                avg_scores.append(avg_score)
                max_scores.append(max_score)
                avg_td_errors_plot.append(avg_td_error_interval)

                print(f"Episode {episode+1}/{num_episodes}, Avg Score: {avg_score:.2f}, "
                      f"Max Score: {max_score}, Avg TD Error: {avg_td_error_interval:.6f}, Alpha: {alpha:.6f}")

                # Clear the temp list for the next interval
                td_errors_current_interval = []

                # Save checkpoint
                network.save_weights(f"{self.weights_dir}/stage{stage}_checkpoint.pkl")



        # Save final weights
        network.save_weights(f"{self.weights_dir}/stage{stage}_weights.pkl")

        # Plot learning curves
        self._plot_learning_curves(avg_scores, max_scores, avg_td_errors_plot, stage, eval_interval)

        # Collect data for the next stage if not the last stage
        next_stage_data = None
        if stage < 5:
            next_stage_data = self._collect_next_stage_data(network, stage)

            # Save collected data
            with open(f"{self.weights_dir}/stage{stage+1}_start_data.pkl", 'wb') as f:
                pickle.dump(next_stage_data, f)

        return network, next_stage_data

    def _collect_next_stage_data(self, network, current_stage, num_episodes=1000):  # 減少默認 episode 數量
        """
        Collect data for the next stage.

        Args:
            network: Trained network for the current stage
            current_stage: Current stage number
            num_episodes: Number of episodes to run

        Returns:
            List of (state, score) tuples that satisfy the next stage condition
        """
        print(f"Collecting data for Stage {current_stage+1}...")

        next_stage_data = []
        td_learner = NStepTDLearner(network)

        for episode in tqdm(range(num_episodes)):
            # Reset environment
            state = td_learner.env.reset()
            done = False

            # Keep track of states and scores
            states_history = []
            scores_history = []

            while not done:
                action = td_learner.choose_action(state)
                if action is None:  # No valid actions available
                    break

                next_state, _, done, _ = td_learner.env.step(action)

                # Store state and score
                states_history.append(state.copy())
                scores_history.append(td_learner.env.score)

                # Check if the next stage condition is met
                if self.stage_conditions[current_stage](next_state):
                    # Find the state just before the condition was met
                    if len(states_history) > 0:
                        next_stage_data.append((states_history[-1], scores_history[-1]))
                        break

                state = next_state

        # 如果沒有找到符合條件的狀態，使用最後一個 episode 的最後狀態
        if len(next_stage_data) == 0 and len(states_history) > 0:
            print(f"Warning: No states found meeting condition for Stage {current_stage+1}. Using last game state.")
            next_stage_data.append((states_history[-1], scores_history[-1]))

        print(f"Collected {len(next_stage_data)} data points for Stage {current_stage+1}")
        return next_stage_data

    def _plot_learning_curves(self, avg_scores, max_scores, avg_td_errors_plot, stage, eval_interval):
        """Plot learning curves for the stage"""
        plt.figure(figsize=(15, 5))

        # Plot average scores
        plt.subplot(1, 3, 1)
        plt.plot(avg_scores)
        plt.title(f'Stage {stage} - Average Scores')
        plt.xlabel(f'Evaluation (x{eval_interval} episodes)')
        plt.ylabel('Average Score')

        # Plot max scores
        plt.subplot(1, 3, 2)
        plt.plot(max_scores)
        plt.title(f'Stage {stage} - Max Scores')
        plt.xlabel(f'Evaluation (x{eval_interval} episodes)')
        plt.ylabel('Max Score')

        # Plot TD errors
        plt.subplot(1, 3, 3)
        plt.plot(avg_td_errors_plot)
        plt.title(f'Stage {stage} - Avg TD Errors (Eval Interval)')
        plt.xlabel(f'Evaluation (x{eval_interval} episodes)')
        plt.ylabel('Average TD Error')

        plt.tight_layout()
        plt.savefig(f"{self.weights_dir}/stage{stage}_learning_curves.png")
        plt.close()

    def train_all_stages(self, num_episodes_per_stage=5000000, start_stage=1, eval_interval=10000):
        """Train all 5 stages"""
        print(f"\nIn train_all_stages: num_episodes_per_stage={num_episodes_per_stage}, start_stage={start_stage}, eval_interval={eval_interval}")
        start_time = time.time()
        trained_network = None  # Start with no network
        next_stage_data = None

        # 如果不是從第一階段開始，嘗試載入前一階段的模型和數據
        if start_stage > 1:
            # 嘗試載入前一階段的模型
            prev_weights_file = f"{self.weights_dir}/stage{start_stage-1}_weights.pkl"
            if os.path.exists(prev_weights_file):
                trained_network = NTupleNetwork()
                trained_network.load_weights(prev_weights_file)
                print(f"Loaded weights from stage {start_stage-1}")
            else:
                print(f"No weights found for stage {start_stage-1}, starting with a new network")
                trained_network = None

            # 嘗試載入前一階段的數據
            prev_data_file = f"{self.weights_dir}/stage{start_stage}_start_data.pkl"
            if os.path.exists(prev_data_file):
                with open(prev_data_file, 'rb') as f:
                    next_stage_data = pickle.load(f)
                print(f"Loaded {len(next_stage_data)} data points for stage {start_stage}")
            else:
                print(f"No data found for stage {start_stage}, will collect new data")
                next_stage_data = None

        # 訓練各階段
        stage_data = {1: None, 2: None, 3: None, 4: None, 5: None}

        # 第一階段
        if start_stage <= 1:
            print("\n=== Training Stage 1 ===")
            trained_network, stage_data[2] = self.train_stage(1, network=trained_network, num_episodes=num_episodes_per_stage, eval_interval=eval_interval)
        else:
            stage_data[2] = next_stage_data

        # 第二階段
        if start_stage <= 2:
            print("\n=== Training Stage 2 ===")
            trained_network, stage_data[3] = self.train_stage(2, network=trained_network, start_data=stage_data[2], num_episodes=num_episodes_per_stage, eval_interval=eval_interval)
        else:
            stage_data[3] = next_stage_data

        # 第三階段
        if start_stage <= 3:
            print("\n=== Training Stage 3 ===")
            trained_network, stage_data[4] = self.train_stage(3, network=trained_network, start_data=stage_data[3], num_episodes=num_episodes_per_stage, eval_interval=eval_interval)
        else:
            stage_data[4] = next_stage_data

        # 第四階段
        if start_stage <= 4:
            print("\n=== Training Stage 4 ===")
            trained_network, stage_data[5] = self.train_stage(4, network=trained_network, start_data=stage_data[4], num_episodes=num_episodes_per_stage, eval_interval=eval_interval)
        else:
            stage_data[5] = next_stage_data

        # 第五階段
        if start_stage <= 5:
            print("\n=== Training Stage 5 ===")
            trained_network, _ = self.train_stage(5, network=trained_network, start_data=stage_data[5], num_episodes=num_episodes_per_stage, eval_interval=eval_interval)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

if __name__ == "__main__":
    import argparse


    # 創建命令行參數解析器
    parser = argparse.ArgumentParser(description='Train 2048 agent with MS-TD learning')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes per stage (default: 1000, use 5000000 for full training)')
    parser.add_argument('--stage', type=int, default=0,
                        help='Train specific stage only (1-5), 0 for all stages (default: 0)')
    parser.add_argument('--start-stage', type=int, default=1,
                        help='Start training from this stage (1-5) when training all stages (default: 1)')
    parser.add_argument('--collect-episodes', type=int, default=1000,
                        help='Number of episodes for collecting next stage data (default: 1000)')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Evaluation interval (default: 100)')

    args = parser.parse_args()

    # 列印命令行參數以進行調試
    print(f"\nCommand line arguments: episodes={args.episodes}, stage={args.stage}, eval_interval={args.eval_interval}")

    trainer = MSTDTrainer()

    # 創建一個包裝函數來設置數據收集的 episode 數量
    original_collect_data = trainer._collect_next_stage_data

    def wrapped_collect_data(network, current_stage):
        return original_collect_data(network, current_stage, num_episodes=args.collect_episodes)

    trainer._collect_next_stage_data = wrapped_collect_data

    if args.stage == 0:
        # 訓練所有階段
        print(f"Training all stages starting from stage {args.start_stage} with {args.episodes} episodes per stage, eval_interval={args.eval_interval}")
        trainer.train_all_stages(num_episodes_per_stage=args.episodes, start_stage=args.start_stage, eval_interval=args.eval_interval)
    else:
        # 訓練特定階段
        stage = args.stage
        print(f"Training stage {stage} with {args.episodes} episodes")

        # 載入前一階段的數據（如果有）
        start_data = None
        if stage > 1:
            prev_stage_data_file = f"{trainer.weights_dir}/stage{stage}_start_data.pkl"
            if os.path.exists(prev_stage_data_file):
                with open(prev_stage_data_file, 'rb') as f:
                    start_data = pickle.load(f)
                print(f"Loaded {len(start_data)} data points from stage {stage-1}")
            else:
                print(f"No data found for stage {stage-1}, starting from scratch")

        # 載入前一階段的網絡（如果有）
        network = None
        if stage > 1:
            prev_weights_file = f"{trainer.weights_dir}/stage{stage-1}_weights.pkl"
            if os.path.exists(prev_weights_file):
                network = NTupleNetwork()
                network.load_weights(prev_weights_file)
                print(f"Loaded weights from stage {stage-1}")

        # 訓練當前階段
        print("start training")
        trained_network, next_stage_data = trainer.train_stage(
            stage=stage,
            network=network,
            start_data=start_data,
            num_episodes=args.episodes,
            eval_interval=args.eval_interval
        )

        # 保存訓練好的權重到 student_agent.py 使用的位置
        weights_path = os.path.join(trainer.weights_dir, "trained_weights.pkl")
        trained_network.save_weights(weights_path)
        print(f"Saved weights to {weights_path}")
