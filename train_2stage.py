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

class TwoStageTrainer:
    def __init__(self):
        self.env = Game2048Env()
        # 使用相對於當前文件的路徑
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_dir = os.path.join(current_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

        # 只有2個階段，條件是首次達到1024方塊
        self.stages = 2
        self.stage_conditions = [
            lambda s: self._has_tile(s, 1024),   # Stage 1 -> 2: First 1024 tile
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

        print(f"\nGame over. Total score: {total_score}")
        return total_score

    def _print_board(self, board):
        """Print the game board in a readable format"""
        for row in board:
            print(row)

    def train_stage(self, stage, network=None, start_data=None, num_episodes=100000, initial_alpha=0.001,
                    alpha_decay_episodes=1000000, min_alpha=0.00025, eval_interval=1000, n_step=5, gamma=1.0):
        """
        Train a network for a specific stage.

        Args:
            stage: Stage number (1-2)
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

        # 檢查是否有檢查點文件
        checkpoint_file = f"{self.weights_dir}/stage{stage}_checkpoint.pkl"
        checkpoint_exists = os.path.exists(checkpoint_file)

        # 初始化統計資料
        scores = []
        avg_scores = []
        max_scores = []
        td_errors = []
        avg_td_errors_plot = []  # For plotting (one per eval_interval)
        td_errors_current_interval = []  # Temporary storage for current interval
        start_episode = 0  # 預設從第 0 個 episode 開始

        # Initialize network (create new if none provided)
        if network is None:
            if checkpoint_exists:
                # 如果有檢查點，優先加載檢查點
                try:
                    with open(checkpoint_file, 'rb') as f:
                        checkpoint_data = pickle.load(f)

                    # 創建新網絡並載入權重
                    network = NTupleNetwork()
                    network.load_weights(checkpoint_file)

                    # 如果檢查點包含統計資料，則恢復它們
                    if isinstance(checkpoint_data, dict) and 'episode' in checkpoint_data:
                        start_episode = checkpoint_data['episode']
                        scores = checkpoint_data.get('scores', [])
                        avg_scores = checkpoint_data.get('avg_scores', [])
                        max_scores = checkpoint_data.get('max_scores', [])
                        td_errors = checkpoint_data.get('td_errors', [])
                        avg_td_errors_plot = checkpoint_data.get('avg_td_errors_plot', [])
                        print(f"Loaded checkpoint for stage {stage} to continue training from episode {start_episode}")
                    else:
                        print(f"Loaded checkpoint for stage {stage} but could not restore episode count")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}. Starting with a new network.")
                    network = NTupleNetwork()
            else:
                # 否則創建新網絡
                network = NTupleNetwork()

        # Initialize TD learner with the network
        td_learner = NStepTDLearner(network, n_step=n_step)

        # Initialize learning rate
        alpha = initial_alpha

        # Training loop - 從 start_episode 開始繼續訓練
        for episode in tqdm(range(start_episode, num_episodes)):
            # 使用固定的學習率
            alpha = initial_alpha

            # Run episode
            if start_data is None:
                # For stage 1, start from the beginning
                episode_data = td_learner.collect_episode_data()
                episode_score = td_learner.env.score
                steps = len(episode_data)
                max_tile = np.max(td_learner.env.board)
            else:
                # For stage 2, start from collected data
                if not start_data:  # 如果 start_data 為空
                    # 嘗試載入更前一個階段的數據
                    prev_stage = stage - 1
                    while prev_stage > 0 and not start_data:
                        prev_stage_data_file = f"{self.weights_dir}/stage{prev_stage}_start_data.pkl"
                        if os.path.exists(prev_stage_data_file):
                            with open(prev_stage_data_file, 'rb') as f:
                                start_data = pickle.load(f)
                            print(f"Using data from stage {prev_stage}. Loaded {len(start_data)} data points.")
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

                # Save checkpoint with episode count
                checkpoint_data = {
                    'weights': network.get_weights(),
                    'episode': episode + 1,
                    'scores': scores,
                    'avg_scores': avg_scores,
                    'max_scores': max_scores,
                    'td_errors': td_errors,
                    'avg_td_errors_plot': avg_td_errors_plot
                }
                with open(f"{self.weights_dir}/stage{stage}_checkpoint.pkl", 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"Saved checkpoint at episode {episode+1}")

        # Save final weights
        print(f"Saving final weights for stage {stage}...")
        network.save_weights(f"{self.weights_dir}/stage{stage}_weights.pkl")

        # Plot learning curves
        self._plot_learning_curves(avg_scores, max_scores, avg_td_errors_plot, stage, eval_interval)

        # Collect data for the next stage if not the last stage
        next_stage_data = None
        if stage < self.stages:
            next_stage_data = self._collect_next_stage_data(network, stage - 1)  # stage - 1 是因為 stage_conditions 的索引從 0 開始

            # Save collected data
            with open(f"{self.weights_dir}/stage{stage+1}_start_data.pkl", 'wb') as f:
                pickle.dump(next_stage_data, f)

        return network, next_stage_data

    def _collect_next_stage_data(self, network, current_stage, num_episodes=5000):
        """
        Collect data for the next stage.

        Args:
            network: Trained network for the current stage
            current_stage: Current stage index (0 for stage 1)
            num_episodes: Number of episodes to run

        Returns:
            List of (state, score) tuples that satisfy the next stage condition
        """
        print(f"Collecting data for Stage {current_stage+2}...")  # +2 因為 current_stage 從 0 開始，而 stage 從 1 開始

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
            print(f"Warning: No states found meeting condition for Stage {current_stage+2}. Using last game state.")
            next_stage_data.append((states_history[-1], scores_history[-1]))

        print(f"Collected {len(next_stage_data)} data points for Stage {current_stage+2}")
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

    def train_all_stages(self, num_episodes_per_stage=100000, start_stage=1, eval_interval=1000):
        """Train both stages with independent weights"""
        start_time = time.time()
        stage1_network = None  # Network for stage 1
        stage2_network = None  # Network for stage 2 (independent from stage 1)
        next_stage_data = None

        # 如果不是從第一階段開始，嘗試載入前一階段的模型和數據
        if start_stage > 1:
            # 嘗試載入前一階段的模型（只用於收集數據，不用於繼承權重）
            prev_weights_file = f"{self.weights_dir}/stage{start_stage-1}_weights.pkl"
            if os.path.exists(prev_weights_file):
                stage1_network = NTupleNetwork()
                stage1_network.load_weights(prev_weights_file)
                print(f"Loaded weights from stage {start_stage-1} for data collection")
            else:
                print(f"No weights found for stage {start_stage-1}, starting with a new network")
                stage1_network = None

            # 嘗試載入前一階段的數據
            prev_data_file = f"{self.weights_dir}/stage{start_stage}_start_data.pkl"
            if os.path.exists(prev_data_file):
                with open(prev_data_file, 'rb') as f:
                    next_stage_data = pickle.load(f)
                print(f"Loaded {len(next_stage_data)} data points for stage {start_stage}")
            else:
                print(f"No data found for stage {start_stage}, will collect new data")
                next_stage_data = None

            # 嘗試載入當前階段的模型（如果存在）
            current_weights_file = f"{self.weights_dir}/stage{start_stage}_weights.pkl"
            if os.path.exists(current_weights_file):
                if start_stage == 2:
                    stage2_network = NTupleNetwork()
                    stage2_network.load_weights(current_weights_file)
                    print(f"Loaded existing weights for stage {start_stage}")

        # 訓練各階段
        stage_data = {1: None, 2: None}

        # 第一階段
        if start_stage <= 1:
            print("\n=== Training Stage 1 ===")
            # 檢查是否有 Stage 1 的檢查點
            checkpoint_file = f"{self.weights_dir}/stage1_checkpoint.pkl"
            if os.path.exists(checkpoint_file) and stage1_network is None:
                stage1_network = NTupleNetwork()
                stage1_network.load_weights(checkpoint_file)
                print(f"Loaded checkpoint for stage 1 to continue training")

            stage1_network, stage_data[2] = self.train_stage(1, network=stage1_network, num_episodes=num_episodes_per_stage, eval_interval=eval_interval)
        else:
            stage_data[2] = next_stage_data

        # 第二階段 - 使用獨立的權重
        if start_stage <= 2:
            print("\n=== Training Stage 2 (with independent weights) ===")
            # 檢查是否有 Stage 2 的檢查點
            checkpoint_file = f"{self.weights_dir}/stage2_checkpoint.pkl"
            if os.path.exists(checkpoint_file) and stage2_network is None:
                stage2_network = NTupleNetwork()
                stage2_network.load_weights(checkpoint_file)
                print(f"Loaded checkpoint for stage 2 to continue training")

            # 如果沒有檢查點且沒有現有權重，則從頭開始訓練
            stage2_network, _ = self.train_stage(2, network=stage2_network, start_data=stage_data[2], num_episodes=num_episodes_per_stage, eval_interval=eval_interval)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

if __name__ == "__main__":
    import argparse

    # 創建命令行參數解析器
    parser = argparse.ArgumentParser(description='Train 2048 agent with 2-Stage TD learning')
    parser.add_argument('--episodes', type=int, default=100000,
                        help='Number of episodes per stage (default: 100000)')
    parser.add_argument('--stage', type=int, default=0,
                        help='Train specific stage only (1-2), 0 for all stages (default: 0)')
    parser.add_argument('--start-stage', type=int, default=1,
                        help='Start training from this stage (1-2) when training all stages (default: 1)')
    parser.add_argument('--collect-episodes', type=int, default=5000,
                        help='Number of episodes for collecting next stage data (default: 5000)')
    parser.add_argument('--eval-interval', type=int, default=1000,
                        help='Evaluation interval (default: 1000)')

    args = parser.parse_args()

    trainer = TwoStageTrainer()

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

        # 先檢查是否有檢查點文件
        network = None
        data_collection_network = None
        checkpoint_file = f"{trainer.weights_dir}/stage{stage}_checkpoint.pkl"
        if os.path.exists(checkpoint_file):
            network = NTupleNetwork()
            network.load_weights(checkpoint_file)
            print(f"Loaded checkpoint for stage {stage} to continue training")
        # 如果沒有檢查點且是第二階段，則載入第一階段的模型作為數據收集用
        elif stage > 1:
            prev_weights_file = f"{trainer.weights_dir}/stage{stage-1}_weights.pkl"
            if os.path.exists(prev_weights_file):
                # 只用於收集數據，不用於訓練
                data_collection_network = NTupleNetwork()
                data_collection_network.load_weights(prev_weights_file)
                print(f"Loaded weights from stage {stage-1} for data collection only")

                # 如果沒有數據，使用載入的模型收集數據
                if start_data is None or len(start_data) == 0:
                    print(f"Collecting data for stage {stage} using stage {stage-1} model...")
                    start_data = trainer._collect_next_stage_data(data_collection_network, stage-2)  # stage-2 因為索引從 0 開始

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
