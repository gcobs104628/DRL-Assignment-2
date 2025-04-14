import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from game_env import Game2048Env
from n_tuple_network import NTupleNetwork
from mcts_search import MCTS
from student_agent import get_action

def test_agent(num_games=10, render_final=True):
    """
    Test the MCTS agent on multiple games.
    
    Args:
        num_games: Number of games to play
        render_final: Whether to render the final state of each game
        
    Returns:
        List of scores
    """
    env = Game2048Env()
    scores = []
    max_tiles = []
    
    for game in tqdm(range(num_games)):
        state = env.reset()
        done = False
        step_count = 0
        
        while not done:
            # Get action from our agent
            action = get_action(state, env.score)
            
            # Perform the action in the environment
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
            step_count += 1
            
            # Optional: Print progress
            if step_count % 100 == 0:
                print(f"Game {game+1}, Step {step_count}, Score: {env.score}")
        
        # Record results
        scores.append(env.score)
        max_tile = np.max(env.board)
        max_tiles.append(max_tile)
        
        # Print results
        print(f"Game {game+1} finished. Score: {env.score}, Max Tile: {max_tile}, Steps: {step_count}")
        
        # Render final state
        if render_final:
            env.render()
    
    # Print statistics
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    avg_max_tile = np.mean(max_tiles)
    
    print(f"\nResults over {num_games} games:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Max Score: {max_score}")
    print(f"Average Max Tile: {avg_max_tile:.2f}")
    
    # Plot histogram of scores
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=20)
    plt.title(f'Score Distribution (Avg: {avg_score:.2f})')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig("DRL/DRL_HW2/DRL-Assignment-2/score_distribution.png")
    plt.close()
    
    return scores

def test_mcts_parameters(iterations_list=[10, 50, 100], exploration_weights=[0.1, 0.5, 1.0], games_per_config=5):
    """
    Test different MCTS parameters.
    
    Args:
        iterations_list: List of iteration counts to test
        exploration_weights: List of exploration weights to test
        games_per_config: Number of games to play for each configuration
        
    Returns:
        Dictionary of results
    """
    env = Game2048Env()
    network = NTupleNetwork()
    
    # Try to load trained weights
    weights_file = "DRL/DRL_HW2/DRL-Assignment-2/weights/trained_weights.pkl"
    try:
        network.load_weights(weights_file)
        print(f"Loaded weights from {weights_file}")
    except:
        print(f"Could not load weights from {weights_file}. Using default weights.")
    
    results = {}
    
    for iterations in iterations_list:
        for exploration_weight in exploration_weights:
            config = (iterations, exploration_weight)
            print(f"\nTesting MCTS with iterations={iterations}, exploration_weight={exploration_weight}")
            
            mcts = MCTS(network, env, iterations, exploration_weight)
            scores = []
            
            for game in tqdm(range(games_per_config)):
                state = env.reset()
                done = False
                
                while not done:
                    # Use MCTS to find the best action
                    action = mcts.search(state)
                    
                    # Perform the action in the environment
                    next_state, reward, done, info = env.step(action)
                    
                    # Update state
                    state = next_state
                
                # Record results
                scores.append(env.score)
                print(f"Game {game+1} finished. Score: {env.score}, Max Tile: {np.max(env.board)}")
            
            # Store results
            results[config] = {
                'scores': scores,
                'avg_score': np.mean(scores),
                'max_score': np.max(scores)
            }
            
            print(f"Average Score: {results[config]['avg_score']:.2f}")
            print(f"Max Score: {results[config]['max_score']}")
    
    # Print summary
    print("\nSummary of Results:")
    for config, result in results.items():
        iterations, exploration_weight = config
        print(f"Iterations: {iterations}, Exploration Weight: {exploration_weight}")
        print(f"  Average Score: {result['avg_score']:.2f}")
        print(f"  Max Score: {result['max_score']}")
    
    return results

if __name__ == "__main__":
    # Test the agent
    print("Testing MCTS agent...")
    scores = test_agent(num_games=3, render_final=True)
    
    # Uncomment to test different MCTS parameters
    # print("\nTesting different MCTS parameters...")
    # results = test_mcts_parameters(
    #     iterations_list=[10, 50, 100],
    #     exploration_weights=[0.1, 0.5],
    #     games_per_config=2
    # )
