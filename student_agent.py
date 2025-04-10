from run_game import NTupleApproximator, Game2048Env, select_best_action_2_step

patterns = [
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)),
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)),
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)

def get_action(state, score):
    env = Game2048Env()
    env.board = state
    env.score = score
    action = select_best_action_2_step(env, approximator)
    return action


def main(num_episodes=5):
    final_scores = []
    
    env = Game2048Env()
    for episode in range(num_episodes):
        env.reset()
        done = False

        while not done:
            action = select_best_action_2_step(env, approximator)
            _, current_score, done, _ = env.step(action)

        print(f"Game {episode + 1} completed! Score: ", env.score)
        final_scores.append(env.score)

    print("Average Score: ", sum(final_scores) / len(final_scores))
    return final_scores

if __name__ == "__main__":
    main()
