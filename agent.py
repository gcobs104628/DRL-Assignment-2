import itertools
import copy
import math
import random
import time
from loguru import logger

BOARD_SIZE = 19 # Assuming a 19x19 board

def is_valid(c, r):
    """Check if coordinates are within the board bounds."""
    return 0 <= c < BOARD_SIZE and 0 <= r < BOARD_SIZE

# def check_winning(board, position, player):
#     r, c = position
#     directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
#     for dr, dc in directions:
#         count = 1
#         for i in range(1, 6):
#             nr, nc = r + i * dr, c + i * dc
#             if not is_valid(nr, nc) or board[nr][nc] != player:
#                 break
#             count += 1
#         for i in range(1, 6):
#             nr, nc = r - i * dr, c - i * dc
#             if not is_valid(nr, nc) or board[nr][nc] != player:
#                 break
#             count += 1
#         if count >= 6:
#             return True
#     return False  

# def winning_in_k_moves(board, player, k):
#     empty_cells = get_empty_cells(board)
#     if k == 1:
#         for r, c in empty_cells:
#             board[r][c] = player
#             if check_winning(board, (r, c), player):
#                 board[r][c] = 0
#                 return (r, c)
#             board[r][c] = 0
#         return None
    
#     for r, c in empty_cells:
#         board[r][c] = player
#         if winning_in_k_moves(board, player, k - 1):
#             board[r][c] = 0
#             return (r, c)
#         board[r][c] = 0
#     return None

def check_win(board, player):
    """
    Checks if the given player has won on the board.
    A win is 6 consecutive stones horizontally, vertically, or diagonally.
    """
    # Directions to check: Horizontal, Vertical, Diagonal Down-Right, Diagonal Up-Right
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player:
                for dr, dc in directions:
                    count = 0
                    for i in range(6):
                        nr, nc = r + i * dr, c + i * dc
                        if is_valid(nr, nc) and board[nr][nc] == player:
                            count += 1
                        else:
                            break # Not consecutive in this direction
                    if count == 6:
                        return True # Found a win
    return False # No win found for this player


def find_patterns(board, player):
    """
    Return all the patterns found in this board.
    Patterns are defined as: (start_point, direction, to_fill)
    Means that starting from start_point, follow direction, filling to_fill will generate a winning pattern
    Of course, inpossible patterns will not be counted. That means that there will be no other players move in this pattern
    """
    logger.debug(f"Finding patterns for player {player}")
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    patterns = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for dir in directions:
                to_fill = []
                for i in range(6):
                    nr, nc = r + i * dir[0], c + i * dir[1]
                    if not is_valid(nr, nc) or board[nr][nc] == 3 - player:
                        break
                    if board[nr][nc] == 0:
                        to_fill.append((nr, nc))
                    if i == 5:
                        patterns.append(((r, c), dir, to_fill))
    return patterns

def generate_dangerous_positions(board, player, depth=2):
    logger.debug("Findind dangerous positions")
    patterns = find_patterns(board, player)
    dangerous_patterns = [p for p in patterns if len(p[2]) <= depth]
    logger.debug(dangerous_patterns)
    dangerous_positions = []
    for p in dangerous_patterns:
        dangerous_positions.extend(p[2])
    return dangerous_positions


def generate_winning_move(board):
    player, move = get_player_and_moves(board)

    logger.debug(f"{player} {move}")
    dangerous_positions = generate_dangerous_positions(board, player, move)

    if not dangerous_positions:
        return None
    
    logger.debug(f"Winning! {dangerous_positions[0]}")
    return dangerous_positions[0]
    

def generate_defending_move(board):
    """
    Finds a move for Black (player 1) to block an immediate win by White (player 2).
    This assumes Black cannot win on the current turn and plays 2 stones.

    Args:
        board: A 2D list representing the game board (e.g., 19x19).
               0 represents empty, 1 represents Black, 2 represents White.

    Returns:
        - ((r1, c1), (r2, c2)): The pair of coordinates Black should play to
                                 block White's immediate win.
        - None: If White does not have an immediate winning move to block.
    """
    dangerous_positions = generate_dangerous_positions(board, 2)

    if not dangerous_positions:
        return None
    
    logger.debug("Defending!", dangerous_positions)
    
    for move1 in dangerous_positions:
        r, c = move1
        board[r][c] = 1
        d = generate_dangerous_positions(board, 2)
        if not d:
            board[r][c] = 0
            return move1
        board[r][c] = 0


    for move1, move2 in itertools.combinations(dangerous_positions, 2):
        r1, c1 = move1
        r2, c2 = move2

        board[r1][c1] = 1
        board[r2][c2] = 1

        d = generate_dangerous_positions(board, 2)
        if not d:
            board[r1][c1] = 0
            board[r2][c2] = 0
            return move1
        
        board[r1][c1] = 0
        board[r2][c2] = 0
        
    return None

def is_valid(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

def check_terminal(board):
    if check_win(board, 1):
        return 1 # Black wins
    if check_win(board, 2):
        return 2 # White wins
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 0:
                return None # Game is ongoing
    return 0 # Draw

def get_player_and_moves(board):
    count_1 = 0
    count_2 = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 1:
                count_1 += 1
            elif board[r][c] == 2:
                count_2 += 1

    logger.debug(f"Player 1: {count_1}, Player 2: {count_2}")

    if count_1 == count_2:
        return 1, 1 # Black's first move
    elif count_1 > count_2:
        return 2, 2 # White's turn
    else: # count_1 == count_2 and count_1 > 0
        return 1, 2 # Black's turn

def get_empty_cells(board):
     return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == 0]

def get_legal_moves(board, player, num_moves, empty_cells):
    if num_moves == 1:
        return empty_cells # List of (r, c)
    elif num_moves == 2:
        if len(empty_cells) < 2:
            return []
        # Return list of ((r1, c1), (r2, c2)) tuples
        return empty_cells
    return []

def apply_move(board, move, player):
    new_board = copy.deepcopy(board)
    if isinstance(move[0], tuple): # It's a pair of moves ((r1, c1), (r2, c2))
        (r1, c1), (r2, c2) = move
        if new_board[r1][c1] == 0: new_board[r1][c1] = player
        if new_board[r2][c2] == 0: new_board[r2][c2] = player
    else: # It's a single move (r, c)
        r, c = move
        if new_board[r][c] == 0: new_board[r][c] = player
    return new_board

# --- MCTS Node ---
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state # The board state
        self.parent = parent
        self.move = move # Move that led from parent to this state
        self.children = []
        self.wins = 0
        self.visits = 0
        self.player_turn, self.num_moves = get_player_and_moves(state)
        self.is_terminal = check_terminal(state) is not None
        self.untried_moves = self.get_potential_moves()


    def get_potential_moves(self):
        if self.is_terminal:
            return []
        empty = get_empty_cells(self.state)
        return get_legal_moves(self.state, self.player_turn, self.num_moves, empty)

    def uct_select_child(self, exploration_value=1.414):
        """Selects child using the UCT formula."""
        log_parent_visits = math.log(self.visits)
        s = sorted(self.children, key=lambda c: (c.wins / c.visits) + exploration_value * math.sqrt(log_parent_visits / c.visits))
        return s[-1] # Return child with highest UCT value

    def expand(self):
        """Expands the node by adding one child for an untried move."""
        if not self.untried_moves:
            # Should ideally not happen if called correctly, maybe raise error or return self
            return self # Or handle error
        
        move = self.untried_moves.pop()
        new_state = apply_move(self.state, move, self.player_turn)
        child_node = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def simulate_random_playout(self):
        """Simulates a random game from the current state."""
        current_state = copy.deepcopy(self.state)
        terminal_status = check_terminal(current_state)

        while terminal_status is None:
            player, num_moves = get_player_and_moves(current_state)
            empty = get_empty_cells(current_state)
            legal_moves = get_legal_moves(current_state, player, num_moves, empty)

            if not legal_moves: # No moves possible, should be a draw?
                terminal_status = 0
                break

            move = random.choice(legal_moves)
            current_state = apply_move(current_state, move, player)
            terminal_status = check_terminal(current_state)

        return terminal_status # Returns winner (1 or 2) or draw (0)

    def backpropagate(self, result):
        """Updates wins/visits from this node back up to the root."""
        node = self
        while node is not None:
            node.visits += 1
            # Important: Increment win count if the simulation result matches
            # the player *whose turn it was* just BEFORE the move leading to this node.
            # Equivalently: if the result matches the player who is NOT about to move from this node.
            parent_player = node.parent.player_turn if node.parent else (3 - node.player_turn) # Infer player who moved to get here
            if result == parent_player: # If the player who moved to get here won
                 node.wins += 1
            elif result == 0: # Handle draws - optional, e.g., 0.5 points
                 node.wins += 0.5
            # Else: the other player won, score is 0 for this node perspective implicitly
            
            node = node.parent # Move up the tree

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_leaf(self):
        return len(self.children) == 0


# --- MCTS Main Function ---
def run_mcts(root_state, iterations=None, time_limit=None, exploration_value=1.414):
    """
    Runs the MCTS algorithm.

    Args:
        root_state: The initial board state.
        iterations: Number of iterations to run.
        time_limit: Time limit in seconds.
        exploration_value: The 'C' parameter for UCT.

    Returns:
        The best move found from the root state.
    """
    root_node = MCTSNode(state=root_state)
    start_time = time.time()
    iter_count = 0

    while True:
        # Check termination condition
        if iterations is not None and iter_count >= iterations:
            break
        if time_limit is not None and time.time() - start_time >= time_limit:
            break

        node = root_node
        # 1. Selection
        while node.is_fully_expanded() and not node.is_leaf() and not node.is_terminal:
            node = node.uct_select_child(exploration_value)

        # 2. Expansion
        if not node.is_terminal and not node.is_fully_expanded():
            node = node.expand() # Expand one child

        # 3. Simulation
        if node.is_terminal:
            simulation_result = check_terminal(node.state)
        else:
             simulation_result = node.simulate_random_playout() # Simulate from the new/selected node

        # 4. Backpropagation
        node.backpropagate(simulation_result)

        iter_count += 1

    # After loop, select the best move based on visits (most robust)
    if not root_node.children:
        empty = get_empty_cells(root_state)
        player, num_moves = get_player_and_moves(root_state)
        legal = get_legal_moves(root_state, player, num_moves, empty)
        return random.choice(legal) if legal else None


    best_child = sorted(root_node.children, key=lambda c: c.visits)[-1]
    return best_child.move


# --- Main Function to Call ---
def generate_best_move(board):
    """
    Generates the best move using MCTS when no immediate win/loss is apparent.
    """

    empty = get_empty_cells(board)
    return random.choice(empty)
    
    # You might want to add calls to generate_winning_move and generate_defending_move
    # here first, for efficiency, before resorting to MCTS.
    # E.g.:
    # my_player, _ = get_player_and_moves(board)
    # win_move = generate_winning_move(board) # Assuming this exists and checks for current player
    # if win_move: return win_move
    # defend_move = generate_defending_move(board) # Assuming this exists and checks opponent threat
    # if defend_move: return defend_move # Or maybe just block one part? Needs thought.
    # Adjust iterations or time_limit based on desired strength/speed
    best_move = run_mcts(root_state=board, iterations=100, time_limit=1.0) # e.g., 1000 iterations
    # best_move = run_mcts(root_state=board, iterations=None, time_limit=5.0) # e.g., 5 seconds

    return best_move

def select_move(board, color):
    if color == 2:
        board_cp = board.copy()
        for c in range(BOARD_SIZE):
            for r in range(BOARD_SIZE):
                if board_cp[c][r] != 0:
                    board_cp[c][r] = 3 - board_cp[c][r]
        return select_move(board_cp, 1)
    
    # player, move = get_player_and_moves(board)
    # logger.debug(f"Player {player}, Move {move}")
    
    # logger.debug("Finding winning move...")
    # winning_move = generate_winning_move(board)
    # if winning_move:
    #     return winning_move
    
    # logger.debug("Finding defending move...")

    # defending_move = generate_defending_move(board)
    # if defending_move:
    #     return defending_move
    
    logger.debug("No winning or defending, using MCTS")
    
    best_move = generate_best_move(board)    
    return best_move


    





    
