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

# def check_win(board, player):
#     """
#     Checks if the given player has won on the board.
#     A win is 6 consecutive stones horizontally, vertically, or diagonally.
#     """
#     # Directions to check: Horizontal, Vertical, Diagonal Down-Right, Diagonal Up-Right
#     directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

#     for r in range(BOARD_SIZE):
#         for c in range(BOARD_SIZE):
#             if board[r][c] == player:
#                 for dr, dc in directions:
#                     count = 0
#                     for i in range(6):
#                         nr, nc = r + i * dr, c + i * dc
#                         if is_valid(nr, nc) and board[nr][nc] == player:
#                             count += 1
#                         else:
#                             break # Not consecutive in this direction
#                     if count == 6:
#                         return True # Found a win
#     return False # No win found for this player


def find_patterns(board, player):
    """
    Return all the patterns found in this board.
    Patterns are defined as: (start_point, direction, to_fill)
    Means that starting from start_point, follow direction, filling to_fill will generate a winning pattern
    Of course, inpossible patterns will not be counted. That means that there will be no other players move in this pattern
    """
    # logger.debug(f"Finding patterns for player {player}")
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
    # logger.debug("Findind dangerous positions")
    patterns = find_patterns(board, player)
    dangerous_patterns = [p for p in patterns if len(p[2]) <= depth]
    # logger.debug(dangerous_patterns)
    dangerous_positions = []
    for p in dangerous_patterns:
        dangerous_positions.extend(p[2])
    return dangerous_positions


def generate_winning_move(board):
    player, move = get_player_and_moves(board)
    dangerous_positions = generate_dangerous_positions(board, player, move)

    if not dangerous_positions:
        return None
    
    # logger.debug(f"Winning! {dangerous_positions[0]}")
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
    
    # logger.debug("Defending!", dangerous_positions)
    
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
    
    # logger.debug("GG, losing")
    return dangerous_positions[0]

def get_player_and_moves(board):
    count_1, count_2 = 0, 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 1:
                count_1 += 1
            elif board[r][c] == 2:
                count_2 += 1

    # logger.debug(f"Player 1: {count_1}, Player 2: {count_2}")

    if count_1 == count_2:
        return 1, 1 # Black's first move
    elif count_1 > count_2:
        return 2, 2 # White's turn
    else: # count_1 == count_2 and count_1 > 0
        return 1, 2 # Black's turn

def get_empty_cells(board):
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == 0]

def generate_best_move(board):
    """
    Generates the best move using MCTS when no immediate win/loss is apparent.
    """

    empty = get_empty_cells(board)
    player, move = get_player_and_moves(board)
    
    best_score = -10000000
    best_move = None

    if move == 1:
        for r, c in empty:
            board[r][c] = 1
            score = evaluate_board(board)
            board[r][c] = 0
            if score > best_score:
                best_score = score
                best_move = (r, c)
    
    elif move == 2:
        dx = [1, 0, 1, 1, -1, 0, -1, -1]
        dy = [0, 1, 1, -1, 1, -1, 0, -1]

        def has_neighbor(r, c):
            for d in range(8):
                nr, nc = r + dx[d], c + dy[d]
                if is_valid(nr, nc) and board[nr][nc] != 0:
                    return True
            return False


        for move1, move2 in itertools.combinations(empty, 2):
            r1, c1 = move1
            r2, c2 = move2
            if not has_neighbor(r1, c1) or not has_neighbor(r2, c2):
                continue
            board[r1][c1] = 1
            board[r2][c2] = 1
            score = evaluate_board(board)
            if score > best_score:
                best_score = score
                best_move = (r1, c1)
            board[r1][c1] = 0
            board[r2][c2] = 0
    
    return best_move        


def evaluate_board(board):
    black_patterns = find_patterns(board, 1)
    white_patterns = find_patterns(board, 2)
    
    score = 0
    for pattern in black_patterns:
        length = 6 - len(pattern[2])
        score += 15 ** (length * 2)
    
    for pattern in white_patterns:
        length = 6 - len(pattern[2])
        score -= 15 ** (1 + length * 2)
    return score


def select_move(board, color):
    if color == "W":
        logger.log("White!")
        board_cp = board.copy()
        for c in range(BOARD_SIZE):
            for r in range(BOARD_SIZE):
                if board_cp[c][r] != 0:
                    board_cp[c][r] = 3 - board_cp[c][r]
        return select_move(board_cp, "B")
    
    player, move = get_player_and_moves(board)
    logger.debug(f"Player {player}, Move {move}")
    
    logger.debug("Finding winning move...")
    winning_move = generate_winning_move(board)
    if winning_move:
        return winning_move
    
    logger.debug("Finding defending move...")
    defending_move = generate_defending_move(board)
    if defending_move:
        return defending_move
    
    logger.debug("No winning or defending, using MCTS")
    best_move = generate_best_move(board)    
    return best_move


    





    
