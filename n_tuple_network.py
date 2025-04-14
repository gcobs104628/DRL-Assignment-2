import numpy as np
import pickle
import os

class NTupleNetwork:
    # Constants for board dimensions and feature limits
    BOARD_SIZE = 4
    MAX_EMPTY_CELLS = 16
    MAX_TILE_TYPES = 16
    MAX_MERGEABLE_PAIRS = 24
    MAX_V_2V_PAIRS = 24

    def __init__(self, tuple_configs=None, feature_configs=None, init_method='zero', init_value=0, max_tile_power=16):
        """
        Initialize the N-Tuple Network with Matsuzaki's 8x6-tuple structure.

        Args:
            tuple_configs: Optional custom tuple configurations (defaults to Matsuzaki's 8x6-tuple)
            feature_configs: Optional configuration for additional features
            init_method: 'zero' or 'optimistic' initialization
            init_value: Value for optimistic initialization (if init_method='optimistic')
            max_tile_power: Maximum power of 2 for tile values (e.g., 16 for 2^16=65536)
        """
        # Store feature configurations for future use if needed
        self.feature_configs = feature_configs
        # Define the 8 different 6-tuples as shown in Fig. 5 (a)-(h) from Matsuzaki's paper
        self.base_tuples = tuple_configs if tuple_configs is not None else [
            # (a) Snake-shaped tuple
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2)],
            # (b) Snake-shaped tuple (different orientation)
            [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (2, 1)],
            # (c) Snake-shaped tuple (different orientation)
            [(3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (2, 2)],
            # (d) Snake-shaped tuple (different orientation)
            [(0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (2, 2)],
            # (e) Corner-based tuple
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
            # (f) Corner-based tuple (different orientation)
            [(0, 3), (0, 2), (0, 1), (1, 3), (1, 2), (2, 3)],
            # (g) Corner-based tuple (different orientation)
            [(3, 3), (3, 2), (3, 1), (2, 3), (2, 2), (1, 3)],
            # (h) Corner-based tuple (different orientation)
            [(3, 0), (3, 1), (3, 2), (2, 0), (2, 1), (1, 0)]
        ]

        # Store configuration parameters
        self.max_tile_power = max_tile_power
        self.num_tuples = len(self.base_tuples)
        self.num_symmetries = 8  # 4 rotations × 2 reflections

        # Initialize lookup tables (LUTs) for each tuple
        self.luts = [{} for _ in range(self.num_tuples)]

        # Additional feature LUTs
        self.large_tile_lut = {}  # Large tile feature
        self.empty_cells_lut = {}  # Number of empty cells
        self.tile_types_lut = {}  # Number of different tile types
        self.mergeable_pairs_lut = {}  # Number of mergeable tile pairs
        self.v_2v_pairs_lut = {}  # Number of adjacent v-2v tile pairs

        # Estimate the number of features per state for weight initialization
        self.num_features_per_state = self.num_tuples * self.num_symmetries + 5  # tuples + 5 additional features

        # Store initialization parameters for lazy initialization
        self.init_method = init_method
        self.init_value = init_value
        self._initialize_luts_on_demand = True

        # Initialize weights based on the specified method
        if init_method == 'optimistic' and init_value > 0:
            self._optimistic_initialization(init_value)

    def _optimistic_initialization(self, v_init):
        """
        Initialize all weights with optimistic values.

        Args:
            v_init: Total initial value to distribute among features
        """
        # Estimate the number of features per state for weight initialization
        # This is a more principled approach than using a fixed number
        # We have num_tuples * num_symmetries tuple features + 5 additional features
        weight_per_feature = v_init / self.num_features_per_state

        # Initialize additional feature LUTs with optimistic values
        # Use class constants for ranges to improve readability
        for i in range(self.MAX_EMPTY_CELLS + 1):  # 0-16 empty cells
            self.empty_cells_lut[i] = weight_per_feature

        for i in range(self.MAX_TILE_TYPES + 1):  # 0-16 different tile types
            self.tile_types_lut[i] = weight_per_feature

        for i in range(self.MAX_MERGEABLE_PAIRS + 1):  # 0-24 mergeable pairs
            self.mergeable_pairs_lut[i] = weight_per_feature

        for i in range(self.MAX_V_2V_PAIRS + 1):  # 0-24 v-2v pairs
            self.v_2v_pairs_lut[i] = weight_per_feature

        # Initialize large tile LUT for combinations of large tiles
        # Format: (n_2048, n_4096, n_8192, n_16384, n_32768)
        # Only initialize combinations that could actually occur (sum <= 16)
        for n2k in range(5):
            for n4k in range(5):
                for n8k in range(5):
                    for n16k in range(5):
                        for n32k in range(5):
                            if n2k + n4k + n8k + n16k + n32k <= 16:  # Maximum 16 tiles on board
                                self.large_tile_lut[(n2k, n4k, n8k, n16k, n32k)] = weight_per_feature

        # Set flag to indicate we've initialized the LUTs
        self._initialize_luts_on_demand = False

    def _initialize_all_luts(self):
        """Initialize all lookup tables with zeros"""
        # We'll use a lazy initialization approach instead of pre-filling all LUTs
        # This keeps the LUTs sparse and only adds entries when they're actually encountered
        # Just set the flag to indicate we should initialize on demand
        self._initialize_luts_on_demand = True

    def get_tuple_indices(self, state):
        """Get indices for all tuples in the state with all symmetries"""
        indices = []
        for tuple_idx, tuple_pattern in enumerate(self.base_tuples):
            # Get values for the basic pattern
            values = tuple(int(np.log2(state[i][j])) if state[i][j] > 0 else 0 for i, j in tuple_pattern)

            # Apply 8 symmetries (rotations and reflections)
            # Pass the full state to _get_symmetries for proper symmetry handling
            symmetries = self._get_symmetries(values, tuple_idx, state)
            indices.append((tuple_idx, symmetries))

        return indices

    def _get_symmetries(self, values, tuple_idx, state):
        """
        Generate 8 symmetrical transformations for a tuple based on the full board state.

        Args:
            values: The original tuple values (not used directly, kept for API compatibility)
            tuple_idx: Index of the tuple pattern
            state: The full game state (4x4 board) for computing symmetries

        Returns:
            List of symmetrical tuple indices
        """
        # For a proper implementation, we need to compute all 8 symmetries
        # This includes 4 rotations and their 4 reflections

        # Safety check if state is somehow None
        if state is None:
            return [tuple(values)]  # Return original values if state is missing

        # Get the tuple pattern coordinates
        pattern = self.base_tuples[tuple_idx]

        # Generate all 8 symmetrical transformations of the FULL board
        symmetrical_boards = self._get_symmetrical_boards(state)

        # Extract the tuple values from each symmetrical board
        symmetries = []
        for sym_board in symmetrical_boards:
            try:
                # Extract values from the symmetrical board using the original pattern coordinates
                # Apply log2 transformation to match what's done in get_tuple_indices
                # Ensure we don't try to compute log2(0) which would cause an error
                sym_values = tuple(int(np.log2(sym_board[row, col])) if sym_board[row, col] > 0 else 0
                                  for row, col in pattern)
                symmetries.append(sym_values)
            except IndexError:
                # Should not happen if board is always 4x4 and patterns are valid
                print(f"Warning: IndexError during symmetry extraction for tuple {tuple_idx}")
                continue  # Skip this symmetry if error occurs

        # Remove duplicates by converting to a set and back to a list
        return list(set(symmetries))

    def _get_symmetrical_boards(self, board):
        """
        Generate all 8 symmetrical transformations of a board.

        Args:
            board: The original board (4x4 numpy array)

        Returns:
            List of 8 symmetrical boards
        """
        # Original board
        boards = [board.copy()]

        # 90-degree rotation
        rot90 = np.rot90(board, k=1)
        boards.append(rot90.copy())

        # 180-degree rotation
        rot180 = np.rot90(board, k=2)
        boards.append(rot180.copy())

        # 270-degree rotation
        rot270 = np.rot90(board, k=3)
        boards.append(rot270.copy())

        # Horizontal reflection
        hflip = np.fliplr(board)
        boards.append(hflip.copy())

        # Horizontal reflection + 90-degree rotation
        hflip_rot90 = np.rot90(hflip, k=1)
        boards.append(hflip_rot90.copy())

        # Horizontal reflection + 180-degree rotation
        hflip_rot180 = np.rot90(hflip, k=2)
        boards.append(hflip_rot180.copy())

        # Horizontal reflection + 270-degree rotation
        hflip_rot270 = np.rot90(hflip, k=3)
        boards.append(hflip_rot270.copy())

        return boards

    def get_extra_features(self, state):
        """Extract additional features from the state"""
        features = {}

        # 1. Large tile feature
        large_tiles = self._count_large_tiles(state)
        features['large_tiles'] = large_tiles

        # 2. Number of empty cells
        empty_cells = np.sum(state == 0)
        features['empty_cells'] = empty_cells

        # 3. Number of different tile types
        unique_tiles = len(np.unique(state[state > 0]))
        features['tile_types'] = unique_tiles

        # 4. Number of mergeable tile pairs
        mergeable_pairs = self._count_mergeable_pairs(state)
        features['mergeable_pairs'] = mergeable_pairs

        # 5. Number of adjacent v-2v tile pairs
        v_2v_pairs = self._count_v_2v_pairs(state)
        features['v_2v_pairs'] = v_2v_pairs

        return features

    def _count_large_tiles(self, state):
        """Count the number of large tiles (2048, 4096, 8192, 16384, 32768)"""
        n_2048 = np.sum(state == 2048)
        n_4096 = np.sum(state == 4096)
        n_8192 = np.sum(state == 8192)
        n_16384 = np.sum(state == 16384)
        n_32768 = np.sum(state == 32768)

        return (n_2048, n_4096, n_8192, n_16384, n_32768)

    def _count_mergeable_pairs(self, state):
        """Count the number of mergeable tile pairs"""
        count = 0

        # Check horizontally
        for i in range(4):
            for j in range(3):
                if state[i, j] > 0 and state[i, j] == state[i, j+1]:
                    count += 1

        # Check vertically
        for i in range(3):
            for j in range(4):
                if state[i, j] > 0 and state[i, j] == state[i+1, j]:
                    count += 1

        return count

    def _count_v_2v_pairs(self, state):
        """Count the number of adjacent v-2v tile pairs"""
        count = 0

        # Check horizontally
        for i in range(4):
            for j in range(3):
                if state[i, j] > 0 and state[i, j+1] > 0:
                    if state[i, j] == 2 * state[i, j+1] or state[i, j+1] == 2 * state[i, j]:
                        count += 1

        # Check vertically
        for i in range(3):
            for j in range(4):
                if state[i, j] > 0 and state[i+1, j] > 0:
                    if state[i, j] == 2 * state[i+1, j] or state[i+1, j] == 2 * state[i, j]:
                        count += 1

        return count

    def evaluate(self, state):
        """Evaluate the value of a state"""
        if state is None:
            return 0.0

        value = 0.0

        # 1. N-Tuple features value
        tuple_indices = self.get_tuple_indices(state)
        for tuple_idx, symmetries in tuple_indices:
            for sym in symmetries:
                # If the feature is not in the LUT, it's considered to have a value of 0
                # or we could initialize it with an optimistic value if using optimistic initialization
                if sym in self.luts[tuple_idx]:
                    value += self.luts[tuple_idx][sym]
                elif self._initialize_luts_on_demand and self.init_method == 'optimistic' and self.init_value > 0:
                    # Lazy initialization for optimistic values during evaluation
                    self.luts[tuple_idx][sym] = self.init_value / self.num_features_per_state
                    value += self.luts[tuple_idx][sym]

        # 2. Additional features value
        extra_features = self.get_extra_features(state)

        # Handle large_tile_lut
        if extra_features['large_tiles'] in self.large_tile_lut:
            value += self.large_tile_lut[extra_features['large_tiles']]
        elif self._initialize_luts_on_demand and self.init_method == 'optimistic' and self.init_value > 0:
            self.large_tile_lut[extra_features['large_tiles']] = self.init_value / self.num_features_per_state
            value += self.large_tile_lut[extra_features['large_tiles']]

        # Handle empty_cells_lut
        if extra_features['empty_cells'] in self.empty_cells_lut:
            value += self.empty_cells_lut[extra_features['empty_cells']]
        elif self._initialize_luts_on_demand and self.init_method == 'optimistic' and self.init_value > 0:
            self.empty_cells_lut[extra_features['empty_cells']] = self.init_value / self.num_features_per_state
            value += self.empty_cells_lut[extra_features['empty_cells']]

        # Handle tile_types_lut
        if extra_features['tile_types'] in self.tile_types_lut:
            value += self.tile_types_lut[extra_features['tile_types']]
        elif self._initialize_luts_on_demand and self.init_method == 'optimistic' and self.init_value > 0:
            self.tile_types_lut[extra_features['tile_types']] = self.init_value / self.num_features_per_state
            value += self.tile_types_lut[extra_features['tile_types']]

        # Handle mergeable_pairs_lut
        if extra_features['mergeable_pairs'] in self.mergeable_pairs_lut:
            value += self.mergeable_pairs_lut[extra_features['mergeable_pairs']]
        elif self._initialize_luts_on_demand and self.init_method == 'optimistic' and self.init_value > 0:
            self.mergeable_pairs_lut[extra_features['mergeable_pairs']] = self.init_value / self.num_features_per_state
            value += self.mergeable_pairs_lut[extra_features['mergeable_pairs']]

        # Handle v_2v_pairs_lut
        if extra_features['v_2v_pairs'] in self.v_2v_pairs_lut:
            value += self.v_2v_pairs_lut[extra_features['v_2v_pairs']]
        elif self._initialize_luts_on_demand and self.init_method == 'optimistic' and self.init_value > 0:
            self.v_2v_pairs_lut[extra_features['v_2v_pairs']] = self.init_value / self.num_features_per_state
            value += self.v_2v_pairs_lut[extra_features['v_2v_pairs']]

        return value

    def update(self, state, td_error, alpha):
        """Update weights based on TD error"""
        if state is None:
            return

        # Calculate the number of active features for normalization
        active_features = 0

        # 1. Update N-Tuple weights
        tuple_indices = self.get_tuple_indices(state)
        for tuple_idx, symmetries in tuple_indices:
            for sym in symmetries:
                if sym not in self.luts[tuple_idx]:
                    # Initialize new features based on initialization method
                    if self.init_method == 'optimistic' and self.init_value > 0:
                        self.luts[tuple_idx][sym] = self.init_value / self.num_features_per_state
                    else:
                        self.luts[tuple_idx][sym] = 0.0
                active_features += 1

        # 2. Update additional feature weights
        extra_features = self.get_extra_features(state)
        active_features += 5  # 5 additional features

        # Safety check: if no active features, return without updating
        if active_features == 0:
            return

        # Normalize the learning rate by the number of active features
        # With proper symmetry handling, active_features will be much larger (up to 8 * num_tuples + 5)
        # We use a more balanced normalization approach to prevent the learning rate from becoming too small
        normalized_alpha = alpha / active_features

        # Update N-Tuple weights
        for tuple_idx, symmetries in tuple_indices:
            for sym in symmetries:
                self.luts[tuple_idx][sym] += normalized_alpha * td_error

        # Update additional feature weights
        if extra_features['large_tiles'] in self.large_tile_lut:
            self.large_tile_lut[extra_features['large_tiles']] += normalized_alpha * td_error
        else:
            # Initialize on first encounter
            if self.init_method == 'optimistic' and self.init_value > 0:
                self.large_tile_lut[extra_features['large_tiles']] = self.init_value / self.num_features_per_state
            else:
                self.large_tile_lut[extra_features['large_tiles']] = 0.0
            # Then update with the current TD error
            self.large_tile_lut[extra_features['large_tiles']] += normalized_alpha * td_error

        if extra_features['empty_cells'] in self.empty_cells_lut:
            self.empty_cells_lut[extra_features['empty_cells']] += normalized_alpha * td_error
        else:
            if self.init_method == 'optimistic' and self.init_value > 0:
                self.empty_cells_lut[extra_features['empty_cells']] = self.init_value / self.num_features_per_state
            else:
                self.empty_cells_lut[extra_features['empty_cells']] = 0.0
            self.empty_cells_lut[extra_features['empty_cells']] += normalized_alpha * td_error

        if extra_features['tile_types'] in self.tile_types_lut:
            self.tile_types_lut[extra_features['tile_types']] += normalized_alpha * td_error
        else:
            if self.init_method == 'optimistic' and self.init_value > 0:
                self.tile_types_lut[extra_features['tile_types']] = self.init_value / self.num_features_per_state
            else:
                self.tile_types_lut[extra_features['tile_types']] = 0.0
            self.tile_types_lut[extra_features['tile_types']] += normalized_alpha * td_error

        if extra_features['mergeable_pairs'] in self.mergeable_pairs_lut:
            self.mergeable_pairs_lut[extra_features['mergeable_pairs']] += normalized_alpha * td_error
        else:
            if self.init_method == 'optimistic' and self.init_value > 0:
                self.mergeable_pairs_lut[extra_features['mergeable_pairs']] = self.init_value / self.num_features_per_state
            else:
                self.mergeable_pairs_lut[extra_features['mergeable_pairs']] = 0.0
            self.mergeable_pairs_lut[extra_features['mergeable_pairs']] += normalized_alpha * td_error

        if extra_features['v_2v_pairs'] in self.v_2v_pairs_lut:
            self.v_2v_pairs_lut[extra_features['v_2v_pairs']] += normalized_alpha * td_error
        else:
            if self.init_method == 'optimistic' and self.init_value > 0:
                self.v_2v_pairs_lut[extra_features['v_2v_pairs']] = self.init_value / self.num_features_per_state
            else:
                self.v_2v_pairs_lut[extra_features['v_2v_pairs']] = 0.0
            self.v_2v_pairs_lut[extra_features['v_2v_pairs']] += normalized_alpha * td_error

    def get_weights(self):
        """Get all weights as a dictionary"""
        return {
            'luts': self.luts,
            'large_tile_lut': self.large_tile_lut,
            'empty_cells_lut': self.empty_cells_lut,
            'tile_types_lut': self.tile_types_lut,
            'mergeable_pairs_lut': self.mergeable_pairs_lut,
            'v_2v_pairs_lut': self.v_2v_pairs_lut
        }

    def save_weights(self, filename):
        """Save all weights to a file"""
        weights = self.get_weights()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, filename):
        """Load weights from a file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

                # 檢查是否是新格式的檢查點文件
                if isinstance(data, dict) and 'weights' in data:
                    # 新格式：從檢查點文件加載權重
                    weights = data['weights']
                elif isinstance(data, dict) and 'luts' in data:
                    # 舊格式：直接加載權重
                    weights = data
                else:
                    raise ValueError("Invalid weights format")

                self.luts = weights['luts']
                self.large_tile_lut = weights['large_tile_lut']
                self.empty_cells_lut = weights['empty_cells_lut']
                self.tile_types_lut = weights['tile_types_lut']
                self.mergeable_pairs_lut = weights['mergeable_pairs_lut']
                self.v_2v_pairs_lut = weights['v_2v_pairs_lut']

            return True
        except Exception as e:
            print(f"Failed to load weights: {e}")
            return False
