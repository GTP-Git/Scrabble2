


#Scrabble 01MAY25 Cython V5




import time
import pickle
import math # Add any other necessary imports
from collections import Counter
import os



GRID_SIZE = 15
CENTER_SQUARE = (7, 7)

# --- Tile Data ---
TILE_DISTRIBUTION = {
    'A': (9, 1), 'B': (2, 3), 'C': (2, 3), 'D': (4, 2), 'E': (12, 1), 'F': (2, 4),
    'G': (3, 2), 'H': (2, 4), 'I': (9, 1), 'J': (1, 8), 'K': (1, 5), 'L': (4, 1),
    'M': (2, 3), 'N': (6, 1), 'O': (8, 1), 'P': (2, 3), 'Q': (1, 10), 'R': (6, 1),
    'S': (4, 1), 'T': (6, 1), 'U': (4, 1), 'V': (2, 4), 'W': (2, 4), 'X': (1, 8),
    'Y': (2, 4), 'Z': (1, 10), ' ': (2, 0)
}

# --- Board Colors (Used by calculate_score for multipliers) ---
# Note: These need to match the values used when creating the board in the main script
RED = (255, 100, 100)    # TW
PINK = (255, 182, 193)   # DW (and Center)
BLUE = (0, 102, 204)     # TL
LIGHT_BLUE = (135, 206, 250) # DL
# WHITE = (255, 255, 255) # Normal square - not strictly needed by calculate_score logic

# --- Coordinate Conversion ---
LETTERS = "ABCDEFGHIJKLMNO"  # For get_coord





##########################################################################################################
##########################################################################################################
##########################################################################################################






def perform_leave_lookup(leave_key_str):
    """
    Performs the lookup in the global LEAVE_LOOKUP_TABLE.
    Called by the Cython function.
    Uses LEAVE_LOOKUP_TABLE which must be defined later in this module.
    """
    # Access the global table (ensure it's loaded and accessible)
    # LEAVE_LOOKUP_TABLE is global in this file
    try:
        # Use the globally loaded table from this module
        # Add a check in case this is called before table is loaded (shouldn't happen now)
        if not isinstance(LEAVE_LOOKUP_TABLE, dict):
             print("Warning: perform_leave_lookup called before LEAVE_LOOKUP_TABLE is a dict.")
             return 0.0

        value = LEAVE_LOOKUP_TABLE.get(leave_key_str)
        if value is not None:
            return float(value)
        else:
            # print(f"--- DEBUG (Python Lookup): Key '{leave_key_str}' NOT FOUND.")
            return 0.0
    except NameError: # Catch if LEAVE_LOOKUP_TABLE doesn't exist yet
        print("Warning: perform_leave_lookup called before LEAVE_LOOKUP_TABLE is defined.")
        return 0.0
    except Exception as e:
        print(f"Error during Python leave lookup for key '{leave_key_str}': {e}")
        return 0.0






##########################################################################################################
##########################################################################################################
##########################################################################################################










# Inside scrabble_helpers.py

LEAVE_LOOKUP_TABLE = {}
lookup_file = "NWL23-leaves.pkl" # <<< USE YOUR CORRECT FILENAME HERE
try:
    # Ensure the path is correct if the file isn't in the same directory
    if os.path.exists(lookup_file):
        with open(lookup_file, 'rb') as f_lookup:
            LEAVE_LOOKUP_TABLE = pickle.load(f_lookup)
        print(f"--- SUCCESS: Loaded Leave Lookup Table from {lookup_file} ---")
        # Optional: Check if it loaded as a dictionary
        if not isinstance(LEAVE_LOOKUP_TABLE, dict):
            print(f"--- WARNING: Loaded leave lookup table is not a dictionary (type: {type(LEAVE_LOOKUP_TABLE)})! Resetting. ---")
            LEAVE_LOOKUP_TABLE = {}
        elif not LEAVE_LOOKUP_TABLE:
             print(f"--- WARNING: Loaded leave lookup table from {lookup_file} is empty. ---")
        

    else:
        print(f"--- WARNING: Leave Lookup Table file not found: {lookup_file} ---")
        print("---          Leave evaluation will return 0.0 for unknown leaves. ---")
        LEAVE_LOOKUP_TABLE = {} # Ensure it's an empty dict if file not found
except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
    print(f"--- ERROR: Failed to load or unpickle Leave Lookup Table from {lookup_file}: {e} ---")
    print("---        Leave evaluation will return 0.0 for unknown leaves. ---")
    LEAVE_LOOKUP_TABLE = {} # Ensure it's an empty dict on error
except Exception as e: # Catch other potential errors during file access/loading
    print(f"--- UNEXPECTED ERROR loading Leave Lookup Table from {lookup_file}: {e} ---")
    LEAVE_LOOKUP_TABLE = {} # Ensure it's an empty dict on error







##########################################################################################################
##########################################################################################################
##########################################################################################################








# --- GADDAG Node Definition (Add this to Scrabble Game.py) ---
class GaddagNode:
    """Represents a node in the GADDAG."""
    __slots__ = ['children', 'is_terminal'] # Memory optimization

    def __init__(self):
        self.children = {}  # Dictionary mapping letter -> GaddagNode
        self.is_terminal = False # True if a path ending here is a valid word/subword





##########################################################################################################
##########################################################################################################
##########################################################################################################





        

# --- GADDAG Class Definition (Add this to Scrabble Game.py) ---
class Gaddag:
    """
    Represents the GADDAG data structure.
    This class definition is needed to correctly unpickle the object.
    The actual building happens in gaddag_builder.py.
    """
    SEPARATOR = '>' # Special character used in GADDAG paths

    def __init__(self):
        # The root node will be populated when loading from pickle
        self.root = GaddagNode()

    # No insert method needed here, as we load a pre-built structure.




##########################################################################################################
##########################################################################################################
##########################################################################################################





class DAWG:
    """
    Represents the DAWG structure for efficient word lookups.
    The actual graph is built separately and loaded from a pickle file.
    This definition primarily supports unpickling and type checking.
    """
    def __init__(self):
        """Initializes the DAWG, typically setting the root node."""
        # The root will be overwritten when loading from pickle,
        # but initializing it is good practice.
        self.root = Node() # Or potentially an empty node object if you have one defined

    def search(self, word):
        """
        Checks if a word exists in the DAWG by traversing from the root.
        """
        node = self.root
        if node is None:
            print("Error: DAWG root is None during search.")
            return False

        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                return False # Character path not found
        return node.is_terminal

    # Add other methods here ONLY if they are fundamental to the class
    # structure itself and not just part of the loaded object's state/logic.
    # For example, methods to build the DAWG would NOT go here if you
    # build it separately.




##########################################################################################################
##########################################################################################################
##########################################################################################################





class Node:
    """Node for the DAWG structure."""
    __slots__ = ['children', 'is_terminal'] # Memory optimization

    def __init__(self):
        self.children = {}  # char -> Node
        self.is_terminal = False





##########################################################################################################
##########################################################################################################
##########################################################################################################








def get_anchor_points(tiles, is_first_play):
    """Get anchor points (empty squares adjacent to existing tiles) for valid moves."""
    anchors = set()
    if is_first_play: anchors.add(CENTER_SQUARE); return anchors
    has_tiles = False
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if tiles[r][c]: has_tiles = True # Check if board has any tiles
            if not tiles[r][c]: # Must be an empty square
                 is_anchor = False
                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                     nr, nc = r + dr, c + dc
                     if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and tiles[nr][nc]: is_anchor = True; break
                 if is_anchor: anchors.add((r, c))
    if not has_tiles and not is_first_play: anchors.add(CENTER_SQUARE) # Fallback if board empty but not first play
    return anchors



##########################################################################################################
##########################################################################################################
##########################################################################################################






def get_coord(start_pos, direction):
    """Converts (row, col), direction into Scrabble notation (e.g., 8H, H8)."""
    # Ensure start_pos is valid before unpacking
    if not isinstance(start_pos, tuple) or len(start_pos) != 2:
        print(f"Warning: Invalid start_pos '{start_pos}' passed to get_coord. Returning '??'.")
        return "??"
        
    row, col = start_pos

    # Validate row and col indices
    if not (0 <= row < GRID_SIZE and 0 <= col < len(LETTERS)):
         print(f"Warning: Invalid coordinates ({row},{col}) passed to get_coord. Returning '??'.")
         return "??"

    if direction == "right":
        # Horizontal: Row number (1-based) then Column Letter
        return f"{row + 1}{LETTERS[col]}"
    elif direction == "down":
        # Vertical: Column Letter then Row number (1-based)
        return f"{LETTERS[col]}{row + 1}"
    else:
        # Handle unexpected direction values
        print(f"Warning: Invalid direction '{direction}' passed to get_coord. Returning '??'.")
        return "??"


##########################################################################################################
##########################################################################################################
##########################################################################################################



'''
def evaluate_leave(rack, verbose=False):
    """
    Retrieves the pre-calculated leave value (float) from the LEAVE_LOOKUP_TABLE.
    Handles leaves of length 1-6. Returns 0.0 for empty leaves or leaves > 6 tiles.
    Converts blanks (' ') to '?' before sorting for lookup key generation.

    Args:
        rack (list): A list of characters representing the tiles left (blanks as ' ').
        verbose (bool): If True, print lookup details (optional).

    Returns:
        float: The score adjustment from the lookup table, or 0.0.
    """
    num_tiles = len(rack)

    if num_tiles == 0:
        if verbose: print("--- Evaluating Leave: Empty rack -> 0.0")
        return 0.0 # Return float
    if num_tiles > 6:
        if verbose: print(f"--- Evaluating Leave: Rack length {num_tiles} > 6 -> 0.0")
        return 0.0 # Return float

    # Create the sorted key for lookup
    rack_with_question_marks = ['?' if tile == ' ' else tile for tile in rack]
    leave_key = "".join(sorted(rack_with_question_marks))

    try:
        # Lookup the value in the global table using the key with '?'
        value = LEAVE_LOOKUP_TABLE.get(leave_key)

        if value is not None:
            # Return float directly ---
            leave_float = float(value) # Ensure it's treated as float
            if verbose: print(f"--- Evaluating Leave: Found '{leave_key}' -> {leave_float:.2f}") # Print with precision
            return leave_float
            
        else:
            # Key not found - this shouldn't happen if the table is complete for 1-6
            print(f"Warning: Leave key '{leave_key}' not found in LEAVE_LOOKUP_TABLE. Returning 0.0.")
            return 0.0 # Return float
    except Exception as e:
        # Catch potential errors during lookup (e.g., if table not loaded properly)
        print(f"Error during leave table lookup for key '{leave_key}': {e}. Returning 0.0.")
        return 0.0 # Return float
'''




##########################################################################################################
##########################################################################################################
##########################################################################################################
