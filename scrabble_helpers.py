
#Scrabble 28APR25 Cython V3




import time
import pickle
import math # Add any other necessary imports
from collections import Counter




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





# --- GADDAG Node Definition (Add this to Scrabble Game.py) ---
class GaddagNode:
    """Represents a node in the GADDAG."""
    __slots__ = ['children', 'is_terminal'] # Memory optimization

    def __init__(self):
        self.children = {}  # Dictionary mapping letter -> GaddagNode
        self.is_terminal = False # True if a path ending here is a valid word/subword

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

