#Scrabble 26APR25 Cython V1




import time
import pickle
import math # Add any other necessary imports
from collections import Counter



##########################################################################################################
##########################################################################################################
##########################################################################################################





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










# scrabble_helpers.py



'''
def is_valid_play(word_positions, tiles, is_first_play, initial_rack_size, original_tiles, rack):
    """Validate a potential play against game rules and dictionary. Includes targeted debug print."""
    print(f"--- is_valid_play DEBUG ---") # Add header
    print(f"  Input word_positions: {word_positions}")
    # print(f"  Input tiles: {tiles}") # Can be large
    print(f"  Input is_first_play: {is_first_play}")
    print(f"  Input initial_rack_size: {initial_rack_size}")
    # print(f"  Input original_tiles: {original_tiles}") # Can be large
    print(f"  Input rack (original): {rack}")

    if not word_positions:
        print("  Validation Fail: No word_positions.")
        return False, False

    newly_placed_positions_coords = set((r, c) for r, c, _ in word_positions)
    if not newly_placed_positions_coords:
        print("  Validation Fail: No newly placed tiles found.")
        return False, False # No new tiles placed

    # --- Alignment and Gap Checks ---
    rows = sorted(list(set(r for r, _, _ in word_positions))); cols = sorted(list(set(c for _, c, _ in word_positions)))
    is_horizontal = len(rows) == 1; is_vertical = len(cols) == 1
    if not (is_horizontal or is_vertical):
        print("  Validation Fail: Not aligned horizontally or vertically.")
        return False, False # Not aligned

    # Check for gaps within the main line of play
    if is_horizontal:
        r = rows[0]; min_col = min(cols); max_col = max(cols)
        temp_min_col = min_col; temp_max_col = max_col
        while temp_min_col > 0 and tiles[r][temp_min_col - 1]: temp_min_col -= 1
        while temp_max_col < GRID_SIZE - 1 and tiles[r][temp_max_col + 1]: temp_max_col += 1
        for c in range(temp_min_col, temp_max_col + 1):
            if not tiles[r][c]:
                print(f"  Validation Fail: Gap found in horizontal word at ({r},{c}).")
                return False, False # Gap found
    elif is_vertical:
        c = cols[0]; min_row = min(rows); max_row = max(rows)
        temp_min_row = min_row; temp_max_row = max_row
        while temp_min_row > 0 and tiles[temp_min_row - 1][c]: temp_min_row -= 1
        while temp_max_row < GRID_SIZE - 1 and tiles[temp_max_row + 1][c]: temp_max_row += 1
        for r in range(temp_min_row, temp_max_row + 1):
            if not tiles[r][c]:
                print(f"  Validation Fail: Gap found in vertical word at ({r},{c}).")
                return False, False # Gap found
    print("  Alignment/Gap Check: Passed.") # DEBUG

    # --- Word Validity Check (Use find_all_words_formed) ---
    all_words_details = find_all_words_formed(word_positions, tiles)
    print(f"  Words Found by find_all_words_formed: {all_words_details}") # DEBUG

    if not all_words_details and len(word_positions) > 1:
         print("  Validation Fail: Multiple tiles placed but find_all_words_formed found no words.")
         return False, False # Should not happen if connection works

    formed_word_strings = ["".join(tile[2] for tile in word_detail) for word_detail in all_words_details]
    if not formed_word_strings and len(word_positions) > 1: # If multiple tiles placed but no words formed -> invalid
         print("  Validation Fail: Multiple tiles placed but no word strings generated.")
         return False, False

    # Check each formed word string against the DAWG
    for word in formed_word_strings:
        dawg_search_result = DAWG.search(word)
        print(f"  DAWG Check: '{word}' -> {dawg_search_result}") # DEBUG
        if not dawg_search_result:
            print(f"  Validation Fail: Word '{word}' not in DAWG.")
            return False, False # Word not in dictionary
    print("  DAWG Check: Passed.") # DEBUG

    # --- Connection Rules Check ---
    if is_first_play:
        if CENTER_SQUARE not in newly_placed_positions_coords:
            print("  Validation Fail: First play does not cover center square.")
            return False, False
        print("  First Play Center Square Check: Passed.") # DEBUG
    else:
        connects = False
        if original_tiles is None:
             print("  Validation Fail: original_tiles is None, cannot check connection.")
             return False, False # Fail if original state unavailable

        for r, c in newly_placed_positions_coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and original_tiles[nr][nc]:
                    print(f"  Connection Check: Tile at ({r},{c}) connects to existing tile at ({nr},{nc}).") # DEBUG
                    connects = True; break
            if connects: break
        if not connects:
            print("  Validation Fail: Play does not connect to existing tiles.")
            return False, False
        print("  Connection Check: Passed.") # DEBUG

    # --- Bingo Check ---
    tiles_played_from_rack = len(newly_placed_positions_coords)
    is_bingo = (initial_rack_size == 7 and tiles_played_from_rack == 7)
    print(f"  Bingo Check: initial_rack_size={initial_rack_size}, tiles_played={tiles_played_from_rack} -> is_bingo={is_bingo}") # DEBUG

    # If all checks passed
    print("--- is_valid_play returning True ---") # DEBUG
    return True, is_bingo
'''




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




'''
def calculate_score(new_tiles, board, tiles, blanks):
    """Calculates the score for a play based on newly placed tiles."""
    total_score = 0; new_positions = set((r, c) for r, c, _ in new_tiles)
    words_formed_details = find_all_words_formed(new_tiles, tiles)
    for word_tiles in words_formed_details:
        word_score = 0; word_multiplier = 1
        for r, c, letter in word_tiles:
            if letter not in TILE_DISTRIBUTION: print(f"Warning: Invalid letter '{letter}' found in word during scoring at ({r},{c}). Skipping."); continue
            is_blank = (r, c) in blanks; letter_value = 0 if is_blank else TILE_DISTRIBUTION[letter][1]; letter_multiplier = 1
            if (r, c) in new_positions:
                square_color = board[r][c]
                if square_color == LIGHT_BLUE: letter_multiplier = 2 # DL
                elif square_color == BLUE: letter_multiplier = 3 # TL
                elif square_color == PINK: word_multiplier *= 2 # DW (Center is also Pink)
                elif square_color == RED: word_multiplier *= 3 # TW
            word_score += letter_value * letter_multiplier
        total_score += word_score * word_multiplier
    if len(new_tiles) == 7: total_score += 50 # Bingo bonus
    return total_score
'''


    


##########################################################################################################
##########################################################################################################
##########################################################################################################




'''
def find_all_words_formed(new_tiles, tiles):
    """Finds all words (main and cross) formed by a play."""
    words = [];
    if not new_tiles: return words
    new_positions_set = set((r, c) for r, c, _ in new_tiles)
    main_word_tiles, orientation = find_main_word(new_tiles, tiles)
    if main_word_tiles:
        words.append(main_word_tiles)
        for tile in new_tiles:
            if (tile[0], tile[1]) in new_positions_set:
                cross_word = find_cross_word(tile, tiles, orientation)
                if cross_word: words.append(cross_word)
    elif len(new_tiles) == 1: # Single tile placement check
        tile = new_tiles[0]
        cross_h = find_cross_word(tile, tiles, "vertical");   # Check H
        if cross_h: words.append(cross_h)
        cross_v = find_cross_word(tile, tiles, "horizontal"); # Check V
        if cross_v: words.append(cross_v)
    unique_word_tile_lists = []; seen_signatures = set()
    for word_tile_list in words:
        signature = tuple(sorted((r, c, l) for r, c, l in word_tile_list))
        if signature not in seen_signatures: unique_word_tile_lists.append(word_tile_list); seen_signatures.add(signature)
    return unique_word_tile_lists
'''





##########################################################################################################
##########################################################################################################
##########################################################################################################




'''
def find_main_word(new_tiles, tiles):
    """Finds the primary word formed by newly placed tiles."""
    if not new_tiles: return [], None
    rows = set(r for r, c, _ in new_tiles); cols = set(c for r, c, _ in new_tiles)
    if len(rows) == 1: # Potential horizontal word
        orientation = "horizontal"; row = rows.pop(); min_col = min(c for r, c, _ in new_tiles if r == row); max_col = max(c for r, c, _ in new_tiles if r == row)
        while min_col > 0 and tiles[row][min_col - 1]: min_col -= 1
        while max_col < GRID_SIZE - 1 and tiles[row][max_col + 1]: max_col += 1
        main_word = [(row, c, tiles[row][c]) for c in range(min_col, max_col + 1) if tiles[row][c]]
        return (main_word, orientation) if len(main_word) > 1 else ([], None)
    elif len(cols) == 1: # Potential vertical word
        orientation = "vertical"; col = cols.pop(); min_row = min(r for r, c, _ in new_tiles if c == col); max_row = max(r for r, c, _ in new_tiles if c == col)
        while min_row > 0 and tiles[min_row - 1][col]: min_row -= 1
        while max_row < GRID_SIZE - 1 and tiles[max_row + 1][col]: max_row += 1
        main_word = [(r, col, tiles[r][col]) for r in range(min_row, max_row + 1) if tiles[r][col]]
        return (main_word, orientation) if len(main_word) > 1 else ([], None)
    else: return [], None
'''






##########################################################################################################
##########################################################################################################
##########################################################################################################





'''
def find_cross_word(tile, tiles, main_orientation):
    """Finds a cross word formed by a single tile perpendicular to the main word."""
    r, c, _ = tile; cross_word = []
    if main_orientation == "horizontal": # Check vertically
        min_row = r;
        while min_row > 0 and tiles[min_row - 1][c]: min_row -= 1
        max_row = r;
        while max_row < GRID_SIZE - 1 and tiles[max_row + 1][c]: max_row += 1
        if max_row > min_row: cross_word = [(rr, c, tiles[rr][c]) for rr in range(min_row, max_row + 1) if tiles[rr][c]]
    elif main_orientation == "vertical": # Check horizontally
        min_col = c;
        while min_col > 0 and tiles[r][min_col - 1]: min_col -= 1
        max_col = c;
        while max_col < GRID_SIZE - 1 and tiles[r][max_col + 1]: max_col += 1
        if max_col > min_col: cross_word = [(r, cc, tiles[r][cc]) for cc in range(min_col, max_col + 1) if tiles[r][cc]]
    return cross_word if len(cross_word) > 1 else []
'''

    



##########################################################################################################
##########################################################################################################
##########################################################################################################




def get_coord(start, direction):
    """
    Generate the coordinate string in GCG format.
    Accepts direction as "right", "down", 'H', or 'V'.
    Horizontal: RowNumberColumnLetter (e.g., 8H)
    Vertical:   ColumnLetterRowNumber (e.g., H8)
    """
    row, col = start
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        # Handle invalid start coordinates if necessary, though ideally prevented earlier
        return "???" # Or raise an error

    # Map 'H'/'V' to 'right'/'down' logic internally
    if direction == "right" or direction == 'H': # Horizontal
        # Format: RowNumberColumnLetter
        return f"{row + 1}{LETTERS[col]}"
    elif direction == "down" or direction == 'V': # Vertical
        # Format: ColumnLetterRowNumber
        return f"{LETTERS[col]}{row + 1}"
    else:
        # Handle truly unexpected direction values
        print(f"Warning: Unexpected direction '{direction}' in get_coord.")
        return "???"



##########################################################################################################
##########################################################################################################
##########################################################################################################



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




##########################################################################################################
##########################################################################################################
##########################################################################################################




LEAVE_LOOKUP_TABLE = {} # Global dictionary
try:
    print("Loading leave evaluation table...")
    load_start = time.time()
    with open("NWL23-leaves.pkl", 'rb') as f_load:
        LEAVE_LOOKUP_TABLE = pickle.load(f_load)
    print(f"Leave table loaded with {len(LEAVE_LOOKUP_TABLE)} entries in {time.time() - load_start:.2f} seconds.")
except FileNotFoundError:
    print("Warning: leave_table.pkl not found. Leave evaluation might be slower or inaccurate.")
    # Optionally, add fallback to load from CSV here if desired
except Exception as e:
    print(f"Error loading leave_table.pkl: {e}")



##########################################################################################################
##########################################################################################################
##########################################################################################################




# --- Trie/DAWG Setup ---
class TrieNode: # Keeping simple Trie for now, can replace with minimized DAWG later
    def __init__(self):
        self.children = {}
        self.is_end = False

class Dawg: # Renaming our existing Trie to Dawg for conceptual clarity
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        """Check if a word exists in the DAWG."""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

# Load the DAWG with words from "All Words 2023.txt"
DAWG = Dawg()
try:
    # Ensure the path is correct or adjust as needed
    with open("All Words 2023.txt", "r") as f:
        for word in (w.strip().upper() for w in f):
            if len(word) > 1: # Optionally filter out single-letter words if not needed
                DAWG.insert(word)
    print("DAWG loaded successfully.")
except FileNotFoundError:
    print("Error: 'All Words 2023.txt' not found. Word validation will not work.")





##########################################################################################################
##########################################################################################################
##########################################################################################################
