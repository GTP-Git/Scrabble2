
#python
#Scrabble 26APR25 Cython V3


# Part 1


import pygame
import random
import math
import sys
import time
import pickle
import os
import datetime
import itertools
from itertools import permutations, product
from collections import Counter
import copy
import threading
import array
import numpy as np
#from gaddag_traverse_cy import _gaddag_traverse

import scrabble_helpers


from scrabble_helpers import (
    get_coord, evaluate_leave,
    get_anchor_points,
    DAWG, Gaddag, GaddagNode,
    GRID_SIZE, CENTER_SQUARE, TILE_DISTRIBUTION,
    RED, PINK, BLUE, LIGHT_BLUE, LETTERS
)


try:
    from gaddag_cython import _gaddag_traverse as _gaddag_traverse_cython
    from gaddag_cython import calculate_score as calculate_score_cython
    from gaddag_cython import is_valid_play as is_valid_play_cython
    from gaddag_cython import find_all_words_formed as find_all_words_formed_cython
    print("--- SUCCESS: Imported Cython functions. ---")
    USE_CYTHON_GADDAG = True
    # --- Add explicit check ---
    print(f"    _gaddag_traverse_cython: {repr(_gaddag_traverse_cython)}")
    print(f"    calculate_score_cython: {repr(calculate_score_cython)}")
    print(f"    is_valid_play_cython: {repr(is_valid_play_cython)}")
    print(f"    find_all_words_formed_cython: {repr(find_all_words_formed_cython)}")
    # --- End explicit check ---

except ImportError as e:
    print(f"--- FAILURE: Cython import failed: {e} ---")
    print("!!! CRITICAL ERROR: Cannot proceed without core Cython functions or fallbacks. !!!")
    # Define dummy functions to prevent immediate crash, but indicate failure
    def _gaddag_traverse_dummy(*args, **kwargs): print("DUMMY _gaddag_traverse called"); return None
    def calculate_score_dummy(*args, **kwargs): print("DUMMY calculate_score called"); return 0
    def is_valid_play_dummy(*args, **kwargs): print("DUMMY is_valid_play called"); return False, False
    def find_all_words_formed_dummy(*args, **kwargs): print("DUMMY find_all_words_formed called"); return []
    _gaddag_traverse_cython = _gaddag_traverse_dummy
    calculate_score_cython = calculate_score_dummy
    is_valid_play_cython = is_valid_play_dummy
    find_all_words_formed_cython = find_all_words_formed_dummy
    USE_CYTHON_GADDAG = False
    # Optionally, exit here if fallbacks are unacceptable
    # sys.exit("Cython module failed to load.")
# --- END CYTHON IMPORT ---

print(f"DEBUG: is_valid_play_cython is assigned to: {is_valid_play_cython}")




try:
    import pyperclip
    pyperclip_available = True
    print("Pyperclip library loaded successfully for paste functionality.")
except ImportError:
    pyperclip = None # Set to None if import fails
    pyperclip_available = False
    print("Warning: Pyperclip library not found. Paste functionality (Ctrl+V/Cmd+V) will be disabled.")
    print("         To enable paste, install it using: pip install pyperclip")





# Initialize Pygame
pygame.init()
main_called = False # Flag to track if main() has been entered

# Constants
GADDAG_STRUCTURE = None

SQUARE_SIZE = 40
BOARD_SIZE = GRID_SIZE * SQUARE_SIZE
FONT_SIZE = SQUARE_SIZE // 2
TILE_WIDTH = 35
TILE_HEIGHT = 35
TILE_GAP = 5
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 30
BUTTON_GAP = 10
OPTIONS_WIDTH = 100
OPTIONS_HEIGHT = 30
DROPDOWN_HEIGHT = 90
OPTIONS_Y = BOARD_SIZE + 60  # Adjusted from +40 to align with shifted UI
DOUBLE_CLICK_TIME = 300
SECOND_CLICK_TIME = 500  # Time window for second click
ALL_WORDS_DIALOG_WIDTH = 750
ALL_WORDS_DIALOG_HEIGHT = 600
SCROLL_SPEED = 20

STATS_LABEL_X_OFFSET = 10
STATS_P1_VAL_X_OFFSET = 160 # Increased from 120
STATS_P2_VAL_X_OFFSET = 270 # Increased from 230



LEAVE_LOOKUP_TABLE = {}

EXCHANGE_PREFERENCE_THRESHOLD = 5.0
LOW_SCORE_THRESHOLD = 10
POOL_QUALITY_FACTOR = 1.5
POOL_TILE_VALUES = {' ': 5, 'S': 3, 'Z': 1, 'X': 1, 'Q': -3, 'J': -2, 'V': -2, 'W': -2, 'U': -2, 'I': -1}

# --- Simulation Defaults ---
DEFAULT_PLY_DEPTH = 2 # Not currently used by run_ai_simulation, but good practice
DEFAULT_AI_CANDIDATES = 10
DEFAULT_OPPONENT_SIMULATIONS = 50
DEFAULT_POST_SIM_CANDIDATES = 10

# Dialog dimensions for game over dialog
DIALOG_WIDTH = 480
DIALOG_HEIGHT = 250
LETTERS = "ABCDEFGHIJKLMNO"  # For column labels

# Get screen resolution
display_info = pygame.display.Info()
# Set window size to fit within screen resolution (with padding for borders)
WINDOW_WIDTH = min(1400, display_info.current_w - 50)  # 50 pixels padding
WINDOW_HEIGHT = min(900, display_info.current_h - 50)  # 50 pixels padding

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HIGHLIGHT_BLUE = (191, 238, 255)
YELLOW = (255, 255, 153)
GRAY = (200, 200, 200)
GREEN = (144, 238, 144)
DARK_GREEN = (100, 200, 100)
PALE_YELLOW = (255, 255, 200)
BUTTON_COLOR = (180, 180, 180)
BUTTON_HOVER = (220, 220, 220)
TURN_INDICATOR_COLOR = (255, 215, 0)
DIALOG_COLOR = (150, 150, 150)
DROPDOWN_COLOR = (200, 200, 200)
SELECTED_TILE_COLOR = (255, 165, 0)
GRAYED_OUT_COLOR = (100, 100, 100)
ARROW_COLOR = (0, 0, 255)
HINT_NORMAL_COLOR = (220, 220, 220)
HINT_SELECTED_COLOR = (180, 180, 255)

# Setup display and font
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Scrabble Game")
font = pygame.font.SysFont("Arial", FONT_SIZE, bold=True)
ui_font = pygame.font.SysFont("Arial", 20)
button_font = pygame.font.SysFont("Arial", 16)
tile_count_font = pygame.font.SysFont("Arial", 14)
dialog_font = pygame.font.SysFont("Arial", 24)




TILE_LETTER_CACHE = {
    'regular': {}, # For standard tiles (black text)
    'blank': {}    # For blank tiles (white text)
}

print("Pre-rendering tile letters...")
# Pre-render regular tile letters (A-Z)
for i in range(26):
    letter = chr(ord('A') + i)
    # Render black text (True for anti-aliasing)
    text_surf = font.render(letter, True, BLACK)
    TILE_LETTER_CACHE['regular'][letter] = text_surf

# Pre-render blank tile symbol (?) - white text
blank_surf = font.render('?', True, WHITE)
TILE_LETTER_CACHE['blank']['?'] = blank_surf
print("Tile letter pre-rendering complete.")



SCORE_CACHE = {
    0: {'score': None, 'surface': None},
    1: {'score': None, 'surface': None}
}



# Global bag - initialized properly in main() or practice setup
bag = []

POWER_TILES = {'J', 'Q', 'X', 'Z'}

# Global game state variables (will be initialized/reset in main)
board = None
tiles = None
racks = None
blanks = None
scores = None
turn = 1
first_play = True
game_mode = None
is_ai = None
practice_mode = None # Added to track practice modes like "eight_letter", "power_tiles"
move_history = []
replay_mode = False
current_replay_turn = 0
last_word = ""
last_score = 0
last_start = None
last_direction = None
gaddag_loading_status = 'idle' # Tracks status: 'idle', 'loading', 'loaded', 'error'
gaddag_load_thread = None # Holds the thread object

is_solving_endgame = False # Flag to indicate AI is in endgame calculation
endgame_start_time = 0 # To track duration if needed



def _load_gaddag_background():
    """Loads the GADDAG structure from pickle in a background thread."""
    global GADDAG_STRUCTURE, gaddag_loading_status
    try:
        print("Background Thread: Attempting to load GADDAG structure from gaddag.pkl...")
        load_start = time.time()
        with open("gaddag.pkl", 'rb') as f_load:
            loaded_gaddag = pickle.load(f_load) # Load into temporary variable first
        GADDAG_STRUCTURE = loaded_gaddag # Assign to global only after successful load
        gaddag_loading_status = 'loaded'
        print(f"Background Thread: GADDAG loaded successfully in {time.time() - load_start:.2f} seconds. Status: {gaddag_loading_status}")
    except FileNotFoundError:
        print("\n--- BACKGROUND LOAD ERROR: gaddag.pkl not found. ---")
        print("Ensure 'gaddag.pkl' exists. AI features will be disabled.")
        GADDAG_STRUCTURE = None
        gaddag_loading_status = 'error'
    except Exception as e:
        print(f"\n--- BACKGROUND LOAD FATAL ERROR: {e} ---")
        GADDAG_STRUCTURE = None
        gaddag_loading_status = 'error'
        # Optionally: sys.exit() or raise an exception if this is critical even in background



def draw_loading_indicator(scoreboard_x, scoreboard_y, scoreboard_width):
    """
    Draws a message indicating that the GADDAG is loading,
    positioned above the scoreboard area.
    """
    global gaddag_loading_status, screen, ui_font, RED # Ensure necessary globals are accessible

    if gaddag_loading_status == 'loading':
        loading_text = "Loading AI Data..."
        loading_surf = ui_font.render(loading_text, True, RED) # Use UI font, red color

        # Calculate position centered above the scoreboard
        target_center_x = scoreboard_x + scoreboard_width // 2
        # Position the BOTTOM of the text slightly above the scoreboard's top
        target_bottom_y = scoreboard_y - 10 # 10 pixels padding above scoreboard

        # Ensure the text doesn't go off the top edge (e.g., y=5 minimum)
        target_top_y = max(5, target_bottom_y - loading_surf.get_height())

        # Use the calculated top position and center x
        loading_rect = loading_surf.get_rect(centerx=target_center_x, top=target_top_y)

        # Optional: Add a semi-transparent background for better visibility
        bg_rect = loading_rect.inflate(20, 10) # Add padding
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA) # Surface with alpha
        bg_surf.fill((200, 200, 200, 180)) # Semi-transparent gray
        screen.blit(bg_surf, bg_rect)

        # Draw the text on top
        screen.blit(loading_surf, loading_rect)






def draw_checkbox(screen, x, y, checked):
    pygame.draw.rect(screen, BLACK, (x, y, 20, 20), 1)
    if checked:
        pygame.draw.line(screen, BLACK, (x+2, y+2), (x+18, y+18), 2)
        pygame.draw.line(screen, BLACK, (x+18, y+2), (x+2, y+18), 2)


# --- GCG Handling ---
def save_game_to_gcg(player_names, move_history, initial_racks, final_scores):
    """Save the game to GCG format using move_history directly."""
    gcg_lines = [
        "#",
        f"#player1 {player_names[0]}",
        f"#player2 {player_names[1]}"
    ]
    cumulative_scores = [0, 0]

    for move in move_history:
        player = move['player'] - 1  # 0-based index
        rack = ''.join(sorted(tile if tile != ' ' else '?' for tile in move['rack']))

        if move['move_type'] == 'place':
            # Use the stored full word with blanks
            word_with_blanks = move.get('word_with_blanks', move.get('word','').upper()) # Use getter for safety
            score = move['score']
            cumulative_scores[player] += score
            gcg_lines.append(
                f">{player_names[player]}: {rack} {move['coord']} {word_with_blanks} +{score} {cumulative_scores[player]}"
            )
        elif move['move_type'] == 'exchange':
            exchanged = ''.join(sorted(tile if tile != ' ' else '?' for tile in move.get('exchanged_tiles',[]))) # Use getter
            gcg_lines.append(
                f">{player_names[player]}: {rack} ({exchanged}) +0 {cumulative_scores[player]}"
            )
        elif move['move_type'] == 'pass':
            gcg_lines.append(
                f">{player_names[player]}: {rack} -- +0 {cumulative_scores[player]}"
            )

    gcg_lines.append(f"Final score: {player_names[0]} {final_scores[0]}, {player_names[1]} {final_scores[1]}")

    return '\n'.join(gcg_lines)



def load_game_from_gcg(filename):
    """Load a game from a GCG file, returning data to enter replay mode."""
    move_history = []
    player_names = ["Player1", "Player2"]
    # Initialize final_scores here ---
    final_scores = [0, 0]
    
    line_num = 0

    try:
        with open(filename, "r") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line or line.startswith("#"): # Skip empty/comments
                    if line.startswith("#player1"):
                        try: player_names[0] = line.split(maxsplit=1)[1]
                        except IndexError: print(f"GCG Load Warning (Line {line_num}): Malformed #player1 line.")
                    elif line.startswith("#player2"):
                        try: player_names[1] = line.split(maxsplit=1)[1]
                        except IndexError: print(f"GCG Load Warning (Line {line_num}): Malformed #player2 line.")
                    continue

                if line.startswith("Final score:"):
                    try:
                        parts = line.split()
                        if len(parts) < 5: # Need at least "Final score: N1 S1, N2 S2"
                            raise ValueError("Too few parts for Final score line")

                        # More robust parsing from the end ---
                        score2_str = parts[-1]
                        score1_str = ""
                        # Find the part before score2 that ends with a comma
                        for i in range(len(parts) - 2, 0, -1):
                            if parts[i].endswith(','):
                                score1_str = parts[i].strip(',')
                                break # Found score1 part

                        if score1_str and score2_str.isdigit() and score1_str.isdigit():
                            # Successfully found both scores
                            final_scores[0] = int(score1_str)
                            final_scores[1] = int(score2_str)
                            print(f"GCG Load Info (Line {line_num}): Parsed final scores: {final_scores}")
                        else:
                            # Fallback or less reliable method if needed, but better to raise error
                            raise ValueError("Could not reliably parse final scores from line parts.")
                        

                    except (IndexError, ValueError) as e:
                        print(f"GCG Load Warning (Line {line_num}): Error parsing final score line: '{line}'. Error: {e}. Using [0, 0].")
                        # Keep final_scores as [0, 0] initialized earlier
                    continue # Move to next line after processing Final score

                if line.startswith(">"):
                    try:
                        parts = line.split() # Split by whitespace
                        if len(parts) < 5: # Minimum parts: >Name: Rack -- +Score CumScore
                            raise ValueError(f"Insufficient parts on move line ({len(parts)})")

                        # --- Robust Parsing Logic ---
                        # 1. Identify parts from the end
                        cumulative_score_str = parts[-1]
                        score_str = parts[-2]
                        # Part before score could be Word, (Exchange), or --
                        third_last_part = parts[-3]

                        # 2. Safely parse scores first
                        try: score = int(score_str[1:]) # Remove initial '+' or '-'
                        except (ValueError, IndexError): raise ValueError(f"Invalid score format '{score_str}'")
                        try: cumulative_score = int(cumulative_score_str)
                        except ValueError: raise ValueError(f"Invalid cumulative score format '{cumulative_score_str}'")

                        # 3. Determine Move Type based on third_last_part
                        move_type = None
                        player_name_parts = []
                        rack_str = ""
                        exchanged_list = []
                        position_str = ""
                        word_played_gcg = ""

                        if third_last_part == "--": # Pass
                            move_type = 'pass'
                            # Structure: > Name(s) : Rack -- +Score CumScore
                            if len(parts) < 5: raise ValueError("Incorrect part count for Pass")
                            rack_str = parts[-4] # Part before -- is Rack
                            player_name_parts = parts[:-4] # All parts before Rack
                        elif third_last_part.startswith("(") and third_last_part.endswith(")"): # Exchange
                            move_type = 'exchange'
                            # Structure: > Name(s) : Rack (Exch) +Score CumScore
                            if len(parts) < 5: raise ValueError("Incorrect part count for Exchange")
                            rack_str = parts[-4] # Part before (Exch) is Rack
                            player_name_parts = parts[:-4] # All parts before Rack
                            exchanged_gcg = third_last_part[1:-1]
                            exchanged_list = [(' ' if char == '?' else char.upper()) for char in exchanged_gcg]
                        else: # Place
                            move_type = 'place'
                            # Structure: > Name(s) : Rack Coord Word +Score CumScore
                            if len(parts) < 6: raise ValueError("Incorrect part count for Place")
                            word_played_gcg = third_last_part # Part before score is Word
                            position_str = parts[-4] # Part before Word is Coord
                            rack_str = parts[-5] # Part before Coord is Rack
                            player_name_parts = parts[:-5] # All parts before Rack

                        # 4. Reconstruct Player Name
                        if not player_name_parts or not player_name_parts[0].startswith(">") or not player_name_parts[-1].endswith(":"):
                            raise ValueError(f"Could not reconstruct player name from parts: {player_name_parts}")
                        player_name_full = " ".join(player_name_parts)
                        player_name = player_name_full[1:-1] # Remove '>' and ':'
                        player = 1 if player_name == player_names[0] else 2
                        # --- End Robust Parsing Logic ---

                        # Append to move_history based on type
                        if move_type == 'pass':
                            move_history.append({
                                'player': player, 'move_type': 'pass', 'score': score,
                                'word': '', 'coord': ''
                            })
                        elif move_type == 'exchange':
                            move_history.append({
                                'player': player, 'move_type': 'exchange', 'exchanged_tiles': exchanged_list,
                                'score': score, 'word': '', 'coord': ''
                            })
                        elif move_type == 'place':
                            # Parse coordinate
                            coord_parse_result = parse_coord(position_str)
                            if coord_parse_result is None or coord_parse_result[0] is None:
                                raise ValueError(f"Invalid coordinate format '{position_str}'")
                            (row, col), direction = coord_parse_result

                            # Reconstruct positions and blanks from the GCG word
                            positions = []; blanks = set()
                            current_r_gcg, current_c_gcg = row, col
                            for i, letter_gcg in enumerate(word_played_gcg):
                                r_place = current_r_gcg if direction == "right" else current_r_gcg + i
                                c_place = current_c_gcg + i if direction == "right" else current_c_gcg
                                if not (0 <= r_place < GRID_SIZE and 0 <= c_place < GRID_SIZE):
                                    raise ValueError(f"Word placement out of bounds: '{word_played_gcg}' at {position_str}")
                                letter_upper = letter_gcg.upper()
                                positions.append((r_place, c_place, letter_upper))
                                if letter_gcg.islower(): blanks.add((r_place, c_place))

                            move_history.append({
                                'player': player, 'move_type': 'place', 'positions': positions,
                                'blanks': blanks, 'score': score, 'word': word_played_gcg.upper(),
                                'start': (row, col), 'direction': direction, 'coord': position_str
                            })

                    except Exception as e:
                        print(f"GCG Load Error (Line {line_num}): Failed to parse move line.")
                        print(f"  Line content: '{line}'")
                        print(f"  Error details: {type(e).__name__}: {e}")
                        raise ValueError(f"Error parsing GCG line {line_num}: {e}") from e

                else:
                    print(f"GCG Load Warning (Line {line_num}): Skipping unrecognized line format: '{line}'")

    except FileNotFoundError:
        print(f"GCG Load Error: File not found '{filename}'")
        raise
    except Exception as e:
        print(f"GCG Load Error: An unexpected error occurred reading file '{filename}' near line {line_num}.")
        print(f"  Error details: {type(e).__name__}: {e}")
        raise

    # Return the final scores that were parsed (or [0,0] if parsing failed/line missing)
    return player_names, move_history, final_scores



# New helper function to create a standard Scrabble bag
def create_standard_bag():
    """Creates and returns a standard Scrabble tile bag list."""
    bag = [letter for letter, (count, _) in TILE_DISTRIBUTION.items() for _ in range(count)]
    return bag



def simulate_game_up_to(target_turn_idx, move_history_loaded, initial_shuffled_bag):
    """
    Simulates a game turn-by-turn up to a target index using loaded move history
    (which lacks 'drawn' info) and an initial shuffled bag state. Handles initial draws.
    Removed Replay Sim Warnings for performance.
    Corrected rack size calculation during simulation.

    Args:
        target_turn_idx (int): The 0-based index of the turn *after* which the state is needed
                               (e.g., 0 for initial state, 1 for state after move 0).
        move_history_loaded (list): The move history loaded from GCG (lacks 'drawn').
        initial_shuffled_bag (list): A list representing the shuffled bag at the very start of the game.

    Returns:
        tuple: (tiles_state, blanks_state, scores_state, racks_state) representing the
               game state after target_turn_idx-1 moves have been applied.
               Racks are sorted alphabetically before returning for display.
    """
    # Initialize game state variables
    tiles_state = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    blanks_state = set()
    scores_state = [0, 0]
    racks_state = [[], []]
    # IMPORTANT: Use a copy of the initial bag for the simulation run to avoid modifying the original
    bag_state = initial_shuffled_bag[:]

    # --- Simulate Drawing Initial Racks ---
    try:
        for _ in range(7):
            if bag_state: racks_state[0].append(bag_state.pop())
            else: raise IndexError("Bag empty during initial draw for P1")
        for _ in range(7):
            if bag_state: racks_state[1].append(bag_state.pop())
            else: raise IndexError("Bag empty during initial draw for P2")
    except IndexError as e:
        print(f"Replay Simulation Error: {e}")
        return tiles_state, blanks_state, scores_state, racks_state

    # Apply moves sequentially up to the target turn index (0 to target_turn_idx-1)
    for i in range(target_turn_idx):
        if i >= len(move_history_loaded):
            # print(f"Replay Sim Info: Reached end of available history at move index {i} before target index {target_turn_idx}.") # Commented out
            break

        move = move_history_loaded[i]
        player_idx = move.get('player')
        if player_idx not in [1, 2]:
            continue
        player_idx -= 1

        rack_after_move = racks_state[player_idx][:]
        # Counter for actual removals ---
        actual_tiles_removed_from_sim_rack = 0
        

        move_type = move.get('move_type')

        if move_type == 'place':
            positions = move.get('positions', [])
            blanks_in_move = move.get('blanks', set())

            for r, c, letter in positions:
                 is_newly_placed_sim = (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and not tiles_state[r][c])

                 if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                     tiles_state[r][c] = letter
                     if (r, c) in blanks_in_move:
                         blanks_state.add((r, c))
                 else:
                     continue

                 if is_newly_placed_sim:
                     # tiles_removed_count += 1 # Original counter - no longer needed for drawing
                     if (r,c) in blanks_in_move:
                         if ' ' in rack_after_move:
                             rack_after_move.remove(' ')
                             # Increment actual removal counter ---
                             actual_tiles_removed_from_sim_rack += 1
                             
                         else:
                             pass # Warning already commented out
                     else:
                         if letter in rack_after_move:
                             rack_after_move.remove(letter)
                             # Increment actual removal counter ---
                             actual_tiles_removed_from_sim_rack += 1
                             
                         else:
                             pass # Warning already commented out

            scores_state[player_idx] += move.get('score', 0)

        elif move_type == 'exchange':
            exchanged_gcg = move.get('exchanged_tiles', [])
            for tile_gcg in exchanged_gcg:
                 tile_to_remove = ' ' if tile_gcg == '?' else tile_gcg.upper()
                 if tile_to_remove in rack_after_move:
                      rack_after_move.remove(tile_to_remove)
                      # Increment actual removal counter ---
                      actual_tiles_removed_from_sim_rack += 1
                      
                 else:
                      pass # Warning already commented out
            # Score doesn't change

        elif move_type == 'pass':
            actual_tiles_removed_from_sim_rack = 0 # No tiles removed
            # Score doesn't change

        else:
             continue # Skip draw calculation if move type is unknown

        # --- Simulate Drawing New Tiles ---
        # Use actual removal count for drawing ---
        num_to_draw = actual_tiles_removed_from_sim_rack
        
        drawn_simulated = []
        for _ in range(num_to_draw):
             if bag_state:
                 drawn_simulated.append(bag_state.pop())
             else:
                 # print(f"Replay Sim Info: Simulated bag ran out while drawing for move {i}.") # Commented out
                 break
        rack_after_move.extend(drawn_simulated)
        racks_state[player_idx] = rack_after_move

    # Sort final racks alphabetically before returning
    for rack in racks_state:
        rack.sort()

    return tiles_state, blanks_state, scores_state, racks_state







# --- Mode Selection Constants ---
MODE_HVH = "Human vs Human"
MODE_HVA = "Human vs AI"
MODE_AVA = "AI vs AI"

# --- Replay Button Positions ---
REPLAY_BUTTON_Y = OPTIONS_Y + OPTIONS_HEIGHT + 10
REPLAY_BUTTON_WIDTH = 50
REPLAY_BUTTON_HEIGHT = 30
REPLAY_BUTTON_GAP = 10
replay_start_rect = pygame.Rect(10, REPLAY_BUTTON_Y, REPLAY_BUTTON_WIDTH, REPLAY_BUTTON_HEIGHT)
replay_prev_rect = pygame.Rect(10 + REPLAY_BUTTON_WIDTH + REPLAY_BUTTON_GAP, REPLAY_BUTTON_Y, REPLAY_BUTTON_WIDTH, REPLAY_BUTTON_HEIGHT)
replay_next_rect = pygame.Rect(10 + 2 * (REPLAY_BUTTON_WIDTH + REPLAY_BUTTON_GAP), REPLAY_BUTTON_Y, REPLAY_BUTTON_WIDTH, REPLAY_BUTTON_HEIGHT)
replay_end_rect = pygame.Rect(10 + 3 * (REPLAY_BUTTON_WIDTH + REPLAY_BUTTON_GAP), REPLAY_BUTTON_Y, REPLAY_BUTTON_WIDTH, REPLAY_BUTTON_HEIGHT)








def parse_coord(coord):
    """Parse a GCG coordinate (e.g., '8H' or 'H8') into (row, col) and direction."""
    if not coord: return None, None # Handle empty coord
    if coord[0].isalpha():  # Vertical: e.g., H8 or H10
        col_char = coord[0]
        row_str = coord[1:]
        if col_char not in LETTERS or not row_str.isdigit(): return None, None
        col = LETTERS.index(col_char)
        row = int(row_str) - 1
        direction = "down"
    else:  # Horizontal: e.g., 8H or 10A
        i = 0
        while i < len(coord) and coord[i].isdigit(): i += 1
        row_str = coord[:i]
        col_char = coord[i:]
        if not row_str.isdigit() or len(col_char) != 1 or col_char not in LETTERS: return None, None
        row = int(row_str) - 1
        col = LETTERS.index(col_char)
        direction = "right"

    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE): return None, None # Bounds check
    return (row, col), direction

# --- Board Creation ---
def create_board():
    """Initialize the Scrabble board with special squares and labels."""
    board = [[WHITE for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    labels = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    tiles = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    tw = [(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)]
    dw = [(1, 1), (2, 2), (3, 3), (4, 4), (10, 10), (11, 11), (12, 12), (13, 13),
          (1, 13), (2, 12), (3, 11), (4, 10), (10, 4), (11, 3), (12, 2), (13, 1)]
    tl = [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13), (9, 1), (9, 5),
          (9, 9), (9, 13), (13, 5), (13, 9)]
    dl = [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14), (6, 2),
          (6, 6), (6, 8), (6, 12), (7, 3), (7, 11), (8, 2), (8, 6), (8, 8),
          (8, 12), (11, 0), (11, 7), (11, 14), (12, 6), (12, 8), (14, 3), (14, 11)]

    for r, c in tw: board[r][c] = RED
    for r, c in dw: board[r][c] = PINK
    for r, c in tl: board[r][c] = BLUE
    for r, c in dl: board[r][c] = LIGHT_BLUE
    board[7][7] = PINK # Center square

    return board, labels, tiles





# --- Drawing Functions ---


# Function to Replace: draw_rack
# REASON: Add caching for player score text surfaces.

def draw_rack(player, rack, scores, turn, player_names, dragged_tile=None, drag_pos=None, display_scores=None):
    """Draw a player's rack with tiles, scores, and buttons. Uses cached surfaces for letters and scores."""
    global practice_mode, TILE_LETTER_CACHE, SCORE_CACHE # Add SCORE_CACHE global
    if not rack: return None, None
    if display_scores is None: display_scores = scores
    if practice_mode == "eight_letter" and player == 2: return None, None # Don't draw P2 rack

    rack_width = 7 * (TILE_WIDTH + TILE_GAP) - TILE_GAP
    replay_area_end_x = 10 + 4 * (REPLAY_BUTTON_WIDTH + REPLAY_BUTTON_GAP) # 10 + 4*(50+10) = 250
    min_rack_start_x = replay_area_end_x + BUTTON_GAP + 20 # 250 + 10 + 20 = 280
    start_x = max(min_rack_start_x, (BOARD_SIZE - rack_width) // 2)
    rack_y = BOARD_SIZE + 80 if player == 1 else BOARD_SIZE + 150

    if turn == player and (practice_mode != "eight_letter" or player == 1): # Draw turn indicator
        center_x = start_x - 20; center_y = rack_y + TILE_HEIGHT // 2; radius = 10; points = []
        for i in range(10): angle = i * math.pi / 5; r = radius if i % 2 == 0 else radius / 2; x = center_x + r * math.cos(angle); y = center_y + r * math.sin(angle); points.append((x, y))
        pygame.draw.polygon(screen, TURN_INDICATOR_COLOR, points)

    for i, tile in enumerate(rack): # Draw tiles (using letter cache)
        tile_x = start_x + i * (TILE_WIDTH + TILE_GAP)
        tile_rect = pygame.Rect(tile_x, rack_y, TILE_WIDTH, TILE_HEIGHT)
        is_this_tile_dragged = (dragged_tile is not None and
                                dragged_tile[0] == player and
                                dragged_tile[1] == i)
        if is_this_tile_dragged:
             continue
        if tile == ' ':
            center = tile_rect.center; radius = TILE_WIDTH // 2 - 2
            pygame.draw.circle(screen, BLACK, center, radius)
            text_surf = TILE_LETTER_CACHE['blank'].get('?')
            if text_surf:
                text_rect = text_surf.get_rect(center=center)
                screen.blit(text_surf, text_rect)
        else:
            pygame.draw.rect(screen, GREEN, tile_rect)
            text_surf = TILE_LETTER_CACHE['regular'].get(tile)
            if text_surf:
                text_rect = text_surf.get_rect(center=tile_rect.center)
                screen.blit(text_surf, text_rect)

    # Draw buttons (unchanged)
    button_x = start_x + rack_width + BUTTON_GAP
    alpha_button_rect = pygame.draw.rect(screen, BUTTON_COLOR, (button_x, rack_y, BUTTON_WIDTH, BUTTON_HEIGHT))
    rand_button_rect = pygame.draw.rect(screen, BUTTON_COLOR, (button_x + BUTTON_WIDTH + BUTTON_GAP, rack_y, BUTTON_WIDTH, BUTTON_HEIGHT))
    alpha_text = button_font.render("Alphabetize", True, BLACK); alpha_rect = alpha_text.get_rect(center=alpha_button_rect.center); screen.blit(alpha_text, alpha_rect)
    rand_text = button_font.render("Randomize", True, BLACK); rand_rect = rand_text.get_rect(center=rand_button_rect.center); screen.blit(rand_text, rand_rect)

    # --- Draw score (with caching) ---
    player_idx = player - 1
    score_surface_to_blit = None # Initialize
    if 0 <= player_idx < len(player_names) and 0 <= player_idx < len(display_scores):
        player_name_display = player_names[player_idx] if player_names[player_idx] else f"Player {player}"
        current_score = display_scores[player_idx]

        # Check cache
        cached_data = SCORE_CACHE.get(player_idx)
        if cached_data and cached_data['score'] == current_score and cached_data['surface'] is not None:
            # Cache hit! Use the cached surface
            score_surface_to_blit = cached_data['surface']
            # print(f"DEBUG: Cache HIT for Player {player} score {current_score}") # Optional debug
        else:
            # Cache miss or score changed: Render new surface and update cache
            # print(f"DEBUG: Cache MISS for Player {player} score {current_score}") # Optional debug
            score_text_str = f"{player_name_display} Score: {current_score}"
            score_surface_to_blit = ui_font.render(score_text_str, True, BLACK)
            SCORE_CACHE[player_idx] = {'score': current_score, 'surface': score_surface_to_blit}

        # Blit the determined surface
        if score_surface_to_blit:
            screen.blit(score_surface_to_blit, (start_x, rack_y - 20))
        else:
             print(f"Warning: Could not get/render score surface for player {player}")

    else: print(f"Warning: Invalid player index {player} for score display.")
    # --- End score drawing ---

    return alpha_button_rect, rand_button_rect




# ADDED: Function to draw the Developer Tools dialog

def draw_dev_tools_dialog(visualize_checked):
    """
    Draws the Developer Tools dialog box.

    Args:
        visualize_checked (bool): The current state of the "Visualize Batch Games" checkbox.

    Returns:
        tuple: (visualize_checkbox_rect, close_button_rect)
    """
    dialog_width, dialog_height = 350, 180 # Adjusted size
    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2

    # Draw dialog background and border
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
    pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

    # Title
    title_text = dialog_font.render("Developer Tools", True, BLACK)
    screen.blit(title_text, (dialog_x + 10, dialog_y + 10))

    # Checkbox for Visualize Batch Games
    checkbox_x = dialog_x + 20
    checkbox_y = dialog_y + 60
    visualize_checkbox_rect = pygame.Rect(checkbox_x, checkbox_y, 20, 20)
    draw_checkbox(screen, checkbox_x, checkbox_y, visualize_checked)
    label_text = ui_font.render("Visualize Batch Games", True, BLACK)
    screen.blit(label_text, (checkbox_x + 30, checkbox_y + 2))

    # Close Button
    close_button_width = 80
    close_button_rect = pygame.Rect(
        dialog_x + (dialog_width - close_button_width) // 2, # Centered
        dialog_y + dialog_height - BUTTON_HEIGHT - 15, # Near bottom
        close_button_width,
        BUTTON_HEIGHT
    )
    hover = close_button_rect.collidepoint(pygame.mouse.get_pos())
    color = BUTTON_HOVER if hover else BUTTON_COLOR
    pygame.draw.rect(screen, color, close_button_rect)
    close_text = button_font.render("Close", True, BLACK)
    screen.blit(close_text, close_text.get_rect(center=close_button_rect.center))

    return visualize_checkbox_rect, close_button_rect






def draw_replay_icon(screen, rect, icon_type):
    """Draw centered replay control icons (arrows) on the buttons."""
    center_x, center_y = rect.center; arrow_size = 8
    if icon_type == "start": pygame.draw.polygon(screen, BLACK, [(center_x + 6, center_y - arrow_size), (center_x - 2, center_y), (center_x + 6, center_y + arrow_size)]); pygame.draw.polygon(screen, BLACK, [(center_x - 2, center_y - arrow_size), (center_x - 10, center_y), (center_x - 2, center_y + arrow_size)])
    elif icon_type == "prev": pygame.draw.polygon(screen, BLACK, [(center_x + 6, center_y - arrow_size), (center_x - 6, center_y), (center_x + 6, center_y + arrow_size)])
    elif icon_type == "next": pygame.draw.polygon(screen, BLACK, [(center_x - 6, center_y - arrow_size), (center_x + 6, center_y), (center_x - 6, center_y + arrow_size)])
    elif icon_type == "end": pygame.draw.polygon(screen, BLACK, [(center_x - 6, center_y - arrow_size), (center_x + 2, center_y), (center_x - 6, center_y + arrow_size)]); pygame.draw.polygon(screen, BLACK, [(center_x + 2, center_y - arrow_size), (center_x + 10, center_y), (center_x + 2, center_y + arrow_size)])




def draw_scoreboard(screen, move_history, scroll_offset, scores, is_ai, player_names, final_scores=None, game_over_state=False): # Added player_names parameter
    """Draws the scrollable scoreboard using full player names."""
    scoreboard_x = BOARD_SIZE + 275
    scoreboard_y = 40
    # Adjust width calculation slightly to prevent potential overlap with window edge
    scoreboard_width = max(200, WINDOW_WIDTH - scoreboard_x - 20) # Use scoreboard_x in calculation
    scoreboard_height = WINDOW_HEIGHT - 80

    # Ensure width doesn't make it go off-screen if window is narrow
    if scoreboard_x + scoreboard_width > WINDOW_WIDTH - 10:
        scoreboard_width = WINDOW_WIDTH - scoreboard_x - 10

    # Fallback if width becomes too small
    if scoreboard_width < 150: # Reduced minimum slightly
        scoreboard_x = WINDOW_WIDTH - 160 # Adjust position too
        scoreboard_width = 150

    scoreboard_surface = pygame.Surface((scoreboard_width, scoreboard_height))
    scoreboard_surface.fill(WHITE)
    running_scores = [0, 0]
    y_pos = 10 - scroll_offset
    line_height = 20 # ui_font.get_linesize() might be better if font changes

    for i, move in enumerate(move_history):
        player_idx = move.get('player', 1) - 1 # Default to player 1 if missing, get 0-based index
        if not (0 <= player_idx < 2): # Basic validation
             print(f"Warning: Invalid player index {player_idx+1} in move history item {i}")
             continue

        running_scores[player_idx] += move.get('score', 0)

        # Use player_names list ---
        player_label = f"P{player_idx + 1}" # Default label
        if player_names and 0 <= player_idx < len(player_names) and player_names[player_idx]:
            player_label = player_names[player_idx] # Use the actual name
        

        display_score = running_scores[player_idx]
        move_score = move.get('score', 0)
        score_sign = "+" if move_score >= 0 else "" # Add sign for score delta

        # Construct text string
        if move.get('move_type') == 'place':
            word = move.get('word_with_blanks', move.get('word', 'N/A')) # Prefer word_with_blanks
            coord = move.get('coord', 'N/A')
            text = f"{i+1}: {player_label} - {word} at {coord} ({score_sign}{move_score}) Total: {display_score}"
        elif move.get('move_type') == 'pass':
            text = f"{i+1}: {player_label} - Pass ({score_sign}{move_score}) Total: {display_score}"
        elif move.get('move_type') == 'exchange':
            exchanged_count = len(move.get('exchanged_tiles', []))
            text = f"{i+1}: {player_label} - Exch. {exchanged_count} ({score_sign}{move_score}) Total: {display_score}"
        else:
            text = f"{i+1}: {player_label} - Unknown Move Type"

        # Render and blit if visible
        text_surface = ui_font.render(text, True, BLACK)
        if y_pos < scoreboard_height and y_pos + line_height > 0:
            # Highlight alternate player turns for readability
            if player_idx == 0: # Player 1's turn background
                 highlight_rect = pygame.Rect(0, y_pos, scoreboard_width, line_height)
                 pygame.draw.rect(scoreboard_surface, HIGHLIGHT_BLUE, highlight_rect) # Light blue for P1
            # else: Player 2's turn uses default WHITE background

            scoreboard_surface.blit(text_surface, (10, y_pos))

        y_pos += line_height

    # Draw final scores if game is over
    if game_over_state and final_scores is not None:
        y_pos += line_height // 2 # Add a small gap
        p1_final_name = player_names[0] if player_names and player_names[0] else "P1"
        p2_final_name = player_names[1] if player_names and player_names[1] else "P2"
        final_text = f"Final: {p1_final_name}: {final_scores[0]}, {p2_final_name}: {final_scores[1]}"
        final_surface = ui_font.render(final_text, True, BLACK)
        if y_pos < scoreboard_height and y_pos + line_height > 0:
            scoreboard_surface.blit(final_surface, (10, y_pos))

    # Blit the complete scoreboard surface onto the main screen
    screen.blit(scoreboard_surface, (scoreboard_x, scoreboard_y))
    # Draw border around scoreboard area
    pygame.draw.rect(screen, BLACK, (scoreboard_x, scoreboard_y, scoreboard_width, scoreboard_height), 1)








# select_seven_letter_word, eight_letter_practice, mode_selection_screen, draw_options_menu (Unchanged from Part 2 provided previously)
def select_seven_letter_word(removed_letter, seven_letter_words):
    """Selects a random 7-letter word containing the removed letter."""
    candidates = [word for word in seven_letter_words if removed_letter in word]
    if not candidates: print(f"Warning: No 7-letter words contain '{removed_letter}'. Choosing random 7-letter word."); return random.choice(seven_letter_words) if seven_letter_words else None
    return random.choice(candidates)






def eight_letter_practice():
    """Handles the setup dialog and initialization for 8-Letter Bingo practice."""
    try:
        with open("7-letter-list.txt", "r") as seven_file, open("8-letter-list.txt", "r") as eight_file:
            seven_letter_words = [line.strip().upper() for line in seven_file.readlines()]; eight_letter_words = [line.strip().upper() for line in eight_file.readlines()]
    except FileNotFoundError: print("Error: Could not find 7-letter-list.txt or 8-letter-list.txt"); return False, None, None, None, None, None
    if not seven_letter_words or not eight_letter_words: print("Error: Word list files are empty."); return False, None, None, None, None, None
    dialog_width, dialog_height = 300, 150; dialog_x = (WINDOW_WIDTH - dialog_width) // 2; dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
    proceed = False; text_box_active = False; probability_input = ""
    while True: # Dialog loop
        screen.fill(WHITE); pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height)); pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)
        title_text = dialog_font.render("8-Letter Bingo Options", True, BLACK); screen.blit(title_text, (dialog_x + 10, dialog_y + 10))
        prob_text = ui_font.render("Probability", True, BLACK); screen.blit(prob_text, (dialog_x + 20, dialog_y + 50))
        text_box_rect = pygame.Rect(dialog_x + 120, dialog_y + 45, 150, 30); pygame.draw.rect(screen, WHITE, text_box_rect); pygame.draw.rect(screen, BLACK, text_box_rect, 1)
        input_text = ui_font.render(probability_input, True, BLACK); screen.blit(input_text, (text_box_rect.x + 5, text_box_rect.y + 5))
        go_rect = pygame.Rect(dialog_x + 50, dialog_y + 100, 100, 30); cancel_rect = pygame.Rect(dialog_x + 160, dialog_y + 100, 100, 30)
        pygame.draw.rect(screen, BUTTON_COLOR, go_rect); pygame.draw.rect(screen, BUTTON_COLOR, cancel_rect)
        go_text = button_font.render("Go", True, BLACK); cancel_text = button_font.render("Cancel", True, BLACK)
        screen.blit(go_text, go_text.get_rect(center=go_rect.center)); screen.blit(cancel_text, cancel_text.get_rect(center=cancel_rect.center))
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos; text_box_active = text_box_rect.collidepoint(x, y)
                if go_rect.collidepoint(x, y):
                    max_index = len(eight_letter_words);
                    if probability_input.isdigit(): prob_val = int(probability_input); max_index = min(max(1, prob_val), len(eight_letter_words))
                    selected_eight = random.choice(eight_letter_words[:max_index]); print("Selected 8-letter word:", selected_eight)
                    remove_idx = random.randint(0, 7); removed_letter = selected_eight[remove_idx]; removed_eight = selected_eight[:remove_idx] + selected_eight[remove_idx + 1:]
                    print("Player 1 rack (7 letters):", removed_eight); print("Removed letter:", removed_letter)
                    selected_seven = select_seven_letter_word(removed_letter, seven_letter_words)
                    if selected_seven is None: print("Error: Could not find a suitable 7-letter word."); return False, None, None, None, None, None
                    print("Selected 7-letter word for board:", selected_seven)
                    board, _, tiles = create_board(); local_racks = [[], []]; local_blanks = set(); local_racks[0] = sorted(list(removed_eight)); local_racks[1] = []
                    center_r, center_c = CENTER_SQUARE; word_len = len(selected_seven); start_offset = word_len // 2; place_horizontally = random.choice([True, False]); placement_successful = False
                    if place_horizontally:
                        start_c_place = center_c - start_offset
                        if 0 <= start_c_place and start_c_place + word_len <= GRID_SIZE:
                            for i, letter in enumerate(selected_seven): tiles[center_r][start_c_place + i] = letter
                            placement_successful = True; print(f"Placed '{selected_seven}' horizontally at ({center_r},{start_c_place})")
                    if not placement_successful: # Try vertically
                        start_r_place = center_r - start_offset
                        if 0 <= start_r_place and start_r_place + word_len <= GRID_SIZE:
                            for i, letter in enumerate(selected_seven): tiles[start_r_place + i][center_c] = letter
                            placement_successful = True; print(f"Placed '{selected_seven}' vertically at ({start_r_place},{center_c})")
                    if not placement_successful: print("Error: Could not place 7-letter word centered H or V."); return False, None, None, None, None, None
                    local_bag = []; return True, board, tiles, local_racks, local_blanks, local_bag
                elif cancel_rect.collidepoint(x, y): return False, None, None, None, None, None
            elif event.type == pygame.KEYDOWN and text_box_active:
                if event.key == pygame.K_BACKSPACE: probability_input = probability_input[:-1]
                elif event.unicode.isdigit(): probability_input += event.unicode
        pygame.display.flip()
    return False, None, None, None, None, None # Should not be reached


def is_word_length_allowed(word_len, number_checks):
    """
    Checks if a given word length is allowed based on the number_checks list.
    number_checks corresponds to lengths [2, 3, 4, 5, 6, 7+].
    """
    if word_len < 2: # Words must be at least 2 letters
        return False
    if word_len == 2 and number_checks[0]: return True
    if word_len == 3 and number_checks[1]: return True
    if word_len == 4 and number_checks[2]: return True
    if word_len == 5 and number_checks[3]: return True
    if word_len == 6 and number_checks[4]: return True
    if word_len >= 7 and number_checks[5]: return True # 7+ checkbox
    return False





# Function to Replace: mode_selection_screen
# REASON: Move Developer Tools button to bottom-left.

def mode_selection_screen():
    """Display and handle the game mode selection screen, including Load Game via text input and Developer Tools."""
    print("--- mode_selection_screen() entered ---")
    global main_called
    global pyperclip_available, pyperclip

    try:
        print("--- mode_selection_screen(): Attempting to load background image... ---")
        image = pygame.image.load("Scrabble_S.png").convert_alpha(); content_width = WINDOW_WIDTH - 200; image = pygame.transform.scale(image, (content_width, WINDOW_HEIGHT)); image.set_alpha(128); content_left = (WINDOW_WIDTH - content_width) // 2
        print("--- mode_selection_screen(): Background image loaded and processed. ---")
    except pygame.error as e:
        print(f"--- mode_selection_screen(): WARNING - Could not load background image 'Scrabble_S.png': {e} ---")
        image = None; content_width = WINDOW_WIDTH; content_left = 0

    modes = [MODE_HVH, MODE_HVA, MODE_AVA]; selected_mode = None; player_names = ["Player 1", "Player 2"]; human_player = 1; input_active = [False, False]; current_input = 0
    practice_mode = None; dropdown_open = False; showing_power_tiles_dialog = False
    letter_checks = [True, True, True, True] # J, Q, X, Z
    number_checks = [True, True, True, True, False, False] # 2, 3, 4, 5, 6, 7+
    practice_state = None
    loaded_game_data = None
    use_endgame_solver_checked = False
    use_ai_simulation_checked = False

    # State for Load Game Text Input
    showing_load_input = False; load_filename_input = ""; load_input_active = False
    load_confirm_button_rect = None; load_input_rect = None; load_cancel_button_rect = None

    # Developer Tools State
    showing_dev_tools_dialog = False
    visualize_batch_checked = False # Default to False (unchecked)
    dev_tools_visualize_rect = None # Rect for checkbox click detection
    dev_tools_close_rect = None # Rect for close button click detection

    print("--- mode_selection_screen(): Entering main loop (while selected_mode is None:)... ---")
    loop_count = 0
    while selected_mode is None:
        loop_count += 1
        pygame.event.pump()

        # --- Define positions INSIDE the loop ---
        option_rects = []
        name_rect_x = content_left + (content_width - 200) // 2

        # Button Positions (Bottom Row)
        play_later_rect = pygame.Rect(WINDOW_WIDTH - BUTTON_WIDTH - 10, WINDOW_HEIGHT - BUTTON_HEIGHT - 10, BUTTON_WIDTH, BUTTON_HEIGHT)
        load_game_button_rect = pygame.Rect(play_later_rect.left - BUTTON_GAP - BUTTON_WIDTH, play_later_rect.top, BUTTON_WIDTH, BUTTON_HEIGHT)
        batch_game_button_rect = pygame.Rect(load_game_button_rect.left - BUTTON_GAP - BUTTON_WIDTH, play_later_rect.top, BUTTON_WIDTH, BUTTON_HEIGHT)
        start_game_button_rect = pygame.Rect(batch_game_button_rect.left - BUTTON_GAP - BUTTON_WIDTH, play_later_rect.top, BUTTON_WIDTH, BUTTON_HEIGHT)
        # --- MODIFICATION: Developer Tools Button Position (Bottom Left) ---
        dev_tools_button_width = 120 # Slightly wider for text
        dev_tools_button_rect = pygame.Rect(
            10, # Left edge padding
            play_later_rect.top, # Align vertically with other bottom buttons
            dev_tools_button_width,
            BUTTON_HEIGHT
        )
        # --- END MODIFICATION ---


        # Calculate Load Input Field/Button Positions unconditionally
        load_input_width = 300; load_input_x = load_game_button_rect.left; load_input_y = load_game_button_rect.top - BUTTON_GAP - BUTTON_HEIGHT
        load_input_rect = pygame.Rect(load_input_x, load_input_y, load_input_width, BUTTON_HEIGHT)
        load_confirm_button_rect = pygame.Rect(load_input_x + load_input_width + BUTTON_GAP, load_input_y, BUTTON_WIDTH, BUTTON_HEIGHT)
        load_cancel_button_rect = pygame.Rect(load_confirm_button_rect.right + BUTTON_GAP, load_input_y, BUTTON_WIDTH, BUTTON_HEIGHT)

        # Calculate Checkbox Positions
        last_mode_button_y = 100 + (len(modes) - 1) * 60 + BUTTON_HEIGHT
        checkbox_x_base = content_left + (content_width - 250) // 2 # Approx center alignment
        checkbox_gap = 25 # Vertical gap between checkboxes

        # Endgame Solver Checkbox Position
        endgame_checkbox_x = checkbox_x_base
        endgame_checkbox_y = last_mode_button_y + 20
        endgame_checkbox_rect = pygame.Rect(endgame_checkbox_x, endgame_checkbox_y, 20, 20)
        endgame_label_x = endgame_checkbox_x + 25
        endgame_label_y = endgame_checkbox_y + 2

        # AI Simulation Checkbox Position
        simulation_checkbox_x = checkbox_x_base
        simulation_checkbox_y = endgame_checkbox_y + checkbox_gap
        simulation_checkbox_rect = pygame.Rect(simulation_checkbox_x, simulation_checkbox_y, 20, 20)
        simulation_label_x = simulation_checkbox_x + 25
        simulation_label_y = simulation_checkbox_y + 2

        # Calculate Player Name Input Positions Dynamically
        name_input_gap = 30 # Gap below the last checkbox
        p1_y_pos = simulation_checkbox_y + simulation_checkbox_rect.height + name_input_gap
        player_name_gap = 40 # Gap between P1 and P2 inputs
        p2_y_pos = p1_y_pos + BUTTON_HEIGHT + player_name_gap # P2 position relative to P1

        # Calculate HVA button rects unconditionally (relative to p2_y_pos)
        hva_button_row_y = p2_y_pos + BUTTON_HEIGHT + 10
        hva_buttons_total_width = (BUTTON_WIDTH * 2 + 20)
        hva_buttons_start_x = content_left + (content_width - hva_buttons_total_width) // 2
        p1_rect_hva = pygame.Rect(hva_buttons_start_x, hva_button_row_y, BUTTON_WIDTH, BUTTON_HEIGHT)
        p2_rect_hva = pygame.Rect(hva_buttons_start_x + BUTTON_WIDTH + 20, hva_button_row_y, BUTTON_WIDTH, BUTTON_HEIGHT)

        # Dropdown positioning (relative to p2_y_pos)
        if modes[current_input] == MODE_HVH and dropdown_open:
            dropdown_x = name_rect_x
            dropdown_button_y = p2_y_pos + BUTTON_HEIGHT + 10 # Position relative to P2 input
            dropdown_y = dropdown_button_y + 30
            options = ["Power Tiles", "8-Letter Bingos", "Bingo, Bango, Bongo", "Only Fives", "End Game"]
            for i, option in enumerate(options): option_rect = pygame.Rect(dropdown_x, dropdown_y + 30 * i, 200, 30); option_rects.append(option_rect)


        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

            # Handle Dev Tools Dialog FIRST
            if showing_dev_tools_dialog:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos
                    # Checkbox click
                    if dev_tools_visualize_rect and dev_tools_visualize_rect.collidepoint(x, y):
                        visualize_batch_checked = not visualize_batch_checked
                    # Close button click
                    elif dev_tools_close_rect and dev_tools_close_rect.collidepoint(x, y):
                        showing_dev_tools_dialog = False
                elif event.type == pygame.KEYDOWN:
                     if event.key == pygame.K_ESCAPE: # Allow Esc to close
                          showing_dev_tools_dialog = False
                continue # Skip other event handling when dialog is open

            # Handle Load Game Input if active
            if showing_load_input:
                # [ ... Load game input handling ... ] (unchanged)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if load_input_rect.collidepoint(x, y): load_input_active = True
                    else: load_input_active = False
                    if load_confirm_button_rect.collidepoint(x, y): # Load Confirm ...
                        filepath = load_filename_input.strip()
                        if filepath:
                             print(f"--- mode_selection_screen(): Attempting to load typed file: {filepath} ---")
                             try:
                                 loaded_p_names, loaded_hist, loaded_f_scores = load_game_from_gcg(filepath)
                                 print(f"--- mode_selection_screen(): GCG loaded successfully. Moves: {len(loaded_hist)} ---")
                                 selected_mode = "LOADED_GAME"; loaded_game_data = (loaded_p_names, loaded_hist, loaded_f_scores)
                                 showing_load_input = False; load_input_active = False; break
                             except FileNotFoundError: print(f"--- mode_selection_screen(): Error: File not found '{filepath}' ---"); show_message_dialog(f"Error: File not found:\n{filepath}", "Load Error"); load_input_active = True
                             except Exception as e: print(f"--- mode_selection_screen(): Error loading GCG file '{filepath}': {e} ---"); show_message_dialog(f"Error loading file:\n{e}", "Load Error"); load_input_active = True
                        else: show_message_dialog("Please enter a filename.", "Load Error"); load_input_active = True
                    elif load_cancel_button_rect.collidepoint(x,y): # Load Cancel ...
                         showing_load_input = False; load_input_active = False; load_filename_input = ""
                    if not load_input_rect.collidepoint(x,y): input_active = [False, False] # Deactivate name input too
                elif event.type == pygame.KEYDOWN and load_input_active: # Load Input Typing ...
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                         if load_confirm_button_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': load_confirm_button_rect.center, 'button': 1}))
                    elif event.key == pygame.K_BACKSPACE: load_filename_input = load_filename_input[:-1]
                    elif event.key == pygame.K_v:
                        mods = pygame.key.get_mods()
                        if (mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META) and pyperclip_available:
                            try:
                                pasted_text = pyperclip.paste()
                                if pasted_text: load_filename_input += pasted_text
                            except Exception as e: print(f"Error pasting from clipboard: {e}")
                    elif event.unicode.isprintable(): load_filename_input += event.unicode
                continue # Skip rest of event handling

            # Handle other events if load input wasn't active
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if showing_power_tiles_dialog: # Power Tiles Dialog ...
                    # [ ... Power tiles dialog handling ... ] (unchanged)
                    dialog_x = (WINDOW_WIDTH - 300) // 2; dialog_y = (WINDOW_HEIGHT - 250) // 2; letter_rects = [pygame.Rect(dialog_x + 20, dialog_y + 40 + i*30, 20, 20) for i in range(4)]; number_rects = [pygame.Rect(dialog_x + 150, dialog_y + 40 + i*30, 20, 20) for i in range(6)]
                    go_rect = pygame.Rect(dialog_x + 50, dialog_y + 220, 100, 30); cancel_rect = pygame.Rect(dialog_x + 160, dialog_y + 220, 100, 30)
                    for i, rect in enumerate(letter_rects):
                        if rect.collidepoint(x, y): letter_checks[i] = not letter_checks[i]
                    for i, rect in enumerate(number_rects):
                        if rect.collidepoint(x, y): number_checks[i] = not number_checks[i] # Update number_checks state
                    if go_rect.collidepoint(x, y):
                        if not any(letter_checks):
                            show_message_dialog("Please select at least one Power Tile (J, Q, X, Z).", "Selection Required")
                        elif not any(number_checks):
                             show_message_dialog("Please select at least one word length.", "Selection Required")
                        else:
                            practice_mode = "power_tiles"; selected_mode = MODE_AVA; showing_power_tiles_dialog = False; print(f"--- mode_selection_screen(): Mode selected via Power Tiles Go: {selected_mode} ---"); break
                    elif cancel_rect.collidepoint(x, y): showing_power_tiles_dialog = False
                else: # Main Selection Screen ...
                    # Must define mode_rects before checking collision
                    mode_rects = []
                    for i, mode in enumerate(modes):
                        y_pos_mode = 100 + i * 60 # Use different variable name to avoid conflict
                        rect = pygame.Rect(content_left + (content_width - (BUTTON_WIDTH * 2 + 20)) // 2, y_pos_mode, BUTTON_WIDTH * 2 + 20, BUTTON_HEIGHT); mode_rects.append(rect)
                    # Check mode buttons
                    for i, rect in enumerate(mode_rects):
                        if rect.collidepoint(x, y):
                            current_input = i; dropdown_open = False
                            if i == 0: player_names = ["Player 1", "Player 2"]; input_active = [False, False]
                            elif i == 1: player_names = ["Player 1", "AI"] if human_player == 1 else ["AI", "Player 2"]; input_active = [True, False] if human_player == 1 else [False, True] # Set initial HVA state
                            elif i == 2: player_names = ["AI 1", "AI 2"]; input_active = [False, False]

                    # Handle Endgame Solver Checkbox Click
                    if endgame_checkbox_rect.collidepoint(x, y):
                        use_endgame_solver_checked = not use_endgame_solver_checked
                    # Handle AI Simulation Checkbox Click
                    elif simulation_checkbox_rect.collidepoint(x, y):
                        use_ai_simulation_checked = not use_ai_simulation_checked

                    # Handle Developer Tools Button Click
                    elif dev_tools_button_rect.collidepoint(x, y):
                        showing_dev_tools_dialog = True

                    # Handle Batch Game Button Click
                    elif batch_game_button_rect.collidepoint(x, y):
                        print("--- mode_selection_screen(): Batch Games button clicked ---")
                        current_selected_game_mode = modes[current_input]
                        if current_selected_game_mode == MODE_HVH:
                             show_message_dialog("Batch mode not available for Human vs Human.", "Mode Error")
                        else: # HVA or AVA selected, proceed
                            num_games = get_batch_game_dialog()
                            if num_games is not None:
                                print(f"--- mode_selection_screen(): Starting batch of {num_games} games ---")
                                selected_mode = "BATCH_MODE"
                                # Include visualize_batch_checked
                                loaded_game_data = (current_selected_game_mode, player_names, human_player, use_endgame_solver_checked, use_ai_simulation_checked, num_games, visualize_batch_checked)
                                break # Exit mode selection loop
                            else:
                                print("--- mode_selection_screen(): Batch game setup cancelled ---")

                    # Handle Start Game Button Click
                    elif start_game_button_rect.collidepoint(x, y):
                        selected_mode = modes[current_input] # Set the selected mode now
                        print(f"--- mode_selection_screen(): Start Game clicked. Mode: {selected_mode} ---")
                        break # Exit mode selection loop

                    # Check Load Game Button
                    elif load_game_button_rect.collidepoint(x, y):
                        print("--- mode_selection_screen(): Load Game button clicked, showing input ---")
                        showing_load_input = True; load_input_active = True; load_filename_input = ""
                        continue # Skip rest

                    # Check Other Buttons
                    elif play_later_rect.collidepoint(x, y): print("--- mode_selection_screen(): Play Later clicked. Exiting. ---"); pygame.quit(); sys.exit()
                    else: # Handle name inputs, HVA selection, practice dropdown
                        # ... (rest of name input, HVA, practice dropdown logic - unchanged) ...
                        clicked_name_input = False
                        p1_name_rect = pygame.Rect(name_rect_x, p1_y_pos, 200, BUTTON_HEIGHT)
                        p2_name_rect = pygame.Rect(name_rect_x, p2_y_pos, 200, BUTTON_HEIGHT) if modes[current_input] == MODE_HVH else None

                        if modes[current_input] == MODE_HVH: # HVH Name Input ...
                            if p1_name_rect.collidepoint(x, y): input_active = [True, False]; clicked_name_input = True
                            elif p2_name_rect and p2_name_rect.collidepoint(x, y): input_active = [False, True]; clicked_name_input = True
                        elif modes[current_input] == MODE_HVA: # HVA Name Input / Play As ...
                            if human_player == 1 and p1_name_rect.collidepoint(x, y): input_active = [True, False]; clicked_name_input = True
                            elif human_player == 2:
                                p2_name_rect_hva = pygame.Rect(name_rect_x, p2_y_pos, 200, BUTTON_HEIGHT)
                                if p2_name_rect_hva.collidepoint(x,y): input_active = [False, True]; clicked_name_input = True
                            if p1_rect_hva.collidepoint(x, y):
                                human_player = 1; player_names = ["Player 1", "AI"]; input_active = [True, False]
                            elif p2_rect_hva.collidepoint(x, y):
                                human_player = 2; player_names = ["AI", "Player 2"]; input_active = [False, True]

                        if modes[current_input] == MODE_HVH: # Practice Dropdown ...
                            dropdown_button_y = p2_y_pos + BUTTON_HEIGHT + 10
                            dropdown_rect = pygame.Rect(name_rect_x, dropdown_button_y, 200, 30)
                            if dropdown_rect.collidepoint(x, y): dropdown_open = not dropdown_open
                            elif dropdown_open:
                                clicked_option = False
                                current_options = ["Power Tiles", "8-Letter Bingos", "Bingo, Bango, Bongo", "Only Fives", "End Game"]
                                for i, option_rect in enumerate(option_rects):
                                    if option_rect.collidepoint(x, y):
                                        clicked_option = True; dropdown_open = False
                                        selected_practice_option = current_options[i]
                                        if selected_practice_option == "Power Tiles": showing_power_tiles_dialog = True
                                        elif selected_practice_option == "8-Letter Bingos":
                                            print("--- mode_selection_screen(): 8-Letter Bingo practice selected. Calling eight_letter_practice()... ---")
                                            proceed, p_board, p_tiles, p_racks, p_blanks, p_bag = eight_letter_practice()
                                            if proceed:
                                                practice_mode = "eight_letter"; selected_mode = MODE_HVH; player_names = ["Player 1", ""]; human_player = 1
                                                practice_state = {"board": p_board, "tiles": p_tiles, "racks": p_racks, "blanks": p_blanks, "bag": p_bag, "first_play": False, "scores": [0, 0], "turn": 1}
                                                print(f"--- mode_selection_screen(): 8-Letter practice setup successful. Selected mode: {selected_mode} ---"); break
                                            else: print("--- mode_selection_screen(): 8-Letter practice setup cancelled or failed. ---")
                                        elif selected_practice_option == "Bingo, Bango, Bongo":
                                            print("--- mode_selection_screen(): Bingo, Bango, Bongo practice selected. ---")
                                            practice_mode = "bingo_bango_bongo"; selected_mode = MODE_AVA; player_names = ["AI 1", "AI 2"]; practice_state = None
                                            print(f"--- mode_selection_screen(): Bingo, Bango, Bongo setup successful. Selected mode: {selected_mode} ---"); break
                                        elif selected_practice_option == "Only Fives":
                                            print("--- mode_selection_screen(): Only Fives practice selected. ---")
                                            practice_mode = "only_fives"; selected_mode = MODE_HVA; human_player = 1; player_names = ["Player 1", "AI"]; practice_state = None
                                            print(f"--- mode_selection_screen(): Only Fives setup successful. Selected mode: {selected_mode} ---"); break
                                        elif selected_practice_option == "End Game": print("End Game practice selected - Not implemented yet")
                                        break
                                if not clicked_option and not dropdown_rect.collidepoint(x,y): dropdown_open = False
                            elif not dropdown_rect.collidepoint(x,y): dropdown_open = False

                        dropdown_button_y_check = p2_y_pos + BUTTON_HEIGHT + 10
                        dropdown_rect_check = pygame.Rect(name_rect_x, dropdown_button_y_check, 200, 30) if modes[current_input] == MODE_HVH else None
                        if not clicked_name_input and not (dropdown_open and any(r.collidepoint(x,y) for r in option_rects)) and not p1_name_rect.collidepoint(x,y) and not (p2_name_rect and p2_name_rect.collidepoint(x,y)) and not (dropdown_rect_check and dropdown_rect_check.collidepoint(x,y)) and not endgame_checkbox_rect.collidepoint(x, y) and not simulation_checkbox_rect.collidepoint(x, y):
                            input_active = [False, False]

            elif event.type == pygame.KEYDOWN: # Keyboard Input (Names only now)
                if not showing_load_input: # Only handle name input if load input isn't showing
                     active_idx = -1
                     if input_active[0]: active_idx = 0
                     elif input_active[1] and modes[current_input] == MODE_HVH: active_idx = 1
                     elif input_active[1] and modes[current_input] == MODE_HVA and human_player == 2: active_idx = 1
                     if active_idx != -1:
                        if event.key == pygame.K_BACKSPACE: player_names[active_idx] = player_names[active_idx][:-1]
                        elif event.key == pygame.K_RETURN: input_active[active_idx] = False
                        elif event.unicode.isalnum() or event.unicode == ' ':
                            if len(player_names[active_idx]) < 15: player_names[active_idx] += event.unicode

        # --- Drawing Logic ---
        screen.fill(WHITE);
        if image: screen.blit(image, (content_left, 0))
        title_text = dialog_font.render("Select Game Mode", True, BLACK); title_x = content_left + (content_width - title_text.get_width()) // 2; screen.blit(title_text, (title_x, 50))
        mode_rects = [] # Draw Mode Buttons ...
        for i, mode in enumerate(modes):
            y_pos_mode = 100 + i * 60 # Use different variable name
            rect = pygame.Rect(content_left + (content_width - (BUTTON_WIDTH * 2 + 20)) // 2, y_pos_mode, BUTTON_WIDTH * 2 + 20, BUTTON_HEIGHT); hover = rect.collidepoint(pygame.mouse.get_pos())
            color = BUTTON_HOVER if i == current_input or hover else BUTTON_COLOR; pygame.draw.rect(screen, color, rect)
            if i == current_input: pygame.draw.rect(screen, BLACK, rect, 2)
            text = button_font.render(mode, True, BLACK); text_rect = text.get_rect(center=rect.center); screen.blit(text, text_rect); mode_rects.append(rect)

        # Draw Endgame Solver Checkbox
        draw_checkbox(screen, endgame_checkbox_rect.x, endgame_checkbox_rect.y, use_endgame_solver_checked)
        endgame_label_surf = ui_font.render("Use AI Endgame Solver", True, BLACK)
        screen.blit(endgame_label_surf, (endgame_label_x, endgame_label_y))

        # Draw AI Simulation Checkbox
        draw_checkbox(screen, simulation_checkbox_rect.x, simulation_checkbox_rect.y, use_ai_simulation_checked)
        simulation_label_surf = ui_font.render("Use AI 2-ply Simulation", True, BLACK)
        screen.blit(simulation_label_surf, (simulation_label_x, simulation_label_y))

        # Draw Play Later Button ...
        hover = play_later_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, play_later_rect)
        play_later_text = button_font.render("Play Later", True, BLACK); play_later_text_rect = play_later_text.get_rect(center=play_later_rect.center); screen.blit(play_later_text, play_later_text_rect)

        # Draw Load Game Button ...
        hover = load_game_button_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, load_game_button_rect)
        load_game_text = button_font.render("Load Game", True, BLACK); load_game_text_rect = load_game_text.get_rect(center=load_game_button_rect.center); screen.blit(load_game_text, load_game_text_rect)

        # Draw Batch Game Button
        hover = batch_game_button_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, batch_game_button_rect)
        batch_game_text = button_font.render("Batch Games", True, BLACK); batch_game_text_rect = batch_game_text.get_rect(center=batch_game_button_rect.center); screen.blit(batch_game_text, batch_game_text_rect)

        # Draw Start Game Button
        hover = start_game_button_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, start_game_button_rect)
        start_game_text = button_font.render("Start Game", True, BLACK); start_game_text_rect = start_game_text.get_rect(center=start_game_button_rect.center); screen.blit(start_game_text, start_game_text_rect)

        # Draw Developer Tools Button (Bottom Left)
        hover = dev_tools_button_rect.collidepoint(pygame.mouse.get_pos())
        color = BUTTON_HOVER if hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, dev_tools_button_rect)
        dev_tools_text = button_font.render("Developer Tools", True, BLACK)
        dev_tools_text_rect = dev_tools_text.get_rect(center=dev_tools_button_rect.center)
        screen.blit(dev_tools_text, dev_tools_text_rect)


        # Draw Load Game Input Field (if active)
        if showing_load_input:
             pygame.draw.rect(screen, WHITE, load_input_rect); pygame.draw.rect(screen, BLACK, load_input_rect, 1 if not load_input_active else 2)
             input_surf = ui_font.render(load_filename_input, True, BLACK); screen.blit(input_surf, (load_input_rect.x + 5, load_input_rect.y + 5))
             if load_input_active and int(time.time() * 2) % 2 == 0: # Blinking cursor
                  cursor_x = load_input_rect.x + 5 + input_surf.get_width(); cursor_y1 = load_input_rect.y + 5; cursor_y2 = load_input_rect.y + load_input_rect.height - 5
                  pygame.draw.line(screen, BLACK, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 1)
             hover = load_confirm_button_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR # Confirm Button
             pygame.draw.rect(screen, color, load_confirm_button_rect); text = button_font.render("Load File", True, BLACK); text_rect = text.get_rect(center=load_confirm_button_rect.center); screen.blit(text, text_rect)
             hover = load_cancel_button_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR # Cancel Button
             pygame.draw.rect(screen, color, load_cancel_button_rect); text = button_font.render("Cancel", True, BLACK); text_rect = text.get_rect(center=load_cancel_button_rect.center); screen.blit(text, text_rect)

        # Name Input / Practice Dropdown / HVA Selection Drawing ...
        # ... (rest of drawing logic for names, HVA, practice dropdown - unchanged) ...
        name_rect_width = 200;
        p1_label_text = "Player 1 Name:"; p1_label = ui_font.render(p1_label_text, True, BLACK);
        p1_name_rect = pygame.Rect(name_rect_x, p1_y_pos, name_rect_width, BUTTON_HEIGHT) # Define for drawing
        p1_label_x = name_rect_x - p1_label.get_width() - 10; screen.blit(p1_label, (p1_label_x, p1_y_pos + 5)); p1_bg_color = LIGHT_BLUE if input_active[0] else (GRAY if modes[current_input] == MODE_AVA else WHITE); pygame.draw.rect(screen, p1_bg_color, p1_name_rect); pygame.draw.rect(screen, BLACK, p1_name_rect, 1)
        p1_name_text = ui_font.render(player_names[0], True, BLACK); screen.blit(p1_name_text, (p1_name_rect.x + 5, p1_name_rect.y + 5))

        if modes[current_input] == MODE_HVH:
            p2_label_text = "Player 2 Name:"; p2_label = ui_font.render(p2_label_text, True, BLACK);
            p2_name_rect = pygame.Rect(name_rect_x, p2_y_pos, name_rect_width, BUTTON_HEIGHT) # Define for drawing
            p2_label_x = name_rect_x - p2_label.get_width() - 10; screen.blit(p2_label, (p2_label_x, p2_y_pos + 5))
            pygame.draw.rect(screen, WHITE if not input_active[1] else LIGHT_BLUE, p2_name_rect); pygame.draw.rect(screen, BLACK, p2_name_rect, 1); p2_name_text = ui_font.render(player_names[1], True, BLACK); screen.blit(p2_name_text, (p2_name_rect.x + 5, p2_name_rect.y + 5))
            dropdown_button_y = p2_y_pos + BUTTON_HEIGHT + 10
            dropdown_rect = pygame.Rect(name_rect_x, dropdown_button_y, 200, 30); hover = dropdown_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, dropdown_rect)
            text = button_font.render("Practice", True, BLACK); text_rect = text.get_rect(center=dropdown_rect.center); screen.blit(text, text_rect)
            if dropdown_open:
                current_options = ["Power Tiles", "8-Letter Bingos", "Bingo, Bango, Bongo", "Only Fives", "End Game"]
                for i, option_rect in enumerate(option_rects):
                     hover = option_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else DROPDOWN_COLOR; pygame.draw.rect(screen, color, option_rect);
                     text = button_font.render(current_options[i], True, BLACK); text_rect = text.get_rect(center=option_rect.center); screen.blit(text, text_rect)
        elif modes[current_input] == MODE_HVA:
            p2_label_text = "AI Name:"; p2_label = ui_font.render(p2_label_text, True, BLACK);
            p2_name_rect = pygame.Rect(name_rect_x, p2_y_pos, name_rect_width, BUTTON_HEIGHT) # Define for drawing
            p2_label_x = name_rect_x - p2_label.get_width() - 10; screen.blit(p2_label, (p2_label_x, p2_y_pos + 5))
            pygame.draw.rect(screen, GRAY, p2_name_rect); pygame.draw.rect(screen, BLACK, p2_name_rect, 1); p2_name_text = ui_font.render(player_names[1], True, BLACK); screen.blit(p2_name_text, (p2_name_rect.x + 5, p2_name_rect.y + 5))
            p1_hover = p1_rect_hva.collidepoint(pygame.mouse.get_pos()); p2_hover = p2_rect_hva.collidepoint(pygame.mouse.get_pos()); pygame.draw.rect(screen, BUTTON_HOVER if p1_hover else BUTTON_COLOR, p1_rect_hva);
            if human_player == 1: pygame.draw.rect(screen, BLACK, p1_rect_hva, 2)
            pygame.draw.rect(screen, BUTTON_HOVER if p2_hover else BUTTON_COLOR, p2_rect_hva);
            if human_player == 2: pygame.draw.rect(screen, BLACK, p2_rect_hva, 2)
            p1_text = button_font.render("Play as P1", True, BLACK); p2_text = button_font.render("Play as P2", True, BLACK); p1_text_rect = p1_text.get_rect(center=p1_rect_hva.center); p2_text_rect = p2_text.get_rect(center=p2_rect_hva.center); screen.blit(p1_text, p1_text_rect); screen.blit(p2_text, p2_text_rect)
        elif modes[current_input] == MODE_AVA:
            p2_label_text = "AI 2 Name:"; p2_label = ui_font.render(p2_label_text, True, BLACK);
            p2_name_rect = pygame.Rect(name_rect_x, p2_y_pos, name_rect_width, BUTTON_HEIGHT) # Define for drawing
            p2_label_x = name_rect_x - p2_label.get_width() - 10; screen.blit(p2_label, (p2_label_x, p2_y_pos + 5))
            pygame.draw.rect(screen, GRAY, p2_name_rect); pygame.draw.rect(screen, BLACK, p2_name_rect, 1); p2_name_text = ui_font.render(player_names[1], True, BLACK); screen.blit(p2_name_text, (p2_name_rect.x + 5, p2_name_rect.y + 5))
            pygame.draw.rect(screen, GRAY, p1_name_rect); pygame.draw.rect(screen, BLACK, p1_name_rect, 1); p1_name_text = ui_font.render(player_names[0], True, BLACK); screen.blit(p1_name_text, (p1_name_rect.x + 5, p1_name_rect.y + 5))

        # Draw Power Tiles Dialog if active ...
        if showing_power_tiles_dialog:
            # ... (power tiles dialog drawing - unchanged) ...
            dialog_width, dialog_height = 300, 250; dialog_x = (WINDOW_WIDTH - dialog_width) // 2; dialog_y = (WINDOW_HEIGHT - dialog_height) // 2; pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height)); pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)
            title_text = dialog_font.render("Power Tiles Options", True, BLACK); screen.blit(title_text, (dialog_x + 10, dialog_y + 10)); letters = ['J', 'Q', 'X', 'Z']
            for i, letter in enumerate(letters): draw_checkbox(screen, dialog_x + 20, dialog_y + 40 + i*30, letter_checks[i]); text = ui_font.render(letter, True, BLACK); screen.blit(text, (dialog_x + 50, dialog_y + 40 + i*30))
            numbers = ['2', '3', '4', '5', '6', '7+']
            for i, num in enumerate(numbers): draw_checkbox(screen, dialog_x + 150, dialog_y + 40 + i*30, number_checks[i]); text = ui_font.render(num, True, BLACK); screen.blit(text, (dialog_x + 180, dialog_y + 40 + i*30))
            go_rect = pygame.Rect(dialog_x + 50, dialog_y + 220, 100, 30); cancel_rect = pygame.Rect(dialog_x + 160, dialog_y + 220, 100, 30); pygame.draw.rect(screen, BUTTON_COLOR, go_rect); pygame.draw.rect(screen, BUTTON_COLOR, cancel_rect)
            go_text = button_font.render("Go", True, BLACK); cancel_text = button_font.render("Cancel", True, BLACK); screen.blit(go_text, go_text.get_rect(center=go_rect.center)); screen.blit(cancel_text, cancel_text.get_rect(center=cancel_rect.center))

        # Draw Developer Tools Dialog
        if showing_dev_tools_dialog:
            dev_tools_visualize_rect, dev_tools_close_rect = draw_dev_tools_dialog(visualize_batch_checked)


        # --- Display Update ---
        pygame.display.flip()

        # --- Exit Condition Check ---
        if selected_mode: break

    print(f"--- mode_selection_screen(): Exiting loop. Returning mode={selected_mode} ---")
    # Return loaded game data or new game setup data
    if selected_mode == "LOADED_GAME":
        return selected_mode, loaded_game_data
    elif selected_mode == "BATCH_MODE":
        # Return visualize_batch_checked
        return selected_mode, loaded_game_data # loaded_game_data already contains visualize_batch_checked
    else:
        # Return visualize_batch_checked (though not used here)
        # Add visualize_batch_checked to the tuple for consistency, even if not used for non-batch modes
        return selected_mode, (player_names, human_player, practice_mode, letter_checks, number_checks, use_endgame_solver_checked, use_ai_simulation_checked, practice_state, visualize_batch_checked)






def draw_options_menu(turn, dropdown_open, bag_count):
    """Draw the options menu with dropdown functionality, adding Stop Batch and Specify Rack."""
    global practice_mode, is_batch_running # Access global flags
    options_x = 10; options_rect = pygame.Rect(options_x, OPTIONS_Y, OPTIONS_WIDTH, OPTIONS_HEIGHT)
    hover = options_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, options_rect)
    options_text = button_font.render("Options", True, BLACK); options_text_rect = options_text.get_rect(center=(options_x + OPTIONS_WIDTH // 2, OPTIONS_Y + OPTIONS_HEIGHT // 2)); screen.blit(options_text, options_text_rect)
    dropdown_rects = []
    if dropdown_open:
        dropdown_y = OPTIONS_Y + OPTIONS_HEIGHT
        # Change options if batch is running ---
        if is_batch_running:
            options = ["Stop Batch", "Quit"]
        elif practice_mode == "eight_letter":
            options = ["Give Up", "Main", "Quit"]
        else: # Standard game or other practice modes
            # Added "Specify Rack" ---
            options = ["Pass", "Exchange", "Specify Rack", "Main", "Quit"]
            
        

        for i, option in enumerate(options):
            rect = pygame.Rect(options_x, dropdown_y + i * OPTIONS_HEIGHT, OPTIONS_WIDTH, OPTIONS_HEIGHT)
            # Disable Exchange if bag < 7 (only in non-batch, non-8letter modes)
            is_disabled = (not is_batch_running and
                           practice_mode != "eight_letter" and
                           option == "Exchange" and
                           bag_count < 7)

            if is_disabled:
                pygame.draw.rect(screen, GRAYED_OUT_COLOR, rect)
                text = button_font.render(option, True, BLACK)
                dropdown_rects.append(None) # Mark as non-clickable
            else:
                hover = rect.collidepoint(pygame.mouse.get_pos())
                color = BUTTON_HOVER if hover else DROPDOWN_COLOR
                pygame.draw.rect(screen, color, rect)
                text = button_font.render(option, True, BLACK)
                dropdown_rects.append(rect) # Store clickable rect

            text_rect = text.get_rect(center=(options_x + OPTIONS_WIDTH // 2, dropdown_y + i * OPTIONS_HEIGHT + OPTIONS_HEIGHT // 2))
            screen.blit(text, text_rect)
    return options_rect, dropdown_rects



# End of Part 2



# Part 3 (Includes dialogs, stats, message boxes)
# draw_suggest_button, draw_exchange_dialog, confirm_quit, draw_game_over_dialog, draw_score_row, calculate_moves_per_player, calculate_bingos_per_player, calculate_bingo_avg_per_player, draw_moves_row, draw_avg_score_row, draw_bingos_row, draw_bingo_avg_row, calculate_blanks_per_player, draw_blanks_row, draw_stats_dialog, show_message_dialog, draw_practice_end_dialog (Unchanged from Part 3 provided previously)
def draw_suggest_button():
    """Draw the suggest button."""
    suggest_x = 10 + OPTIONS_WIDTH + BUTTON_GAP; suggest_rect = pygame.Rect(suggest_x, OPTIONS_Y, OPTIONS_WIDTH, OPTIONS_HEIGHT)
    hover = suggest_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, suggest_rect)
    suggest_text = button_font.render("Suggest", True, BLACK); suggest_text_rect = suggest_text.get_rect(center=(suggest_x + OPTIONS_WIDTH // 2, OPTIONS_Y + OPTIONS_HEIGHT // 2)); screen.blit(suggest_text, suggest_text_rect)
    return suggest_rect

def draw_exchange_dialog(rack, selected_tiles):
    """Draw the tile exchange dialog."""
    dialog_width, dialog_height = 400, 200; dialog_x = (WINDOW_WIDTH - dialog_width) // 2; dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height)); pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)
    prompt_text = dialog_font.render("Select tiles to exchange:", True, BLACK); screen.blit(prompt_text, (dialog_x + 10, dialog_y + 10))
    tile_rects = []; rack_display_width = len(rack) * (TILE_WIDTH + TILE_GAP) - TILE_GAP; start_tile_x = dialog_x + (dialog_width - rack_display_width) // 2
    for i, tile in enumerate(rack):
        tile_x = start_tile_x + i * (TILE_WIDTH + TILE_GAP); tile_y = dialog_y + 50; rect = pygame.Rect(tile_x, tile_y, TILE_WIDTH, TILE_HEIGHT); tile_rects.append(rect)
        if tile == ' ':
            center = rect.center; radius = TILE_WIDTH // 2 - 2; pygame.draw.circle(screen, BLACK, center, radius)
            if i in selected_tiles: pygame.draw.circle(screen, SELECTED_TILE_COLOR, center, radius + 2, 2)
            text = font.render('?', True, WHITE); text_rect = text.get_rect(center=center); screen.blit(text, text_rect)
        else: color = SELECTED_TILE_COLOR if i in selected_tiles else GREEN; pygame.draw.rect(screen, color, rect); text = font.render(tile, True, BLACK); screen.blit(text, (tile_x + 5, tile_y + 5))
    total_button_width = BUTTON_WIDTH * 2 + BUTTON_GAP; button_start_x = dialog_x + (dialog_width - total_button_width) // 2; button_y = dialog_y + dialog_height - BUTTON_HEIGHT - 10
    exchange_button_rect = pygame.Rect(button_start_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT); cancel_button_rect = pygame.Rect(button_start_x + BUTTON_WIDTH + BUTTON_GAP, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    hover = exchange_button_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, exchange_button_rect)
    exchange_text = button_font.render("Exchange", True, BLACK); exchange_text_rect = exchange_text.get_rect(center=exchange_button_rect.center); screen.blit(exchange_text, exchange_text_rect)
    hover = cancel_button_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, cancel_button_rect)
    cancel_text = button_font.render("Cancel", True, BLACK); cancel_text_rect = cancel_text.get_rect(center=cancel_button_rect.center); screen.blit(cancel_text, cancel_text_rect)
    return tile_rects, exchange_button_rect, cancel_button_rect

def confirm_quit():
    """Prompt the user to confirm quitting the game."""
    dialog_width, dialog_height = 300, 100; dialog_x = (WINDOW_WIDTH - dialog_width) // 2; dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height)); pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)
    prompt_text = dialog_font.render("Quit game? (Y/N)", True, BLACK); screen.blit(prompt_text, (dialog_x + (dialog_width - prompt_text.get_width()) // 2, dialog_y + 30)); pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y: return True
                elif event.key == pygame.K_n: return False

def draw_game_over_dialog(dialog_x, dialog_y, final_scores, reason, player_names):
    """Draw the game over dialog and return button rectangles."""
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, DIALOG_WIDTH, DIALOG_HEIGHT)); pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, DIALOG_WIDTH, DIALOG_HEIGHT), 2)
    title_text = dialog_font.render(f"Game Over - {reason}", True, BLACK); p1_score_text = ui_font.render(f"{player_names[0]} Score: {final_scores[0]}", True, BLACK)
    p2_name_display = player_names[1] if player_names[1] else "Player 2"; p2_score_text = ui_font.render(f"{p2_name_display} Score: {final_scores[1]}", True, BLACK)
    save_text = button_font.render("Save (S)", True, BLACK); quit_text = button_font.render("Quit (Q)", True, BLACK); replay_text = button_font.render("Replay (R)", True, BLACK); play_again_text = button_font.render("Play Again (P)", True, BLACK); stats_text = button_font.render("Statistics", True, BLACK)
    screen.blit(title_text, (dialog_x + 10, dialog_y + 20)); screen.blit(p1_score_text, (dialog_x + 10, dialog_y + 60)); screen.blit(p2_score_text, (dialog_x + 10, dialog_y + 90))
    first_row_width = 3 * BUTTON_WIDTH + 2 * BUTTON_GAP; first_row_start_x = dialog_x + (DIALOG_WIDTH - first_row_width) // 2; second_row_width = 2 * BUTTON_WIDTH + BUTTON_GAP; second_row_start_x = dialog_x + (DIALOG_WIDTH - second_row_width) // 2
    save_rect = pygame.Rect(first_row_start_x, dialog_y + 150, BUTTON_WIDTH, BUTTON_HEIGHT); quit_rect = pygame.Rect(first_row_start_x + BUTTON_WIDTH + BUTTON_GAP, dialog_y + 150, BUTTON_WIDTH, BUTTON_HEIGHT); replay_rect = pygame.Rect(first_row_start_x + 2 * (BUTTON_WIDTH + BUTTON_GAP), dialog_y + 150, BUTTON_WIDTH, BUTTON_HEIGHT)
    play_again_rect = pygame.Rect(second_row_start_x, dialog_y + 190, BUTTON_WIDTH, BUTTON_HEIGHT); stats_rect = pygame.Rect(second_row_start_x + BUTTON_WIDTH + BUTTON_GAP, dialog_y + 190, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, BUTTON_COLOR, save_rect); pygame.draw.rect(screen, BUTTON_COLOR, quit_rect); pygame.draw.rect(screen, BUTTON_COLOR, replay_rect); pygame.draw.rect(screen, BUTTON_COLOR, play_again_rect); pygame.draw.rect(screen, BUTTON_COLOR, stats_rect)
    screen.blit(save_text, save_text.get_rect(center=save_rect.center)); screen.blit(quit_text, quit_text.get_rect(center=quit_rect.center)); screen.blit(replay_text, replay_text.get_rect(center=replay_rect.center)); screen.blit(play_again_text, play_again_text.get_rect(center=play_again_rect.center)); screen.blit(stats_text, stats_text.get_rect(center=stats_rect.center))
    return save_rect, quit_rect, replay_rect, play_again_rect, stats_rect

def draw_score_row(screen, dialog_x, y_pos, final_scores):
    """Draw the score row for the statistics dialog."""
    score_label = ui_font.render("Score:", True, BLACK)
    p1_score_text = ui_font.render(str(final_scores[0]), True, BLACK)
    p2_score_text = ui_font.render(str(final_scores[1]), True, BLACK)
    screen.blit(score_label, (dialog_x + STATS_LABEL_X_OFFSET, y_pos))
    screen.blit(p1_score_text, (dialog_x + STATS_P1_VAL_X_OFFSET, y_pos))
    screen.blit(p2_score_text, (dialog_x + STATS_P2_VAL_X_OFFSET, y_pos))


def calculate_moves_per_player(move_history):
    """Calculate the number of moves made by each player from move_history."""
    moves_count = {1: 0, 2: 0}
    for move in move_history:
        player = move['player']
        # --- CORRECTED INDENTATION START ---
        if move['move_type'] in ['place', 'pass', 'exchange']:
            moves_count[player] += 1
        # --- CORRECTED INDENTATION END ---
    return moves_count[1], moves_count[2]

def calculate_bingos_per_player(move_history):
    """Calculate the number of bingos played by each player."""
    bingo_count = {1: 0, 2: 0}
    for move in move_history:
        if move['move_type'] == 'place' and move.get('is_bingo', False): player = move['player']; bingo_count[player] += 1
    return bingo_count[1], bingo_count[2]

def calculate_bingo_avg_per_player(move_history):
    """Calculate the average score of bingos for each player."""
    bingo_scores = {1: [], 2: []}
    for move in move_history:
        if move['move_type'] == 'place' and move.get('is_bingo', False): player = move['player']; bingo_scores[player].append(move['score'])
    avg_p1 = sum(bingo_scores[1]) / len(bingo_scores[1]) if bingo_scores[1] else 0.0; avg_p2 = sum(bingo_scores[2]) / len(bingo_scores[2]) if bingo_scores[2] else 0.0
    return avg_p1, avg_p2

def calculate_tiles_per_turn(move_history):
    """
    Calculate the average number of tiles played per turn for each player.
    Only counts 'place' moves.
    """
    tiles_played_count = {1: 0, 2: 0}
    place_moves_count = {1: 0, 2: 0}

    for move in move_history:
        if move['move_type'] == 'place':
            player = move['player']
            place_moves_count[player] += 1
            # Use 'newly_placed' if available, otherwise fallback to 'positions'
            # 'newly_placed' should be more accurate for tiles *actually* played from the rack
            tiles_played = move.get('newly_placed', move.get('positions', []))
            tiles_played_count[player] += len(tiles_played)

    avg_p1 = tiles_played_count[1] / place_moves_count[1] if place_moves_count[1] > 0 else 0.0
    avg_p2 = tiles_played_count[2] / place_moves_count[2] if place_moves_count[2] > 0 else 0.0

    return avg_p1, avg_p2

def calculate_avg_leave(move_history):
    """
    Calculate the average leave value for each player's rack after their turn.
    Uses the 'rack' field from the move history (which stores the rack *before* the move).
    """
    leave_scores_sum = {1: 0.0, 2: 0.0}
    moves_count = {1: 0, 2: 0}

    for move in move_history:
        player = move['player']
        # Only count moves where a rack was recorded (should be all moves)
        if 'rack' in move and move['rack'] is not None:
            moves_count[player] += 1
            # Evaluate the leave *before* the move was made
            # The 'rack' field in move_history stores the rack *before* the move
            leave_score = evaluate_leave(move['rack'])
            leave_scores_sum[player] += leave_score

    avg_p1 = leave_scores_sum[1] / moves_count[1] if moves_count[1] > 0 else 0.0
    avg_p2 = leave_scores_sum[2] / moves_count[2] if moves_count[2] > 0 else 0.0

    return avg_p1, avg_p2

# --- NEW Statistics Drawing Functions ---

def draw_tiles_per_turn_row(screen, dialog_x, y_pos, avg_p1, avg_p2):
    """Draw the average tiles per turn row for the statistics dialog."""
    label = ui_font.render("Tiles Per Turn:", True, BLACK)
    p1_text = ui_font.render(f"{avg_p1:.2f}", True, BLACK)
    p2_text = ui_font.render(f"{avg_p2:.2f}", True, BLACK)
    screen.blit(label, (dialog_x + STATS_LABEL_X_OFFSET, y_pos))
    screen.blit(p1_text, (dialog_x + STATS_P1_VAL_X_OFFSET, y_pos))
    screen.blit(p2_text, (dialog_x + STATS_P2_VAL_X_OFFSET, y_pos))



def draw_avg_leave_row(screen, dialog_x, y_pos, avg_p1, avg_p2):
    """Draw the average leave value row for the statistics dialog."""
    label = ui_font.render("Avg Leave:", True, BLACK)
    p1_text = ui_font.render(f"{avg_p1:.2f}", True, BLACK)
    p2_text = ui_font.render(f"{avg_p2:.2f}", True, BLACK)
    screen.blit(label, (dialog_x + STATS_LABEL_X_OFFSET, y_pos))
    screen.blit(p1_text, (dialog_x + STATS_P1_VAL_X_OFFSET, y_pos))
    screen.blit(p2_text, (dialog_x + STATS_P2_VAL_X_OFFSET, y_pos))



def draw_moves_row(screen, dialog_x, y_pos, moves_p1, moves_p2):
    """Draw the moves row for the statistics dialog."""
    moves_label = ui_font.render("Moves:", True, BLACK)
    p1_moves_text = ui_font.render(str(moves_p1), True, BLACK)
    p2_moves_text = ui_font.render(str(moves_p2), True, BLACK)
    screen.blit(moves_label, (dialog_x + STATS_LABEL_X_OFFSET, y_pos))
    screen.blit(p1_moves_text, (dialog_x + STATS_P1_VAL_X_OFFSET, y_pos))
    screen.blit(p2_moves_text, (dialog_x + STATS_P2_VAL_X_OFFSET, y_pos))


def draw_avg_score_row(screen, dialog_x, y_pos, avg_p1, avg_p2):
    """Draw the average score per move row for the statistics dialog."""
    avg_label = ui_font.render("Avg Score:", True, BLACK)
    p1_avg_text = ui_font.render(f"{avg_p1:.2f}", True, BLACK)
    p2_avg_text = ui_font.render(f"{avg_p2:.2f}", True, BLACK)
    screen.blit(avg_label, (dialog_x + STATS_LABEL_X_OFFSET, y_pos))
    screen.blit(p1_avg_text, (dialog_x + STATS_P1_VAL_X_OFFSET, y_pos))
    screen.blit(p2_avg_text, (dialog_x + STATS_P2_VAL_X_OFFSET, y_pos))


def draw_bingos_row(screen, dialog_x, y_pos, bingos_p1, bingos_p2):
    """Draw the bingos row for the statistics dialog."""
    bingos_label = ui_font.render("Bingos:", True, BLACK)
    p1_bingos_text = ui_font.render(str(bingos_p1), True, BLACK)
    p2_bingos_text = ui_font.render(str(bingos_p2), True, BLACK)
    screen.blit(bingos_label, (dialog_x + STATS_LABEL_X_OFFSET, y_pos))
    screen.blit(p1_bingos_text, (dialog_x + STATS_P1_VAL_X_OFFSET, y_pos))
    screen.blit(p2_bingos_text, (dialog_x + STATS_P2_VAL_X_OFFSET, y_pos))


def draw_bingo_avg_row(screen, dialog_x, y_pos, bingo_avg_p1, bingo_avg_p2, bingos_p1, bingos_p2):
    """Draw the average bingo score row."""
    bingo_avg_label = ui_font.render("Bingo Avg:", True, BLACK)
    p1_text = "N/A" if bingos_p1 == 0 else f"{bingo_avg_p1:.2f}"
    p2_text = "N/A" if bingos_p2 == 0 else f"{bingo_avg_p2:.2f}"
    p1_bingo_avg_text = ui_font.render(p1_text, True, BLACK)
    p2_bingo_avg_text = ui_font.render(p2_text, True, BLACK)
    screen.blit(bingo_avg_label, (dialog_x + STATS_LABEL_X_OFFSET, y_pos))
    screen.blit(p1_bingo_avg_text, (dialog_x + STATS_P1_VAL_X_OFFSET, y_pos))
    screen.blit(p2_bingo_avg_text, (dialog_x + STATS_P2_VAL_X_OFFSET, y_pos))


def calculate_blanks_per_player(move_history):
    """Calculate the number of blanks played by each player."""
    blanks_p1 = 0; blanks_p2 = 0
    for move in move_history:
        if move['move_type'] == 'place':
            player = move['player']
            blanks_count = len(move.get('blanks', set()))
            # --- CORRECTED INDENTATION START ---
            if player == 1:
                blanks_p1 += blanks_count
            elif player == 2:
                blanks_p2 += blanks_count
            # --- CORRECTED INDENTATION END ---
    return blanks_p1, blanks_p2

def draw_blanks_row(screen, dialog_x, y_pos, blanks_p1, blanks_p2):
    """Draw the blanks played row."""
    blanks_label = ui_font.render("Blanks:", True, BLACK)
    p1_blanks_text = ui_font.render(str(blanks_p1), True, BLACK)
    p2_blanks_text = ui_font.render(str(blanks_p2), True, BLACK)
    screen.blit(blanks_label, (dialog_x + STATS_LABEL_X_OFFSET, y_pos))
    screen.blit(p1_blanks_text, (dialog_x + STATS_P1_VAL_X_OFFSET, y_pos))
    screen.blit(p2_blanks_text, (dialog_x + STATS_P2_VAL_X_OFFSET, y_pos))



# Helper function (can be placed above draw_stats_dialog or inside if preferred)
def get_word_index(word, word_list):
    """Finds the 1-based index of a word in a list, case-insensitive."""
    try:
        # Convert both word and list items to uppercase for case-insensitive search
        return word_list.index(word.upper()) + 1
    except ValueError:
        return None # Word not found in the list



def draw_stats_dialog(dialog_x, dialog_y, player_names, final_scores, tiles, scroll_offset):
    """Draw the complete statistics dialog with scrolling, including quadrant counts and luck factor."""
    # --- Fixed Dialog Dimensions ---
    stats_dialog_width = 480
    stats_dialog_height = 600 # Fixed height, content will scroll if needed

    # --- Load Word Lists (same as before) ---
    seven_letter_words = []
    eight_letter_words = []
    try:
        with open("7-letter-list.txt", "r") as f:
            seven_letter_words = [line.strip().upper() for line in f]
    except FileNotFoundError:
        print("Warning: 7-letter-list.txt not found for stats dialog.")
    try:
        with open("8-letter-list.txt", "r") as f:
            eight_letter_words = [line.strip().upper() for line in f]
    except FileNotFoundError:
        print("Warning: 8-letter-list.txt not found for stats dialog.")

    # --- Collect Bingo Data (same as before) ---
    p1_bingos_data = []
    p2_bingos_data = []
    for move in move_history:
        if move.get('is_bingo', False):
            player = move['player']
            word = move.get('word', 'N/A').upper() # Use uppercase for lookup
            score = move.get('score', 0)
            word_len = len(word)
            index = None
            if word_len == 7 and seven_letter_words:
                index = get_word_index(word, seven_letter_words)
            elif word_len == 8 and eight_letter_words:
                index = get_word_index(word, eight_letter_words)
            bingo_info = {'word': word, 'score': score, 'index': index, 'len': word_len}
            if player == 1: p1_bingos_data.append(bingo_info)
            elif player == 2: p2_bingos_data.append(bingo_info)

    # --- Calculate Quadrant Counts ---
    quad_counts = calculate_quadrant_counts(move_history) # Uses new logic

    # --- Calculate Luck Factor (by summing stored values) ---
    luck_p1 = 0.0
    luck_p2 = 0.0
    for move in move_history:
        player = move.get('player')
        luck = move.get('luck_factor', 0.0) # Get stored luck
        if player == 1:
            luck_p1 += luck
        elif player == 2:
            luck_p2 += luck
    # --- End Luck Calculation ---

    # --- Calculate Content Dimensions ---
    padding = 10
    title_height = 30 # Approx height for title
    header_height = 40 # Player names header
    button_area_height = BUTTON_HEIGHT + padding * 2
    line_height = 25 # Height per stat/bingo line
    bingo_font = pygame.font.SysFont("Arial", 18) # Font for bingo lines

    # Calculate total height needed for ALL content
    fixed_stats_rows = 9 # Score, Moves, Avg Score, Tiles/Turn, Bingos Count, Bingo Avg, Blanks, Avg Leave, Luck Factor
    quadrant_rows = 3 # Header + 2 rows for Q2/Q1 and Q3/Q4
    p1_bingo_lines = len(p1_bingos_data)
    p2_bingo_lines = len(p2_bingos_data)
    bingo_header_lines = 0
    if p1_bingo_lines > 0: bingo_header_lines += 1
    if p2_bingo_lines > 0: bingo_header_lines += 1

    total_content_height = (title_height + header_height +
                           (fixed_stats_rows * line_height) +
                           (quadrant_rows * line_height) + # Use updated quadrant row count
                           ((p1_bingo_lines + p2_bingo_lines + bingo_header_lines) * line_height) +
                           padding * 5) # Add some padding between sections

    # --- Create Content Surface ---
    content_surface_width = stats_dialog_width - padding * 2
    content_surface = pygame.Surface((content_surface_width, total_content_height))
    content_surface.fill(DIALOG_COLOR) # Fill with dialog background

    # --- Draw Content onto Surface ---
    y_on_surface = padding # Start drawing from top of surface

    # Title
    title_text = dialog_font.render("Game Statistics", True, BLACK)
    content_surface.blit(title_text, (padding, y_on_surface))
    y_on_surface += title_height + padding

    # Player Names Header
    p1_name_display = player_names[0] if player_names[0] else "Player 1"
    p2_name_display = player_names[1] if player_names[1] else "Player 2"
    p1_name_text = ui_font.render(p1_name_display, True, BLACK)
    p2_name_text = ui_font.render(p2_name_display, True, BLACK)
    # Adjust x-offsets relative to the content surface width
    p1_x_offset_on_surf = STATS_P1_VAL_X_OFFSET - STATS_LABEL_X_OFFSET # Relative offset
    p2_x_offset_on_surf = STATS_P2_VAL_X_OFFSET - STATS_LABEL_X_OFFSET # Relative offset
    content_surface.blit(p1_name_text, (p1_x_offset_on_surf, y_on_surface))
    content_surface.blit(p2_name_text, (p2_x_offset_on_surf, y_on_surface))
    y_on_surface += header_height # Approx height for this header row

    # Helper to draw stat rows onto the surface (accepts format_spec)
    def draw_row_on_surface(label_text, val1_text, val2_text, y_pos, format_spec="{:.2f}"):
        label_surf = ui_font.render(label_text, True, BLACK)
        # Apply formatting only if it's likely a number (simple check)
        try:
            # Attempt to format val1 and val2 using the specifier
            val1_str = format_spec.format(float(val1_text)) if isinstance(val1_text, (int, float)) else str(val1_text)
            val2_str = format_spec.format(float(val2_text)) if isinstance(val2_text, (int, float)) else str(val2_text)
        except (ValueError, TypeError):
            # Fallback to string conversion if formatting fails
            val1_str = str(val1_text)
            val2_str = str(val2_text)

        val1_surf = ui_font.render(val1_str, True, BLACK)
        val2_surf = ui_font.render(val2_str, True, BLACK)

        content_surface.blit(label_surf, (padding, y_pos))
        content_surface.blit(val1_surf, (p1_x_offset_on_surf, y_pos))
        content_surface.blit(val2_surf, (p2_x_offset_on_surf, y_pos))


    # Helper to draw quadrant rows onto the surface
    def draw_quad_row(label1, val1, label2, val2, y_pos):
         text1 = f"{label1}: {val1}"
         text2 = f"{label2}: {val2}"
         surf1 = ui_font.render(text1, True, BLACK)
         surf2 = ui_font.render(text2, True, BLACK)
         # Position them side-by-side
         x_pos1 = padding + 10 # Indent slightly
         x_pos2 = padding + content_surface_width // 2
         content_surface.blit(surf1, (x_pos1, y_pos))
         content_surface.blit(surf2, (x_pos2, y_pos))

    # --- Draw Fixed Stats ---
    moves_p1, moves_p2 = calculate_moves_per_player(move_history)
    avg_p1 = final_scores[0] / moves_p1 if moves_p1 > 0 else 0.00
    avg_p2 = final_scores[1] / moves_p2 if moves_p2 > 0 else 0.00
    bingos_p1_count = len(p1_bingos_data)
    bingos_p2_count = len(p2_bingos_data)
    bingo_avg_p1, bingo_avg_p2 = calculate_bingo_avg_per_player(move_history)
    blanks_p1, blanks_p2 = calculate_blanks_per_player(move_history)
    tiles_per_turn_p1, tiles_per_turn_p2 = calculate_tiles_per_turn(move_history)
    avg_leave_p1, avg_leave_p2 = calculate_avg_leave(move_history)

    draw_row_on_surface("Score:", final_scores[0], final_scores[1], y_on_surface, format_spec="{}"); y_on_surface += line_height # No decimals for score
    draw_row_on_surface("Moves:", moves_p1, moves_p2, y_on_surface, format_spec="{}"); y_on_surface += line_height # No decimals for moves
    draw_row_on_surface("Avg Score:", avg_p1, avg_p2, y_on_surface); y_on_surface += line_height
    draw_row_on_surface("Tiles Per Turn:", tiles_per_turn_p1, tiles_per_turn_p2, y_on_surface); y_on_surface += line_height
    draw_row_on_surface("Bingos:", bingos_p1_count, bingos_p2_count, y_on_surface, format_spec="{}"); y_on_surface += line_height # No decimals for count
    p1_bingo_avg_str = "N/A" if bingos_p1_count == 0 else f"{bingo_avg_p1:.2f}"
    p2_bingo_avg_str = "N/A" if bingos_p2_count == 0 else f"{bingo_avg_p2:.2f}"
    draw_row_on_surface("Bingo Avg:", p1_bingo_avg_str, p2_bingo_avg_str, y_on_surface, format_spec="{}"); y_on_surface += line_height # Already formatted
    draw_row_on_surface("Blanks:", blanks_p1, blanks_p2, y_on_surface, format_spec="{}"); y_on_surface += line_height # No decimals for count
    draw_row_on_surface("Avg Leave:", avg_leave_p1, avg_leave_p2, y_on_surface); y_on_surface += line_height
    # --- Draw Luck Factor Row using the helper ---
    draw_row_on_surface("Luck Factor:", luck_p1, luck_p2, y_on_surface, format_spec="{:+.2f}"); y_on_surface += line_height # Use sign formatting
    # --- END Draw Luck Factor Row ---

    # --- Draw Quadrant Counts ---
    quad_header_surf = ui_font.render("Quadrant Tile Counts:", True, BLACK)
    content_surface.blit(quad_header_surf, (padding, y_on_surface)); y_on_surface += line_height
    # Draw Q2 (TL) and Q1 (TR)
    draw_quad_row("Q2 (TL)", quad_counts["Q2"], "Q1 (TR)", quad_counts["Q1"], y_on_surface); y_on_surface += line_height
    # Draw Q3 (BL) and Q4 (BR)
    draw_quad_row("Q3 (BL)", quad_counts["Q3"], "Q4 (BR)", quad_counts["Q4"], y_on_surface); y_on_surface += line_height

    # --- Draw Bingo Lists ---
    if p1_bingos_data:
        p1_bingo_header = ui_font.render(f"{p1_name_display} Bingos:", True, BLACK)
        content_surface.blit(p1_bingo_header, (padding, y_on_surface))
        y_on_surface += line_height
        for bingo in p1_bingos_data:
            word, score, index, length = bingo['word'], bingo['score'], bingo['index'], bingo['len']
            prob_text = ""
            if (length == 7 or length == 8) and index is not None: prob_text = f" Prob: {index}"
            elif length > 8: prob_text = ""
            else: prob_text = " (N/L)"
            bingo_line = f"  {word} ({score} pts){prob_text}"
            bingo_surf = bingo_font.render(bingo_line, True, BLACK)
            content_surface.blit(bingo_surf, (padding + 10, y_on_surface))
            y_on_surface += line_height

    if p2_bingos_data:
        p2_bingo_header = ui_font.render(f"{p2_name_display} Bingos:", True, BLACK)
        content_surface.blit(p2_bingo_header, (padding, y_on_surface))
        y_on_surface += line_height
        for bingo in p2_bingos_data:
            word, score, index, length = bingo['word'], bingo['score'], bingo['index'], bingo['len']
            prob_text = ""
            if (length == 7 or length == 8) and index is not None: prob_text = f" Prob: {index}"
            elif length > 8: prob_text = ""
            else: prob_text = " (N/L)"
            bingo_line = f"  {word} ({score} pts){prob_text}"
            bingo_surf = bingo_font.render(bingo_line, True, BLACK)
            content_surface.blit(bingo_surf, (padding + 10, y_on_surface))
            y_on_surface += line_height

    # --- Draw Dialog Background and Border on Main Screen ---
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, stats_dialog_width, stats_dialog_height))
    pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, stats_dialog_width, stats_dialog_height), 2)

    # --- Blit Visible Portion of Content Surface ---
    content_area_y = dialog_y + padding # Top of the scrollable area
    content_area_height = stats_dialog_height - padding * 2 - button_area_height # Height available for scrolling
    visible_area_on_surface = pygame.Rect(0, scroll_offset, content_surface_width, content_area_height)
    screen.blit(content_surface, (dialog_x + padding, content_area_y), visible_area_on_surface)

    # --- Draw OK Button ---
    ok_button_y = dialog_y + stats_dialog_height - button_area_height + padding
    ok_button_rect = pygame.Rect(dialog_x + stats_dialog_width - BUTTON_WIDTH - padding,
                                 ok_button_y,
                                 BUTTON_WIDTH, BUTTON_HEIGHT)
    hover = ok_button_rect.collidepoint(pygame.mouse.get_pos())
    color = BUTTON_HOVER if hover else BUTTON_COLOR
    pygame.draw.rect(screen, color, ok_button_rect)
    ok_text = button_font.render("OK", True, BLACK)
    ok_text_rect = ok_text.get_rect(center=ok_button_rect.center)
    screen.blit(ok_text, ok_text_rect)

    # Return the OK button rect and the total content height for scroll calculation
    return ok_button_rect, total_content_height




def calculate_quadrant_counts(move_history):
    """
    Calculates the number of tiles played in each quadrant (inclusive counting).
    Q1=TopRight, Q2=TopLeft, Q3=BottomLeft, Q4=BottomRight.
    Center tile (7,7) is ignored. Tiles on center row/col are counted in adjacent quadrants.
    """
    counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    center_r, center_c = 7, 7

    for move in move_history:
        if move['move_type'] == 'place':
            new_tiles = move.get('newly_placed', move.get('positions', []))
            for r, c, _ in new_tiles:
                if r == center_r and c == center_c:
                    continue # Ignore exact center

                # Check membership for each quadrant inclusively
                is_q1 = (r <= center_r and c >= center_c)
                is_q2 = (r <= center_r and c <= center_c)
                is_q3 = (r >= center_r and c <= center_c)
                is_q4 = (r >= center_r and c >= center_c)

                if is_q1: counts["Q1"] += 1
                if is_q2: counts["Q2"] += 1
                if is_q3: counts["Q3"] += 1
                if is_q4: counts["Q4"] += 1
    return counts



def show_message_dialog(message, title="Message"):
    """
    Display a general message dialog with an OK button, wrapping text
    and dynamically adjusting height.
    """
    # --- MODIFICATION START: Dynamic Height Calculation ---
    base_dialog_width = 400
    min_dialog_height = 150 # Minimum height
    padding = 10 # Padding around elements
    line_spacing = 5 # Extra space between lines
    button_area_height = BUTTON_HEIGHT + padding * 2 # Space for OK button

    # 1. Wrap text first to determine needed lines
    words = message.split(' ')
    lines = []
    current_line = ''
    # Use a slightly reduced width for text wrapping calculation to avoid edge cases
    max_line_width = base_dialog_width - padding * 2 - 10

    for word in words:
        # Handle newline characters explicitly
        if '\n' in word:
            parts = word.split('\n')
            for i, part in enumerate(parts):
                if not part: # Handle consecutive newlines or newline at start/end
                    if current_line: # Add previous line if any
                         lines.append(current_line.strip())
                    lines.append("") # Add empty line for the newline itself
                    current_line = ""
                    continue

                test_line = current_line + part + ' '
                if ui_font.size(test_line)[0] < max_line_width:
                    current_line = test_line
                else:
                    if current_line: # Add the line before the current part
                        lines.append(current_line.strip())
                    # Start new line with the current part, handle if it's too long itself
                    if ui_font.size(part + ' ')[0] < max_line_width:
                         current_line = part + ' '
                    else: # Word itself is too long, just put it on its own line (will overflow visually)
                         lines.append(part)
                         current_line = "" # Start fresh after the long word

                # Add empty line if this part was followed by a newline (except for the last part)
                if i < len(parts) - 1:
                     if current_line: # Add the line formed by the part first
                          lines.append(current_line.strip())
                     lines.append("") # Add the blank line for the newline
                     current_line = ""

        else: # Word does not contain newline
            test_line = current_line + word + ' '
            if ui_font.size(test_line)[0] < max_line_width:
                current_line = test_line
            else:
                if current_line: # Add previous line if any
                    lines.append(current_line.strip())
                # Start new line with the current word, handle if it's too long itself
                if ui_font.size(word + ' ')[0] < max_line_width:
                     current_line = word + ' '
                else: # Word itself is too long
                     lines.append(word)
                     current_line = ""

    if current_line: # Add the last line
        lines.append(current_line.strip())

    # 2. Calculate required height
    title_height = dialog_font.get_linesize()
    text_height = len(lines) * ui_font.get_linesize() + max(0, len(lines) - 1) * line_spacing
    required_height = title_height + text_height + button_area_height + padding * 3 # Title + Text + Button Area + Paddings

    dialog_height = max(min_dialog_height, required_height)
    dialog_width = base_dialog_width # Keep width fixed for now

    # 3. Calculate dialog position
    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
    # --- MODIFICATION END ---

    # Drawing uses calculated dimensions
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
    pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

    title_surf = dialog_font.render(title, True, BLACK)
    screen.blit(title_surf, (dialog_x + padding, dialog_y + padding))

    # Draw the wrapped text lines
    y_offset = dialog_y + padding + title_height + padding # Start below title
    for line in lines:
        text = ui_font.render(line, True, BLACK)
        screen.blit(text, (dialog_x + padding, y_offset))
        y_offset += ui_font.get_linesize() + line_spacing

    # OK Button position adjusted to new height
    ok_button_rect = pygame.Rect(dialog_x + dialog_width - BUTTON_WIDTH - padding,
                                 dialog_y + dialog_height - BUTTON_HEIGHT - padding,
                                 BUTTON_WIDTH, BUTTON_HEIGHT)

    hover = ok_button_rect.collidepoint(pygame.mouse.get_pos())
    color = BUTTON_HOVER if hover else BUTTON_COLOR
    pygame.draw.rect(screen, color, ok_button_rect)
    ok_text = button_font.render("OK", True, BLACK)
    ok_text_rect = ok_text.get_rect(center=ok_button_rect.center)
    screen.blit(ok_text, ok_text_rect)

    pygame.display.flip() # Update display to show the dialog immediately

    # Event loop for the dialog
    dialog_running = True
    while dialog_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                 if ok_button_rect.collidepoint(event.pos):
                     dialog_running = False # Exit dialog loop
                     return # Return control to the main loop
            elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                     dialog_running = False # Exit dialog loop
                     return # Return control to the main loop

        # Keep drawing the dialog while waiting for input
        # (Redrawing everything might be overkill, but ensures it stays visible if covered/revealed)
        # Alternatively, just flip without redrawing if performance is an issue.
        pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)
        screen.blit(title_surf, (dialog_x + padding, dialog_y + padding))
        y_offset = dialog_y + padding + title_height + padding
        for line in lines:
            text = ui_font.render(line, True, BLACK)
            screen.blit(text, (dialog_x + padding, y_offset))
            y_offset += ui_font.get_linesize() + line_spacing
        hover = ok_button_rect.collidepoint(pygame.mouse.get_pos())
        color = BUTTON_HOVER if hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, ok_button_rect)
        screen.blit(ok_text, ok_text_rect)
        pygame.display.flip()
        pygame.time.Clock().tick(30) # Limit frame rate in dialog loop


def draw_practice_end_dialog(message):
    """
    Draw the dialog shown at the end of a practice puzzle,
    dynamically adjusting height based on the message content.
    """
    # --- MODIFICATION START: Dynamic Height Calculation ---
    base_dialog_width = 400
    min_dialog_height = 180 # Keep original minimum height
    padding = 15 # Padding around elements
    line_spacing = 5 # Extra space between lines
    button_area_height = BUTTON_HEIGHT + padding * 2 # Space for buttons + padding

    # 1. Wrap text first to determine needed lines
    words = message.split(' ')
    lines = []
    current_line = ''
    # Use a slightly reduced width for text wrapping calculation
    max_line_width = base_dialog_width - padding * 2 - 10

    for word in words:
        # Handle newline characters explicitly
        if '\n' in word:
            parts = word.split('\n')
            for i, part in enumerate(parts):
                if not part:
                    if current_line: lines.append(current_line.strip())
                    lines.append("")
                    current_line = ""
                    continue
                test_line = current_line + part + ' '
                if ui_font.size(test_line)[0] < max_line_width:
                    current_line = test_line
                else:
                    if current_line: lines.append(current_line.strip())
                    if ui_font.size(part + ' ')[0] < max_line_width:
                         current_line = part + ' '
                    else:
                         lines.append(part)
                         current_line = ""
                if i < len(parts) - 1:
                     if current_line: lines.append(current_line.strip())
                     lines.append("")
                     current_line = ""
        else: # Word does not contain newline
            test_line = current_line + word + ' '
            if ui_font.size(test_line)[0] < max_line_width:
                current_line = test_line
            else:
                if current_line: lines.append(current_line.strip())
                if ui_font.size(word + ' ')[0] < max_line_width:
                     current_line = word + ' '
                else:
                     lines.append(word)
                     current_line = ""
    if current_line: # Add the last line
        lines.append(current_line.strip())

    # 2. Calculate required height
    # No explicit title, start text height calculation directly
    text_height = len(lines) * ui_font.get_linesize() + max(0, len(lines) - 1) * line_spacing
    required_height = text_height + button_area_height + padding * 2 # Text + Button Area + Top/Bottom Padding

    dialog_height = max(min_dialog_height, required_height)
    dialog_width = base_dialog_width # Keep width fixed

    # 3. Calculate dialog position
    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
    # --- MODIFICATION END ---

    # Drawing uses calculated dimensions
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
    pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

    # Draw the wrapped text lines starting near the top
    y_offset = dialog_y + padding
    for line in lines:
        text_surf = ui_font.render(line, True, BLACK)
        # Center text horizontally within the padded area
        text_rect = text_surf.get_rect(centerx=dialog_x + dialog_width // 2)
        text_rect.top = y_offset # Align top
        screen.blit(text_surf, text_rect)
        # screen.blit(text_surf, (dialog_x + padding, y_offset)) # Original left-align
        y_offset += ui_font.get_linesize() + line_spacing

    # Buttons positioned relative to the new dynamic bottom
    button_y = dialog_y + dialog_height - BUTTON_HEIGHT - padding
    total_button_width = 3 * BUTTON_WIDTH + 2 * BUTTON_GAP
    button_start_x = dialog_x + (dialog_width - total_button_width) // 2

    play_again_rect = pygame.Rect(button_start_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    main_menu_rect = pygame.Rect(button_start_x + BUTTON_WIDTH + BUTTON_GAP, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    quit_rect = pygame.Rect(button_start_x + 2 * (BUTTON_WIDTH + BUTTON_GAP), button_y, BUTTON_WIDTH, BUTTON_HEIGHT)

    # Draw buttons
    pygame.draw.rect(screen, BUTTON_COLOR, play_again_rect)
    pygame.draw.rect(screen, BUTTON_COLOR, main_menu_rect)
    pygame.draw.rect(screen, BUTTON_COLOR, quit_rect)

    play_again_text = button_font.render("Play Again", True, BLACK)
    main_menu_text = button_font.render("Main Menu", True, BLACK)
    quit_text = button_font.render("Quit", True, BLACK)

    screen.blit(play_again_text, play_again_text.get_rect(center=play_again_rect.center))
    screen.blit(main_menu_text, main_menu_text.get_rect(center=main_menu_rect.center))
    screen.blit(quit_text, quit_text.get_rect(center=quit_rect.center))

    return play_again_rect, main_menu_rect, quit_rect



# End of Part 3

# Part 4 (Includes game logic helpers: get_words_played, remaining tiles, validation, anchors)
# get_words_played, get_remaining_tiles, draw_remaining_tiles, draw_arrow, is_valid_play, get_anchor_points (Unchanged from Part 4 provided previously)
def get_words_played(word_positions, tiles):
    """Get all word strings formed by a play based on tile positions."""
    if not word_positions: return []
    words_found = set(); rows_involved_list = []; cols_involved_list = []; valid_positions = True
    for item in word_positions:
        if isinstance(item, tuple) and len(item) >= 3: rows_involved_list.append(item[0]); cols_involved_list.append(item[1])
        else: print(f"Error in get_words_played: Invalid item format: {item}"); valid_positions = False; break
    if not valid_positions: return []
    rows_involved = set(rows_involved_list); cols_involved = set(cols_involved_list)
    if len(rows_involved) == 1: # Horizontal Check
        r = rows_involved.pop(); min_c = min(cols_involved_list); max_c = max(cols_involved_list)
        while min_c > 0 and tiles[r][min_c - 1]: min_c -= 1
        while max_c < GRID_SIZE - 1 and tiles[r][max_c + 1]: max_c += 1
        word_h = "".join(tiles[r][c] for c in range(min_c, max_c + 1) if tiles[r][c])
        if len(word_h) > 1: words_found.add(word_h)
    if len(cols_involved) == 1: # Vertical Check
        c = cols_involved.pop(); min_r = min(rows_involved_list); max_r = max(rows_involved_list)
        while min_r > 0 and tiles[min_r - 1][c]: min_r -= 1
        while max_r < GRID_SIZE - 1 and tiles[max_r + 1][c]: max_r += 1
        word_v = "".join(tiles[r][c] for r in range(min_r, max_r + 1) if tiles[r][c])
        if len(word_v) > 1: words_found.add(word_v)
    for r_new, c_new, _ in word_positions: # Cross Checks
        if len(rows_involved) == 1: # Vertical Cross
            min_r_cross = r_new; max_r_cross = r_new;
            while min_r_cross > 0 and tiles[min_r_cross - 1][c_new]: min_r_cross -= 1;
            while max_r_cross < GRID_SIZE - 1 and tiles[max_r_cross + 1][c_new]: max_r_cross += 1
            if max_r_cross > min_r_cross: cross_word_v = "".join(tiles[r][c_new] for r in range(min_r_cross, max_r_cross + 1) if tiles[r][c_new]);
            if len(cross_word_v) > 1: words_found.add(cross_word_v)
        if len(cols_involved) == 1: # Horizontal Cross
            min_c_cross = c_new; max_c_cross = c_new;
            while min_c_cross > 0 and tiles[r_new][min_c_cross - 1]: min_c_cross -= 1;
            while max_c_cross < GRID_SIZE - 1 and tiles[r_new][max_c_cross + 1]: max_c_cross += 1
            if max_c_cross > min_c_cross: cross_word_h = "".join(tiles[r_new][c] for c in range(min_c_cross, max_c_cross + 1) if tiles[r_new][c]);
            if len(cross_word_h) > 1: words_found.add(cross_word_h)
    return list(words_found)



# python
# MODIFIED: Use board_tile_counts instead of scanning board
def get_remaining_tiles(rack, board_tile_counts): # Removed tiles, blanks; Added board_tile_counts
    """
    Calculate the remaining tiles (bag + opponent rack) using pre-calculated board counts.
    """
    # Start with full distribution
    remaining_counts = Counter({letter: count for letter, (count, _) in TILE_DISTRIBUTION.items()})

    # Subtract current player's rack
    rack_counter = Counter(rack)
    remaining_counts.subtract(rack_counter)

    # Subtract tiles on the board
    remaining_counts.subtract(board_tile_counts)

    # Ensure counts don't go below zero (though they shouldn't if logic is correct)
    final_remaining = {tile: max(0, count) for tile, count in remaining_counts.items()}

    return final_remaining



def draw_remaining_tiles(remaining, turn):
    """Draw the list of remaining tiles by repeating letters."""
    global practice_mode
    if practice_mode == "eight_letter": return # Don't draw remaining tiles in this mode
    remaining_x = min(BOARD_SIZE + 70, WINDOW_WIDTH - 200);
    if remaining_x < BOARD_SIZE + 10: remaining_x = BOARD_SIZE + 10
    title_text = ui_font.render(f"P{turn}'s Unseen Tiles:", True, BLACK); screen.blit(title_text, (remaining_x, 10))
    y_pos = 40; col_width = 60; max_y = WINDOW_HEIGHT - 100; sorted_letters = sorted(remaining.keys()); current_col_x = remaining_x; items_in_col = 0; max_items_per_col = (max_y - y_pos) // 20
    if max_items_per_col <= 0: max_items_per_col = 1
    last_drawn_y = 40 # Initialize with starting y_pos in case list is empty
    last_drawn_col_x = remaining_x # Track the column x of the last item

    for letter in sorted_letters:
        count = remaining.get(letter, 0)
        if count > 0:
            display_letter = "?" if letter == " " else letter
            text_str = display_letter * count
            text = tile_count_font.render(text_str, True, BLACK)
            if items_in_col >= max_items_per_col:
                y_pos = 40 # Reset y_pos for the new column
                current_col_x += col_width
                items_in_col = 0
            screen.blit(text, (current_col_x, y_pos))
            last_drawn_y = y_pos # Update the last drawn y position in the current column
            last_drawn_col_x = current_col_x # Update the column x
            y_pos += 20; items_in_col += 1

    # Calculate summary position directly below the last drawn item's position
    # Add the height of one line (20) to get the position below it, plus a small gap (e.g., 5)
    summary_y_start = last_drawn_y + 20 + 5
    # Ensure it doesn't go off screen
    summary_y_start = min(summary_y_start, max_y + 40) # Use max_y for clamping

    total_tiles = sum(remaining.values()); vowels = sum(remaining.get(letter, 0) for letter in 'AEIOU'); consonants = sum(remaining.get(letter, 0) for letter in remaining if letter.isalpha() and letter not in 'AEIOU '); blanks_rem = remaining.get(' ', 0)
    text1 = tile_count_font.render(f"Tiles remaining: {total_tiles}", True, BLACK); text2 = tile_count_font.render(f"V: {vowels}  C: {consonants}  B: {blanks_rem}", True, BLACK)

    # Align summary text x-coordinate with the start of the list columns
    summary_x = remaining_x
    screen.blit(text1, (summary_x, summary_y_start));
    screen.blit(text2, (summary_x, summary_y_start + 20)) # Draw second line below first




def draw_arrow(row, col, direction):
    """Draw an arrow indicating play direction."""
    center_x = 40 + col * SQUARE_SIZE + SQUARE_SIZE // 2; center_y = 40 + row * SQUARE_SIZE + SQUARE_SIZE // 2; arrow_length = SQUARE_SIZE * 0.4; arrow_width = SQUARE_SIZE * 0.2
    if direction == "right": pygame.draw.line(screen, ARROW_COLOR, (center_x - arrow_length / 2, center_y), (center_x + arrow_length / 2, center_y), 3); pygame.draw.polygon(screen, ARROW_COLOR, [(center_x + arrow_length / 2, center_y - arrow_width / 2), (center_x + arrow_length / 2, center_y + arrow_width / 2), (center_x + arrow_length / 2 + arrow_width, center_y)])
    elif direction == "down": pygame.draw.line(screen, ARROW_COLOR, (center_x, center_y - arrow_length / 2), (center_x, center_y + arrow_length / 2), 3); pygame.draw.polygon(screen, ARROW_COLOR, [(center_x - arrow_width / 2, center_y + arrow_length / 2), (center_x + arrow_width / 2, center_y + arrow_length / 2), (center_x, center_y + arrow_length / 2 + arrow_width)])













# End of Part 4

# python
# MODIFIED: 01APR24 - Implementing 8-Letter Bingo Practice Mode Fixes (Part 5 - Corrected Attribute Name)

# Part 5

# --- NEW Move Generation (Based on kamilmielnik/scrabble-solver principles) ---
# --- NEW Move Generation (Based on kamilmielnik/scrabble-solver principles - REVISED) ---
















# Function to Replace: generate_all_moves_gaddag
# REASON: Revert initial calls to pass copies of initial_tiles list

def generate_all_moves_gaddag(rack, tiles, board, blanks, gaddag_root):
    """
    Generates ALL valid Scrabble moves using GADDAG traversal.
    Converts rack Counter to C array (using numpy.array with dtype=np.intc)
    before calling Cython. Passes the GADDAG root node to Cython.
    Passes copies of initial tile lists.
    """
    # Removed global DAWG access - assume DAWG is accessible if needed by cross_check
    global gaddag_loading_status
    # Ensure DAWG is accessible for cross_check_sets calculation below if needed
    # If DAWG is not loaded globally, this part might fail later.
    # Consider passing DAWG as an argument if it's not reliably global.
    global DAWG
    from scrabble_helpers import get_anchor_points
    # Import the Cython function (assuming the import block exists near the top)
    # from gaddag_cython import _gaddag_traverse as _gaddag_traverse_cython

    # Check GADDAG status
    if gaddag_loading_status != 'loaded' or gaddag_root is None:
        print("Error: GADDAG not loaded or has no root. Cannot generate moves.")
        return []
    # Removed DAWG check - rely on cross_check logic below


    # Variable Initializations (unchanged)
    all_found_moves = []
    unique_move_signatures = set()
    is_first_play = sum(1 for row in tiles for t in row if t) == 0
    anchors = get_anchor_points(tiles, is_first_play)
    original_tiles_state = [row[:] for row in tiles]
    full_rack_size = len(rack)

    # --- Convert Python Counter to NumPy C array ---
    rack_counts_py = Counter(rack)
    rack_counts_c_arr = np.zeros(27, dtype=np.intc) # Use np.intc (platform C int)
    for i in range(26):
        letter = chr(ord('A') + i)
        rack_counts_c_arr[i] = rack_counts_py.get(letter, 0)
    rack_counts_c_arr[26] = rack_counts_py.get(' ', 0) # Index 26 for blank
    # --- END Convert ---

    # Precompute Cross-Check Sets (using the global DAWG object)
    cross_check_sets = {}
    # --- Ensure DAWG is available for this block ---
    if DAWG is None:
        print("Error: DAWG object is None, cannot compute cross_check_sets.")
        return [] # Cannot proceed without DAWG for cross-checks
    # --- End DAWG check ---
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not tiles[r][c]:
                # Vertical check
                up_word = ""; rr = r - 1
                while rr >= 0 and tiles[rr][c]: up_word = tiles[rr][c] + up_word; rr -= 1
                down_word = ""; rr = r + 1
                while rr < GRID_SIZE and tiles[rr][c]: down_word += tiles[rr][c]; rr += 1
                allowed_letters_v = set()
                if not up_word and not down_word: allowed_letters_v = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
                else:
                    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        # Uses global DAWG implicitly here
                        if DAWG.search(up_word + letter + down_word): allowed_letters_v.add(letter)
                    if allowed_letters_v: allowed_letters_v.add(' ')

                # Horizontal check
                left_word = ""; cc = c - 1
                while cc >= 0 and tiles[r][cc]: left_word = tiles[r][cc] + left_word; cc -= 1
                right_word = ""; cc = c + 1
                while cc < GRID_SIZE and tiles[r][cc]: right_word += tiles[r][cc]; cc += 1
                allowed_letters_h = set()
                if not left_word and not right_word: allowed_letters_h = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
                else:
                    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                         # Uses global DAWG implicitly here
                        if DAWG.search(left_word + letter + right_word): allowed_letters_h.add(letter)
                    if allowed_letters_h: allowed_letters_h.add(' ')
                cross_check_sets[(r, c)] = {'V': allowed_letters_v, 'H': allowed_letters_h}


    # --- Initiate Traversal using NumPy array ---
    processed_adjacent_starts = set()
    initial_rack_counts_c_copy = rack_counts_c_arr.copy() # numpy copy
    # --- gaddag_root is already passed as argument to this function ---

    for r_anchor, c_anchor in anchors:
        anchor_pos = (r_anchor, c_anchor)
        allowed_h = cross_check_sets.get(anchor_pos, {}).get('H', set())
        allowed_v = cross_check_sets.get(anchor_pos, {}).get('V', set())

        # --- Strategy 1 ---
        for tile_letter, count in list(rack_counts_py.items()):
            if count > 0 and tile_letter != ' ':
                if tile_letter in gaddag_root.children:
                    next_node = gaddag_root.children[tile_letter]
                    next_rack_counts_c_arr_s1 = rack_counts_c_arr.copy() # numpy copy
                    letter_idx_s1 = ord(tile_letter) - ord('A')
                    next_rack_counts_c_arr_s1[letter_idx_s1] -= 1
                    # --- REVERT: Create list copy for call ---
                    initial_tiles = [(r_anchor, c_anchor, tile_letter, False, True)]
                    # --- END REVERT ---

                    if tile_letter in allowed_v:
                         # --- REVERT: Pass list copy ---
                         _gaddag_traverse_cython(anchor_pos, next_rack_counts_c_arr_s1, tiles, board, blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, 'H', all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size)
                         # --- END REVERT ---

                    if tile_letter in allowed_h:
                         # --- REVERT: Pass list copy ---
                         _gaddag_traverse_cython(anchor_pos, next_rack_counts_c_arr_s1, tiles, board, blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, 'V', all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size)
                         # --- END REVERT ---

        # Handle blank for Strategy 1
        if rack_counts_py[' '] > 0:
            if ' ' in allowed_h or ' ' in allowed_v:
                next_rack_counts_c_arr_blank = rack_counts_c_arr.copy() # numpy copy
                next_rack_counts_c_arr_blank[26] -= 1
                for assigned_letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    if assigned_letter in gaddag_root.children:
                        next_node = gaddag_root.children[assigned_letter]
                        # --- REVERT: Create list copy for call ---
                        initial_tiles = [(r_anchor, c_anchor, assigned_letter, True, True)]
                        # --- END REVERT ---
                        if ' ' in allowed_v:
                            # --- REVERT: Pass list copy ---
                            _gaddag_traverse_cython(anchor_pos, next_rack_counts_c_arr_blank.copy(), tiles, board, blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, 'H', all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size)
                            # --- END REVERT ---
                        if ' ' in allowed_h:
                            # --- REVERT: Pass list copy ---
                            _gaddag_traverse_cython(anchor_pos, next_rack_counts_c_arr_blank.copy(), tiles, board, blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, 'V', all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size)
                            # --- END REVERT ---

        # --- Strategy 2 ---
        if not is_first_play:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r_anchor + dr, c_anchor + dc
                adj_pos = (nr, nc)
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    existing_tile_letter = tiles[nr][nc]
                    if existing_tile_letter and adj_pos not in processed_adjacent_starts:
                         processed_adjacent_starts.add(adj_pos)
                         if existing_tile_letter in gaddag_root.children:
                            next_node = gaddag_root.children[existing_tile_letter]
                            # --- REVERT: Create list copy for call ---
                            initial_tiles = [(nr, nc, existing_tile_letter, False, False)]
                            # --- END REVERT ---
                            start_axis = 'V' if dr != 0 else 'H'
                            # --- REVERT: Pass list copies ---
                            _gaddag_traverse_cython(anchor_pos, initial_rack_counts_c_copy, tiles, board, blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), False, start_axis, all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size)
                            _gaddag_traverse_cython(anchor_pos, initial_rack_counts_c_copy, tiles, board, blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, start_axis, all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size)
                            # --- END REVERT ---

    # Post-processing (unchanged)
    all_found_moves.sort(key=lambda m: m['score'], reverse=True)
    final_unique_moves = []
    seen_final_signatures = set()
    for move in all_found_moves:
        sig_details = move.get('newly_placed')
        if sig_details is None:
             print(f"Warning: 'newly_placed' missing for move {move.get('word','?')}. Using 'positions'.")
             sig_details = move.get('positions', [])
        try:
            sig_tuple_list = []
            valid_sig = True
            for item in sig_details:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    sig_tuple_list.append(tuple(item))
                else:
                    print(f"Warning: Invalid item format in sig_details for move {move.get('word','?')}: {item}")
                    valid_sig = False
                    break
            if not valid_sig:
                continue
            sig_tuple = tuple(sorted(sig_tuple_list)) + (move['score'],)
        except TypeError as e:
             print(f"Warning: Could not create signature for move {move.get('word','?')}. Details: {sig_details}. Error: {e}")
             continue
        if sig_tuple not in seen_final_signatures:
            final_unique_moves.append(move)
            seen_final_signatures.add(sig_tuple)

    return final_unique_moves




# python
# MODIFIED: Accept and display exchange info
def draw_hint_dialog(moves, selected_index, is_simulation_result=False, best_exchange_tiles=None, best_exchange_score=None): # Added exchange args
    """Draw the hint dialog showing top 5 moves or simulation results, plus best exchange option."""
    dialog_width, dialog_height = 400, 280 # Increased height slightly for exchange line
    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2

    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
    pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

    # --- MODIFIED TITLE ---
    title_str = "Simulation Results" if is_simulation_result else ("Top 5 Moves" if moves else "No Moves Available")
    title_text = dialog_font.render(title_str, True, BLACK)

    screen.blit(title_text, (dialog_x + 10, dialog_y + 10))

    hint_rects = []
    y_pos = dialog_y + 40
    line_height = 30
    max_moves_to_show = 5

    for i, move_data in enumerate(moves[:max_moves_to_show]): # Iterate through the list (might be dicts from sim)
        # Extract move dict ---
        if is_simulation_result and isinstance(move_data, dict):
            move = move_data.get('move', {}) # Get the inner move dict
            final_score = move_data.get('final_score', 0.0)
        else:
            move = move_data # Assume it's already the move dict
            final_score = 0.0 # Not used for standard hints


        color = HINT_SELECTED_COLOR if i == selected_index else HINT_NORMAL_COLOR
        rect = pygame.Rect(dialog_x + 10, y_pos, dialog_width - 20, line_height)
        pygame.draw.rect(screen, color, rect)

        word = move.get('word', 'N/A')
        score = move.get('score', 0)
        start_pos = move.get('start', (0,0))
        direction = move.get('direction', 'right')
        leave = move.get('leave', [])
        word_display = move.get('word_with_blanks', word.upper()) # Use formatted word
        coord = get_coord(start_pos, direction)
        leave_str = ''.join(sorted(l if l != ' ' else '?' for l in leave))
        avg_opp_score = move.get('avg_opp_score', 0.0) # Get from augmented dict if sim
        leave_val = evaluate_leave(leave) # Recalculate leave value

        # --- MODIFIED TEXT FORMAT ---
        if is_simulation_result:
            # Format: Raw + Leave - Opponent = Final
            text_str = f"{i+1}. {word_display} ({score}{leave_val:+0.1f}-{avg_opp_score:.1f}={final_score:.1f}) L:{leave_str}"
        else:
            # Original format
            text_str = f"{i+1}. {word_display} ({score} pts) at {coord} ({leave_str})"


        text = ui_font.render(text_str, True, BLACK)

        # Truncate text if too wide
        max_text_width = rect.width - 10
        if text.get_width() > max_text_width:
             avg_char_width = text.get_width() / len(text_str) if len(text_str) > 0 else 10
             if avg_char_width > 0:
                 max_chars = int(max_text_width / avg_char_width) - 3
                 if max_chars < 5: max_chars = 5
                 text_str = text_str[:max_chars] + "..."
                 text = ui_font.render(text_str, True, BLACK)

        screen.blit(text, (dialog_x + 15, y_pos + 5))
        hint_rects.append(rect)
        y_pos += line_height

    # --- ADDED: Draw Exchange Info ---
    if best_exchange_tiles is not None and best_exchange_score is not None:
        # Add a small gap
        y_pos += 5
        exchange_str = "".join(sorted(t if t != ' ' else '?' for t in best_exchange_tiles))
        exchange_text = f"Best Exchange: {exchange_str} (Eval: {best_exchange_score:.1f})"
        exchange_surf = ui_font.render(exchange_text, True, BLACK)
        # Draw with a slightly different background? Or just text.
        exchange_rect = pygame.Rect(dialog_x + 10, y_pos, dialog_width - 20, line_height)
        pygame.draw.rect(screen, GRAY, exchange_rect) # Gray background for distinction
        screen.blit(exchange_surf, (dialog_x + 15, y_pos + 5))
        y_pos += line_height # Increment y_pos after drawing exchange line
    # --- END ADDED ---


    # Buttons (Play button might not make sense for simulation results?)
    # Let's keep it for now, but maybe disable it later if needed.
    button_y = dialog_y + dialog_height - BUTTON_HEIGHT - 10
    total_button_width = 3 * BUTTON_WIDTH + 2 * BUTTON_GAP
    button_start_x = dialog_x + (dialog_width - total_button_width) // 2

    play_button_rect = pygame.Rect(button_start_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    all_words_button_rect = pygame.Rect(button_start_x + BUTTON_WIDTH + BUTTON_GAP, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    ok_button_rect = pygame.Rect(button_start_x + 2 * (BUTTON_WIDTH + BUTTON_GAP), button_y, BUTTON_WIDTH, BUTTON_HEIGHT)

    # Draw buttons
    pygame.draw.rect(screen, BUTTON_COLOR, play_button_rect)
    pygame.draw.rect(screen, BUTTON_COLOR, all_words_button_rect)
    pygame.draw.rect(screen, BUTTON_COLOR, ok_button_rect)

    play_text = button_font.render("Play", True, BLACK)
    all_words_text = button_font.render("All Words", True, BLACK)
    ok_text = button_font.render("OK", True, BLACK)

    screen.blit(play_text, play_text.get_rect(center=play_button_rect.center))
    screen.blit(all_words_text, all_words_text.get_rect(center=all_words_button_rect.center))
    screen.blit(ok_text, ok_text.get_rect(center=ok_button_rect.center))

    return hint_rects, play_button_rect, ok_button_rect, all_words_button_rect



# End of Part 5




# Part 6

def draw_all_words_dialog(moves, selected_index, current_scroll_offset): # Changed 3rd arg name
    """Draw the dialog showing all valid moves with scrolling."""
    dialog_x = (WINDOW_WIDTH - ALL_WORDS_DIALOG_WIDTH) // 2; dialog_y = (WINDOW_HEIGHT - ALL_WORDS_DIALOG_HEIGHT) // 2
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, ALL_WORDS_DIALOG_WIDTH, ALL_WORDS_DIALOG_HEIGHT)); pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, ALL_WORDS_DIALOG_WIDTH, ALL_WORDS_DIALOG_HEIGHT), 2)
    header_height = 40; unique_words_count = len(set(move.get('word', '') for move in moves if move.get('word'))); title_text = dialog_font.render(f"All Valid Moves ({unique_words_count} unique words, {len(moves)} plays)", True, BLACK); screen.blit(title_text, (dialog_x + 10, dialog_y + 10))
    content_area_y = dialog_y + header_height; button_area_height = BUTTON_HEIGHT + 30; content_area_height = ALL_WORDS_DIALOG_HEIGHT - header_height - button_area_height; content_area_rect = pygame.Rect(dialog_x, content_area_y, ALL_WORDS_DIALOG_WIDTH, content_area_height)
    content_height = len(moves) * 30; content_surface_width = max(1, ALL_WORDS_DIALOG_WIDTH - 20); content_surface = pygame.Surface((content_surface_width, content_height)); content_surface.fill(DIALOG_COLOR)
    all_words_rects = []; item_height = 30
    for i, move in enumerate(moves):
        y_pos_on_surface = i * item_height
        # Use current_scroll_offset for visibility check ---
        if y_pos_on_surface >= current_scroll_offset - item_height and y_pos_on_surface < current_scroll_offset + content_area_height:
            color = HINT_SELECTED_COLOR if i == selected_index else HINT_NORMAL_COLOR; rect_on_surface = pygame.Rect(10, y_pos_on_surface, content_surface_width - 20, item_height); pygame.draw.rect(content_surface, color, rect_on_surface)
            word = move.get('word', 'N/A'); score = move.get('score', 0); start_pos = move.get('start', (0,0)); direction = move.get('direction', 'right'); leave = move.get('leave', []); word_display = move.get('word_with_blanks', word.upper())
            coord = get_coord(start_pos, direction); leave_str = ''.join(sorted(l if l != ' ' else '?' for l in leave)); text_str = f"{i+1}. {word_display} ({score} pts) at {coord} ({leave_str})"; text = ui_font.render(text_str, True, BLACK)
            max_text_width = rect_on_surface.width - 10 # Truncate text

            # --- CORRECTED INDENTATION BLOCK START ---
            if text.get_width() > max_text_width:
                avg_char_width = text.get_width() / len(text_str) if len(text_str) > 0 else 10
                if avg_char_width > 0:
                    max_chars = int(max_text_width / avg_char_width) - 3
                    if max_chars < 5: max_chars = 5 # Ensure at least a few chars show
                    text_str = text_str[:max_chars] + "..."
                    text = ui_font.render(text_str, True, BLACK) # Re-render truncated text
                # else: # Handle case where avg_char_width is 0 (e.g., empty text_str) - text won't be truncated anyway
                #    pass
            # --- CORRECTED INDENTATION BLOCK END ---

            content_surface.blit(text, (15, y_pos_on_surface + 5))
            # Use current_scroll_offset for screen position calculation ---
            screen_y = content_area_y + y_pos_on_surface - current_scroll_offset;
            screen_rect = pygame.Rect(dialog_x + 10, screen_y, content_surface_width - 20, item_height)
            visible_top = content_area_y; visible_bottom = content_area_y + content_area_height; clipped_top = max(visible_top, screen_rect.top); clipped_bottom = min(visible_bottom, screen_rect.bottom)
            if clipped_bottom > clipped_top: clipped_rect = pygame.Rect(screen_rect.left, clipped_top, screen_rect.width, clipped_bottom - clipped_top); all_words_rects.append((clipped_rect, i))

    # Use current_scroll_offset for blitting source rect ---
    visible_area_on_surface = pygame.Rect(0, current_scroll_offset, content_surface_width, content_area_height);
    screen.blit(content_surface, (dialog_x + 10, content_area_y), visible_area_on_surface)
    pygame.draw.rect(screen, BLACK, (dialog_x + 10, content_area_y, content_surface_width, content_area_height), 1) # Optional border
    total_button_width = 2 * BUTTON_WIDTH + BUTTON_GAP; buttons_x = dialog_x + (ALL_WORDS_DIALOG_WIDTH - total_button_width) // 2; button_y = dialog_y + ALL_WORDS_DIALOG_HEIGHT - BUTTON_HEIGHT - 20
    play_button_rect = pygame.Rect(buttons_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT); ok_button_rect = pygame.Rect(buttons_x + BUTTON_WIDTH + BUTTON_GAP, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, BUTTON_COLOR, play_button_rect); pygame.draw.rect(screen, BUTTON_COLOR, ok_button_rect)
    play_text = button_font.render("Play", True, BLACK); ok_text = button_font.render("OK", True, BLACK)
    screen.blit(play_text, play_text.get_rect(center=play_button_rect.center)); screen.blit(ok_text, ok_text.get_rect(center=ok_button_rect.center))
    return all_words_rects, play_button_rect, ok_button_rect



def get_insertion_index(x, rack_start_x, rack_len):
    """Determine the insertion index (0 to rack_len) based on mouse x-position."""
    # --- DEBUG PRINT: Function Entry ---
    print(f"--- get_insertion_index(x={x}, rack_start_x={rack_start_x}, rack_len={rack_len}) ---")

    # Boundary before the first tile (index 0)
    # Consider the midpoint of the first tile's visual space
    boundary_before_first = rack_start_x + TILE_WIDTH // 2
    # --- DEBUG PRINT: Index 0 Check ---
    print(f"  Checking index 0 boundary: Is x ({x}) < boundary_before_first ({boundary_before_first})?")
    if x < boundary_before_first:
        print(f"  -> Returning index 0") # DEBUG
        return 0

    # Iterate through the potential insertion points *between* tiles (indices 1 to rack_len)
    # The loop goes from i=0 to rack_len-1 (representing the tile *before* the gap)
    for i in range(rack_len):
        # Calculate the midpoint of the gap *after* tile i
        tile_i_right_edge = rack_start_x + i * (TILE_WIDTH + TILE_GAP) + TILE_WIDTH
        gap_mid_x = tile_i_right_edge + TILE_GAP // 2
        # --- DEBUG PRINT: Gap Check ---
        print(f"  Checking index {i+1} boundary: Is x ({x}) < gap_mid_x ({gap_mid_x}) after tile {i}?")
        if x < gap_mid_x:
            # If x is less than the midpoint of the gap after tile i, insert at index i+1
            print(f"  -> Returning index {i+1}") # DEBUG
            return i + 1

    # If the loop completes, it means x was to the right of the midpoint of the gap after the last tile.
    # Insert at the very end (index rack_len).
    # --- DEBUG PRINT: End of Rack ---
    print(f"  Loop finished. Returning index {rack_len} (end of rack)")
    return rack_len



def count_consecutive_existing(row, col, direction, tiles):
    """Count consecutive existing tiles in a direction (unused currently)."""
    count = 0
    if direction == "right": c = col;
    while c < GRID_SIZE and tiles[row][c]: count += 1; c += 1
    else: r = row;
    while r < GRID_SIZE and tiles[r][col]: count += 1; r += 1
    return count

# New function to handle end-of-game score adjustments
def calculate_final_scores(current_scores, racks, bag):
    """
    Calculates the final scores based on game end conditions and remaining tiles.

    Args:
        current_scores (list): List containing the scores of [Player1, Player2] before adjustments.
        racks (list): List containing the tile lists for [Player1's rack, Player2's rack].
        bag (list): The list representing the tile bag.

    Returns:
        list: A new list containing the final adjusted scores for [Player1, Player2].
    """
    final_scores = list(current_scores) # Start with a copy of current scores
    rack_values = [0, 0]

    # Calculate the value of tiles left on each rack (blanks count as 0)
    for i in range(2):
        if i < len(racks) and racks[i]: # Check if rack exists and is not None
             rack_values[i] = sum(TILE_DISTRIBUTION[tile][1] for tile in racks[i] if tile != ' ')
        # else: rack_values[i] remains 0

    # Determine if a player went out (must have empty rack AND bag must be empty)
    p1_out = (not racks[0]) and (not bag)
    p2_out = (not racks[1]) and (not bag)

    if p1_out:
        print("Final Score Adjust: P1 went out.")
        # Player 1 went out: Add double Player 2's remaining tile value to P1's score. P2 score unchanged.
        adjustment = 2 * rack_values[1]
        final_scores[0] += adjustment
        # final_scores[1] remains unchanged
        print(f"  P2 tiles value: {rack_values[1]}, P1 adjustment: +{adjustment}")
    elif p2_out:
        print("Final Score Adjust: P2 went out.")
        # Player 2 went out: Add double Player 1's remaining tile value to P2's score. P1 score unchanged.
        adjustment = 2 * rack_values[0]
        final_scores[1] += adjustment
        # final_scores[0] remains unchanged
        print(f"  P1 tiles value: {rack_values[0]}, P2 adjustment: +{adjustment}")
    else:
        # Neither player went out: Subtract each player's own remaining tile value from their score.
        print("Final Score Adjust: Neither player went out.")
        final_scores[0] -= rack_values[0]
        final_scores[1] -= rack_values[1]
        print(f"  P1 adjustment: -{rack_values[0]}, P2 adjustment: -{rack_values[1]}")

    print(f"  Scores before: {current_scores}, Scores after: {final_scores}")
    return final_scores






# python
# MODIFIED: Update board_tile_counts
def play_hint_move(move, tiles, racks, blanks, scores, turn, bag, board, board_tile_counts): # Added board_tile_counts
    """
    Plays a move (usually from hint/AI), updates state. Handles practice mode. Sorts human rack.
    Updates board_tile_counts.
    """
    global practice_mode, is_ai # Need is_ai to check if player is human

    player_idx = turn - 1
    if not (0 <= player_idx < len(racks)):
        print(f"Error: Invalid player index {player_idx} in play_hint_move.")
        return turn, [], [], board_tile_counts # Return counter unchanged

    current_rack = racks[player_idx]
    move_positions = move.get('positions', [])
    move_blanks_coords = move.get('blanks', set())
    newly_placed_details = move.get('newly_placed', [])

    # --- Verification ---
    needed_tiles = Counter()
    blanks_needed_count = 0
    for r, c, letter in newly_placed_details:
        if (r, c) in move_blanks_coords: blanks_needed_count += 1
        else: needed_tiles[letter] += 1
    rack_counter = Counter(current_rack)
    if blanks_needed_count > rack_counter.get(' ', 0):
        print(f"Error playing move: Needs {blanks_needed_count} blanks, only {rack_counter.get(' ', 0)} available.")
        return turn, [], [], board_tile_counts # Return counter unchanged
    for letter, count in needed_tiles.items():
        if rack_counter.get(letter, 0) < count:
            print(f"Error playing move: Needs {count} '{letter}', only {rack_counter.get(letter, 0)} available.")
            return turn, [], [], board_tile_counts # Return counter unchanged
    # --- End Verification ---

    # Place tiles & update rack
    rack_after_play = current_rack[:] # Work with a copy
    for r, c, letter in newly_placed_details:
        tiles[r][c] = letter
        board_tile_counts[letter] += 1 # Increment counter for the placed letter
        if (r, c) in move_blanks_coords:
            if ' ' in rack_after_play: rack_after_play.remove(' '); blanks.add((r, c))
            else: print("Error: Tried to play blank but none in rack_after_play!")
        else:
            if letter in rack_after_play: rack_after_play.remove(letter)
            else: print(f"Error: Tried to play '{letter}' but not in rack_after_play!")

    score = move.get('score', 0)
    scores[player_idx] += score # Update score

    # Draw new tiles (Skip in 8-letter practice)
    drawn_tiles = []
    if practice_mode != "eight_letter":
        num_to_draw = len(newly_placed_details)
        drawn_tiles = [bag.pop() for _ in range(num_to_draw) if bag]
        rack_after_play.extend(drawn_tiles)

    # --- ADDED SORTING ---
    # Sort the rack alphabetically if the player is human
    if not is_ai[player_idx]:
        rack_after_play.sort()
    # --- END ADDED SORTING ---

    racks[player_idx] = rack_after_play # Update main rack AFTER potential sort

    next_turn = turn if practice_mode == "eight_letter" else 3 - turn # Determine next turn
    return next_turn, drawn_tiles, newly_placed_details, board_tile_counts # Return updated counter



def get_tile_under_mouse(x, y, rack_start_x, rack_y, rack_len):
    """Determine which tile index (0 to rack_len-1) is under the mouse cursor."""
    # Iterate through each potential tile position on the rack
    for i in range(rack_len):
        # Calculate the x-coordinate for the current tile
        tile_x = rack_start_x + i * (TILE_WIDTH + TILE_GAP)
        # Create the rectangle for the current tile *inside* the loop
        tile_rect = pygame.Rect(tile_x, rack_y, TILE_WIDTH, TILE_HEIGHT)
        # Check if the mouse coordinates collide with this specific tile's rectangle *inside* the loop
        if tile_rect.collidepoint(x, y):
            return i # Return the index of the collided tile immediately
    # If the loop finishes without finding any collision (or if rack_len was 0), return None
    return None






def get_batch_game_dialog():
    """Displays a dialog to get the number of batch games."""
    dialog_width, dialog_height = 300, 150
    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
    input_text = ""
    input_active = True
    error_msg = None

    while True:
        screen.fill(WHITE) # Or redraw the mode selection screen behind it
        pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

        title_surf = dialog_font.render("Batch Games", True, BLACK)
        screen.blit(title_surf, (dialog_x + 10, dialog_y + 10))

        prompt_surf = ui_font.render("Number of games:", True, BLACK)
        screen.blit(prompt_surf, (dialog_x + 10, dialog_y + 50))

        input_rect = pygame.Rect(dialog_x + 180, dialog_y + 45, 100, 30)
        pygame.draw.rect(screen, WHITE, input_rect)
        pygame.draw.rect(screen, BLACK, input_rect, 1 if not input_active else 2)
        input_surf = ui_font.render(input_text, True, BLACK)
        screen.blit(input_surf, (input_rect.x + 5, input_rect.y + 5))

        if input_active and int(time.time() * 2) % 2 == 0: # Blinking cursor
            cursor_x = input_rect.x + 5 + input_surf.get_width()
            cursor_y1 = input_rect.y + 5
            cursor_y2 = input_rect.y + input_rect.height - 5
            pygame.draw.line(screen, BLACK, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 1)

        ok_rect = pygame.Rect(dialog_x + 40, dialog_y + 100, BUTTON_WIDTH, BUTTON_HEIGHT)
        cancel_rect = pygame.Rect(dialog_x + dialog_width - BUTTON_WIDTH - 40, dialog_y + 100, BUTTON_WIDTH, BUTTON_HEIGHT)
        pygame.draw.rect(screen, BUTTON_COLOR, ok_rect)
        pygame.draw.rect(screen, BUTTON_COLOR, cancel_rect)
        ok_text_surf = button_font.render("Run Batch", True, BLACK)
        cancel_text_surf = button_font.render("Cancel", True, BLACK)
        screen.blit(ok_text_surf, ok_text_surf.get_rect(center=ok_rect.center))
        screen.blit(cancel_text_surf, cancel_text_surf.get_rect(center=cancel_rect.center))

        if error_msg:
            error_surf = ui_font.render(error_msg, True, RED)
            screen.blit(error_surf, (dialog_x + 10, dialog_y + 75))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                input_active = input_rect.collidepoint(x, y)
                error_msg = None # Clear error on click
                if ok_rect.collidepoint(x, y):
                    try:
                        num_games = int(input_text)
                        if num_games > 0:
                            return num_games
                        else:
                            error_msg = "Enter a positive number."
                    except ValueError:
                        error_msg = "Invalid number."
                elif cancel_rect.collidepoint(x, y):
                    return None # User cancelled
            if event.type == pygame.KEYDOWN:
                if input_active:
                    error_msg = None # Clear error on key press
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        try:
                            num_games = int(input_text)
                            if num_games > 0:
                                return num_games
                            else:
                                error_msg = "Enter a positive number."
                        except ValueError:
                            error_msg = "Invalid number."
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.unicode.isdigit():
                        input_text += event.unicode
                elif event.key == pygame.K_ESCAPE: # Allow escape to cancel
                     return None

        pygame.display.flip()
        pygame.time.Clock().tick(30)


# --- MODIFIED Batch Game Helper Function ---

# python
# MODIFIED: Initialize board_tile_counts
def reset_game_state(initial_config):
    """
    Resets game variables for a new game in a batch and returns the new state.
    Initializes board_tile_counts.
    """
    print("--- Resetting Game State for New Batch Game ---")
    # Initialize local variables for the new state
    new_board, _, new_tiles = create_board()
    local_blanks = set()
    local_scores = [0, 0]
    local_turn = 1
    local_first_play = True
    local_pass_count = 0
    local_exchange_count = 0
    local_consecutive_zero_point_turns = 0
    local_move_history = []
    local_last_played_highlight_coords = set()
    local_is_solving_endgame = False
    local_board_tile_counts = Counter() # Initialize counter

    # Create and shuffle a new bag
    local_bag = create_standard_bag()
    random.shuffle(local_bag)

    # Deal new racks
    local_racks = [[], []]
    try:
        local_racks[0] = [local_bag.pop() for _ in range(7)]
        local_racks[1] = [local_bag.pop() for _ in range(7)]
    except IndexError:
        print("Error: Not enough tiles in bag for initial deal.")
        return None # Indicate failure by returning None

    # Sort racks based on initial config (human vs AI)
    is_ai_config = initial_config.get('is_ai', [False, False])
    for i, rack in enumerate(local_racks):
        if 0 <= i < len(is_ai_config) and not is_ai_config[i]:
            rack.sort()
        # Optionally sort AI racks too: rack.sort()

    # Return the newly created state variables, including the counter
    return (new_board, new_tiles, local_racks, local_blanks, local_scores,
            local_turn, local_first_play, local_bag, local_move_history,
            local_pass_count, local_exchange_count, local_consecutive_zero_point_turns,
            local_last_played_highlight_coords, local_is_solving_endgame,
            local_board_tile_counts) # Added counter



def format_duration(total_seconds):
    """Formats a duration in seconds into a string (Hh Mm Ss or Mm Ss)."""
    if total_seconds < 0:
        return "0m 0s"

    total_seconds = int(round(total_seconds)) # Round to nearest second

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{minutes}m {seconds}s"




def collect_game_stats(game_num, player_names, final_scores, move_history, gcg_filename):
    """
    Calculates and returns stats for a single completed game by summing stored values,
    including move history, quadrant counts, GCG filename, game duration, and luck factor.
    """
    stats = {'game_number': game_num}
    stats['player1_name'] = player_names[0]
    stats['player2_name'] = player_names[1]
    stats['player1_score'] = final_scores[0]
    stats['player2_score'] = final_scores[1]
    stats['winner'] = 'Draw'
    if final_scores[0] > final_scores[1]:
        stats['winner'] = player_names[0]
    elif final_scores[1] > final_scores[0]:
        stats['winner'] = player_names[1]

    moves_p1, moves_p2 = calculate_moves_per_player(move_history)
    stats['player1_moves'] = moves_p1
    stats['player2_moves'] = moves_p2

    avg_p1 = final_scores[0] / moves_p1 if moves_p1 > 0 else 0.00
    avg_p2 = final_scores[1] / moves_p2 if moves_p2 > 0 else 0.00
    stats['player1_avg_score'] = avg_p1
    stats['player2_avg_score'] = avg_p2

    tiles_p1, tiles_p2 = calculate_tiles_per_turn(move_history)
    stats['player1_avg_tiles'] = tiles_p1
    stats['player2_avg_tiles'] = tiles_p2

    bingos_p1, bingos_p2 = calculate_bingos_per_player(move_history)
    stats['player1_bingos'] = bingos_p1
    stats['player2_bingos'] = bingos_p2

    bingo_avg_p1, bingo_avg_p2 = calculate_bingo_avg_per_player(move_history)
    stats['player1_bingo_avg'] = bingo_avg_p1
    stats['player2_bingo_avg'] = bingo_avg_p2

    blanks_p1, blanks_p2 = calculate_blanks_per_player(move_history)
    stats['player1_blanks'] = blanks_p1
    stats['player2_blanks'] = blanks_p2

    leave_p1, leave_p2 = calculate_avg_leave(move_history)
    stats['player1_avg_leave'] = leave_p1
    stats['player2_avg_leave'] = leave_p2

    stats['quadrant_counts'] = calculate_quadrant_counts(move_history)
    stats['move_history'] = copy.deepcopy(move_history)
    stats['gcg_filename'] = gcg_filename

    # --- Sum stored game duration and luck factor ---
    total_duration_seconds = 0.0
    player1_total_luck = 0.0
    player2_total_luck = 0.0
    for move in move_history:
        total_duration_seconds += move.get('turn_duration', 0.0)
        player = move.get('player')
        luck = move.get('luck_factor', 0.0) # Get stored luck
        if player == 1:
            player1_total_luck += luck
        elif player == 2:
            player2_total_luck += luck

    stats['game_duration_seconds'] = total_duration_seconds
    stats['player1_total_luck'] = player1_total_luck
    stats['player2_total_luck'] = player2_total_luck
    # --- END Summation ---

    return stats




def save_batch_statistics(batch_results, player_names, batch_summary_filename):
    """
    Calculates aggregate stats and saves batch results to a specified summary file,
    including detailed bingo info, quadrant counts, GCG filename, game duration,
    and luck factor for each game. Also includes total batch duration and luck.
    """
    if not batch_results:
        print("No batch results to save.")
        return

    # --- Load Word Lists ---
    seven_letter_words = []
    eight_letter_words = []
    try:
        with open("7-letter-list.txt", "r") as f:
            seven_letter_words = [line.strip().upper() for line in f]
    except FileNotFoundError:
        print("Warning: 7-letter-list.txt not found for batch stats file.")
    try:
        with open("8-letter-list.txt", "r") as f:
            eight_letter_words = [line.strip().upper() for line in f]
    except FileNotFoundError:
        print("Warning: 8-letter-list.txt not found for batch stats file.")
    # --- End Load Word Lists ---

    num_games = len(batch_results)
    p1_wins = sum(1 for game in batch_results if game['winner'] == player_names[0])
    p2_wins = sum(1 for game in batch_results if game['winner'] == player_names[1])
    draws = num_games - p1_wins - p2_wins

    p1_total_score = sum(g['player1_score'] for g in batch_results)
    p2_total_score = sum(g['player2_score'] for g in batch_results)
    p1_avg_game_score = p1_total_score / num_games if num_games > 0 else 0.0
    p2_avg_game_score = p2_total_score / num_games if num_games > 0 else 0.0

    # Calculate other aggregate averages (same as before)
    agg_stats = {
        'p1_avg_score_turn': sum(g['player1_avg_score'] * g['player1_moves'] for g in batch_results if g['player1_moves'] > 0) / sum(g['player1_moves'] for g in batch_results if g['player1_moves'] > 0) if sum(g['player1_moves'] for g in batch_results if g['player1_moves'] > 0) > 0 else 0,
        'p2_avg_score_turn': sum(g['player2_avg_score'] * g['player2_moves'] for g in batch_results if g['player2_moves'] > 0) / sum(g['player2_moves'] for g in batch_results if g['player2_moves'] > 0) if sum(g['player2_moves'] for g in batch_results if g['player2_moves'] > 0) > 0 else 0,
        'p1_avg_tiles': sum(g['player1_avg_tiles'] * g['player1_moves'] for g in batch_results if g['player1_moves'] > 0) / sum(g['player1_moves'] for g in batch_results if g['player1_moves'] > 0) if sum(g['player1_moves'] for g in batch_results if g['player1_moves'] > 0) > 0 else 0,
        'p2_avg_tiles': sum(g['player2_avg_tiles'] * g['player2_moves'] for g in batch_results if g['player2_moves'] > 0) / sum(g['player2_moves'] for g in batch_results if g['player2_moves'] > 0) if sum(g['player2_moves'] for g in batch_results if g['player2_moves'] > 0) > 0 else 0,
        'p1_avg_bingo_score': sum(g['player1_bingo_avg'] * g['player1_bingos'] for g in batch_results if g['player1_bingos'] > 0) / sum(g['player1_bingos'] for g in batch_results if g['player1_bingos'] > 0) if sum(g['player1_bingos'] for g in batch_results if g['player1_bingos'] > 0) > 0 else 0,
        'p2_avg_bingo_score': sum(g['player2_bingo_avg'] * g['player2_bingos'] for g in batch_results if g['player2_bingos'] > 0) / sum(g['player2_bingos'] for g in batch_results if g['player2_bingos'] > 0) if sum(g['player2_bingos'] for g in batch_results if g['player2_bingos'] > 0) > 0 else 0,
        'p1_avg_leave': sum(g['player1_avg_leave'] * g['player1_moves'] for g in batch_results if g['player1_moves'] > 0) / sum(g['player1_moves'] for g in batch_results if g['player1_moves'] > 0) if sum(g['player1_moves'] for g in batch_results if g['player1_moves'] > 0) > 0 else 0,
        'p2_avg_leave': sum(g['player2_avg_leave'] * g['player2_moves'] for g in batch_results if g['player2_moves'] > 0) / sum(g['player2_moves'] for g in batch_results if g['player2_moves'] > 0) if sum(g['player2_moves'] for g in batch_results if g['player2_moves'] > 0) > 0 else 0,
        'total_p1_bingos': sum(g['player1_bingos'] for g in batch_results),
        'total_p2_bingos': sum(g['player2_bingos'] for g in batch_results),
        'total_p1_blanks': sum(g['player1_blanks'] for g in batch_results),
        'total_p2_blanks': sum(g['player2_blanks'] for g in batch_results),
        # Aggregate Luck Calculation ---
        'total_p1_luck': sum(g.get('player1_total_luck', 0.0) for g in batch_results),
        'total_p2_luck': sum(g.get('player2_total_luck', 0.0) for g in batch_results),
        
    }

    # --- Average Bingos per Game ---
    p1_avg_bingos_per_game = agg_stats['total_p1_bingos'] / num_games if num_games > 0 else 0.0
    p2_avg_bingos_per_game = agg_stats['total_p2_bingos'] / num_games if num_games > 0 else 0.0

    # Average Luck per Game ---
    p1_avg_luck_per_game = agg_stats['total_p1_luck'] / num_games if num_games > 0 else 0.0
    p2_avg_luck_per_game = agg_stats['total_p2_luck'] / num_games if num_games > 0 else 0.0
    

    # --- Aggregate Average Bingo Index ---
    total_bingo_index_sum = 0
    bingos_with_index_count = 0

    # --- Aggregate Power Tile First Play Scores ---
    power_tiles = {'J', 'Q', 'X', 'Z'}
    power_tile_scores = {'J': 0.0, 'Q': 0.0, 'X': 0.0, 'Z': 0.0}
    power_tile_counts = {'J': 0, 'Q': 0, 'X': 0, 'Z': 0}

    # Calculate Total Batch Duration ---
    total_batch_duration_seconds = sum(game.get('game_duration_seconds', 0.0) for game in batch_results)
    

    # Iterate through games to calculate combined index and power tile scores
    for game in batch_results:
        game_move_history = game.get('move_history', [])
        first_power_used_in_game = set() # Track first use *within this game*

        for move in game_move_history:
            if move.get('move_type') == 'place':
                word = move.get('word', '').upper()
                score = move.get('score', 0)
                word_len = len(word)

                # Calculate Bingo Index (Existing Logic)
                if move.get('is_bingo', False):
                    index = None
                    if word_len == 7 and seven_letter_words:
                        index = get_word_index(word, seven_letter_words)
                    elif word_len == 8 and eight_letter_words:
                        index = get_word_index(word, eight_letter_words)
                    if index is not None:
                        total_bingo_index_sum += index
                        bingos_with_index_count += 1

                # --- NEW: Calculate Power Tile First Play Scores ---
                power_in_word = {char for char in word if char in power_tiles}
                for pt in power_in_word:
                    if pt not in first_power_used_in_game:
                        power_tile_scores[pt] += score
                        power_tile_counts[pt] += 1
                        first_power_used_in_game.add(pt) # Mark as used for this game
                # --- END NEW Power Tile Logic ---

    aggregate_avg_bingo_index = total_bingo_index_sum / bingos_with_index_count if bingos_with_index_count > 0 else 0.0

    # --- Calculate Average Power Tile Scores ---
    avg_power_scores = {}
    for pt in power_tiles:
        avg_power_scores[pt] = power_tile_scores[pt] / power_tile_counts[pt] if power_tile_counts[pt] > 0 else 0.0
    # --- END Power Tile Calculation ---


    # --- Aggregate Quadrant Averages ---
    total_quad_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for game in batch_results:
        quad_counts = game.get('quadrant_counts', {})
        for key in total_quad_counts:
            total_quad_counts[key] += quad_counts.get(key, 0)
    avg_quad_counts = {key: val / num_games if num_games > 0 else 0 for key, val in total_quad_counts.items()}


    # Generate filename (use the passed-in name)
    # save_filename = batch_summary_filename # Use the parameter

    # Write to file
    try:
        with open(batch_summary_filename, "w") as f: # Use the parameter here
            f.write(f"--- Batch Game Results ---\n")
            f.write(f"Total Games: {num_games}\n")
            # Write Total Batch Duration ---
            f.write(f"Total Batch Duration: {format_duration(total_batch_duration_seconds)}\n")
            
            f.write(f"Players: {player_names[0]} vs {player_names[1]}\n")
            f.write("-" * 25 + "\n")
            f.write("Overall Summary:\n")
            f.write(f"  {player_names[0]} Wins: {p1_wins} ({p1_wins/num_games:.1%})\n")
            f.write(f"  {player_names[1]} Wins: {p2_wins} ({p2_wins/num_games:.1%})\n")
            f.write(f"  Draws: {draws} ({draws/num_games:.1%})\n")
            f.write(f"  Avg Game Score {player_names[0]}: {p1_avg_game_score:.2f}\n")
            f.write(f"  Avg Game Score {player_names[1]}: {p2_avg_game_score:.2f}\n")
            f.write("-" * 25 + "\n")
            f.write("Aggregate Statistics (Per Turn / Game Averages):\n")
            f.write(f"                     {player_names[0]:>12} {player_names[1]:>12}\n")
            f.write(f"Avg Score/Turn:    {agg_stats['p1_avg_score_turn']:>12.2f} {agg_stats['p2_avg_score_turn']:>12.2f}\n")
            f.write(f"Avg Tiles/Turn:    {agg_stats['p1_avg_tiles']:>12.2f} {agg_stats['p2_avg_tiles']:>12.2f}\n")
            f.write(f"Total Bingos:      {agg_stats['total_p1_bingos']:>12} {agg_stats['total_p2_bingos']:>12}\n")
            f.write(f"Avg Bingos/Game:   {p1_avg_bingos_per_game:>12.2f} {p2_avg_bingos_per_game:>12.2f}\n")
            f.write(f"Avg Bingo Score:   {agg_stats['p1_avg_bingo_score']:>12.2f} {agg_stats['p2_avg_bingo_score']:>12.2f}\n")
            f.write(f"Total Blanks Used: {agg_stats['total_p1_blanks']:>12} {agg_stats['total_p2_blanks']:>12}\n")
            f.write(f"Avg Leave Value:   {agg_stats['p1_avg_leave']:>12.2f} {agg_stats['p2_avg_leave']:>12.2f}\n")
            # Write Aggregate Luck ---
            f.write(f"Total Luck Factor: {agg_stats['total_p1_luck']:>+12.2f} {agg_stats['total_p2_luck']:>+12.2f}\n")
            f.write(f"Avg Luck / Game:   {p1_avg_luck_per_game:>+12.2f} {p2_avg_luck_per_game:>+12.2f}\n")
            
            f.write("-" * 25 + "\n")
            f.write("Aggregate Bingo Index (7/8 Letter Words):\n")
            f.write(f"  Avg Index (Combined): {aggregate_avg_bingo_index:>6.1f}  (Based on {bingos_with_index_count} bingos)\n")
            f.write("-" * 25 + "\n")
            # Write Power Tile Scores ---
            f.write("Power Tile First Play Scores (Aggregate Avg):\n")
            for pt in sorted(power_tiles): # Sort J, Q, X, Z for consistent order
                 count = power_tile_counts[pt]
                 avg_score = avg_power_scores[pt]
                 f.write(f"  {pt}: {avg_score:>10.2f}  (Based on {count} plays)\n")
            f.write("-" * 25 + "\n")
            # --- END Power Tile Scores ---
            f.write("Aggregate Quadrant Usage (Avg Tiles Per Game):\n")
            f.write(f"  Q2 (Top-Left):  {avg_quad_counts['Q2']:>6.2f}    Q1 (Top-Right):   {avg_quad_counts['Q1']:>6.2f}\n")
            f.write(f"  Q3 (Bot-Left):  {avg_quad_counts['Q3']:>6.2f}    Q4 (Bot-Right):  {avg_quad_counts['Q4']:>6.2f}\n")
            f.write("=" * 40 + "\n")
            f.write("Individual Game Results:\n")
            f.write("=" * 40 + "\n")

            # --- Loop through games and add bingo/quadrant details ---
            for game in batch_results:
                f.write(f"Game {game['game_number']}:\n")
                f.write(f"  Score: {game['player1_name']} {game['player1_score']} - {game['player2_name']} {game['player2_score']}\n")
                f.write(f"  Winner: {game['winner']}\n")
                f.write(f"  Moves: P1={game['player1_moves']}, P2={game['player2_moves']}\n")

                # Write Individual Game Duration ---
                game_duration_str = format_duration(game.get('game_duration_seconds', 0.0))
                f.write(f"  Duration: {game_duration_str}\n")
                

                # Write Individual Game Luck ---
                p1_luck = game.get('player1_total_luck', 0.0)
                p2_luck = game.get('player2_total_luck', 0.0)
                f.write(f"  Luck Factor: P1={p1_luck:+.2f}, P2={p2_luck:+.2f}\n")
                

                # Write Quadrant Counts for this game
                quad_counts = game.get('quadrant_counts', {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0})
                f.write(f"  Quadrants: Q2={quad_counts['Q2']}, Q1={quad_counts['Q1']}, Q3={quad_counts['Q3']}, Q4={quad_counts['Q4']}\n")

                # Write GCG Filename ---
                gcg_file = game.get('gcg_filename', 'N/A')
                f.write(f"  Saved GCG: {gcg_file}\n")
                

                # Write Bingo Details for this game
                game_move_history = game.get('move_history', [])
                game_p1_bingos = []
                game_p2_bingos = []

                for move in game_move_history:
                    if move.get('is_bingo', False):
                        player = move['player']
                        word = move.get('word', 'N/A').upper()
                        score = move.get('score', 0)
                        word_len = len(word)
                        index = None
                        if word_len == 7 and seven_letter_words:
                            index = get_word_index(word, seven_letter_words)
                        elif word_len == 8 and eight_letter_words:
                            index = get_word_index(word, eight_letter_words)

                        prob_text = ""
                        if (word_len == 7 or word_len == 8) and index is not None:
                            prob_text = f" Prob: {index}"
                        elif word_len > 8:
                            prob_text = ""
                        else: # Index is None
                            prob_text = " (N/L)"

                        bingo_line = f"    {word} ({score} pts){prob_text}"
                        if player == 1:
                            game_p1_bingos.append(bingo_line)
                        elif player == 2:
                            game_p2_bingos.append(bingo_line)

                if game_p1_bingos:
                    f.write(f"  {game['player1_name']} Bingos:\n")
                    for line in game_p1_bingos:
                        f.write(line + "\n")
                if game_p2_bingos:
                    f.write(f"  {game['player2_name']} Bingos:\n")
                    for line in game_p2_bingos:
                        f.write(line + "\n")

                f.write("-" * 20 + "\n")

        print(f"Batch statistics saved to {batch_summary_filename}")
        show_message_dialog(f"Batch complete.\nStats saved to {batch_summary_filename}", "Batch Finished")
    except IOError as e:
        print(f"Error saving batch statistics to {batch_summary_filename}: {e}")
        show_message_dialog(f"Error saving batch stats: {e}", "Save Error")
    except NameError as e:
         print(f"Error during batch save - function missing? {e}")
         show_message_dialog(f"Error saving batch stats (missing function?): {e}", "Save Error")
    except Exception as e: # Catch broader exceptions during file writing
        print(f"An unexpected error occurred saving batch statistics: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        show_message_dialog(f"Unexpected error saving batch stats: {e}", "Save Error")









# --- AI Logic ---

# --- AI Strategy Helper Functions ---

# --- Static Leave Values & Constants (Assumed to be defined elsewhere in the full code) ---
#GOOD_LEAVE_BONUS = {' ': 25.6, 'S': 8.0, 'Z': 5.1, 'X': 3.3, 'E': 0.3, 'A': 0.6, 'R': 1.1, 'N': 0.2,'H': 1.1, 'C': 0.9, 'M': 0.6, 'D': 0.5}
#BAD_LEAVE_PENALTY = {'T': -0.1, 'I': -2.1, 'L': -0.2, 'Q': -6.8, 'J': -1.5, 'V': -5.5, 'W': -3.8, 'K': -0.5, 'F': -2.2, 'Y': -0.6, 'U': -5.1, 'B': -2.0, 'G': -2.9, 'P': -0.5, 'O': -2.5}
#DUPLICATE_PENALTY = -4.2 # Applied for each duplicate beyond the first
#VOWELS = "AEIOU"
#BALANCE_PENALTY_FACTOR = -4.0 # Penalty per vowel count away from ideal range
# --- END Constants ---




# Assume LEAVE_LOOKUP_TABLE is loaded globally earlier in the script
# LEAVE_LOOKUP_TABLE = {} # Example initialization






def evaluate_single_move(move_dict, leave_evaluation_func):
    """
    Combines the immediate score of a move with the evaluated score of its leave.

    Args:
        move_dict (dict): The dictionary representing the move (must contain 'score' and 'leave').
        leave_evaluation_func (function): The function to use for evaluating the leave (e.g., evaluate_leave).

    Returns:
        float: The combined evaluation score for the move.
               Using float allows for potential future weighting.
    """
    immediate_score = move_dict.get('score', 0)
    leave = move_dict.get('leave', [])
    leave_score_adjustment = leave_evaluation_func(leave)

    # Simple combination for now: add leave adjustment to immediate score
    # Future difficulty levels could apply weights here:
    # e.g., weight_score * immediate_score + weight_leave * leave_score_adjustment
    combined_score = float(immediate_score + leave_score_adjustment)

    return combined_score


def analyze_unseen_pool(remaining_tiles_dict):
    """
    Analyzes the quality of the unseen tile pool (bag + opponent rack).
    NOTE: In reality, we only know the bag for sure. This uses the combined pool.

    Args:
        remaining_tiles_dict (dict): Dictionary mapping letters to counts of unseen tiles.

    Returns:
        dict: Analysis containing counts and a quality score.
              {'total': int, 'vowels': int, 'consonants': int, 'blanks': int,
               's_count': int, 'power_tiles': int, 'bad_tiles': int, 'quality_score': float}
    """
    analysis = {
        'total': 0, 'vowels': 0, 'consonants': 0, 'blanks': 0,
        's_count': 0, 'power_tiles': 0, 'bad_tiles': 0, 'quality_score': 0.0
    }
    if not remaining_tiles_dict:
        return analysis

    vowel_letters = "AEIOU"
    power_letters = "SZJQX" # Tiles often opening up bingos or high scores
    bad_letters = "VWUQKI" # Tiles often considered difficult or less valuable

    raw_quality_sum = 0.0

    for tile, count in remaining_tiles_dict.items():
        if count <= 0: continue # Skip tiles with zero or negative count

        analysis['total'] += count

        if tile == ' ':
            analysis['blanks'] += count
        elif tile in vowel_letters:
            analysis['vowels'] += count
        else:
            analysis['consonants'] += count

        if tile == 'S':
            analysis['s_count'] += count
        if tile in power_letters:
            analysis['power_tiles'] += count
        if tile in bad_letters:
            analysis['bad_tiles'] += count

        # Add to raw quality score based on predefined values
        raw_quality_sum += POOL_TILE_VALUES.get(tile, 0) * count

    # Normalize quality score slightly based on pool size (average value per tile)
    if analysis['total'] > 0:
        analysis['quality_score'] = raw_quality_sum / analysis['total']
    else:
        analysis['quality_score'] = 0.0

    # print(f"DEBUG analyze_unseen_pool: Pool: {remaining_tiles_dict}, Analysis: {analysis}") # Optional debug
    return analysis




def estimate_draw_value(num_to_draw, pool_analysis):
    """
    Provides a simple heuristic estimate of the value gained by drawing tiles.

    Args:
        num_to_draw (int): The number of tiles to be drawn.
        pool_analysis (dict): The result from analyze_unseen_pool.

    Returns:
        float: An estimated score adjustment based on pool quality.
    """
    # Simple heuristic: base value is proportional to pool quality score and number drawn
    # Positive quality score means pool is generally good, negative means generally bad
    estimated_value = pool_analysis.get('quality_score', 0.0) * num_to_draw * POOL_QUALITY_FACTOR
    # print(f"DEBUG estimate_draw_value: Draw {num_to_draw}, Quality {pool_analysis.get('quality_score', 0.0):.2f}, Estimated Value: {estimated_value:.1f}") # Optional debug
    return estimated_value


def find_best_exchange_option(rack, remaining_tiles_dict, bag_count):
    """
    Determines the best set of tiles to exchange to maximize the
    evaluated score of the *remaining* tiles, plus an estimated draw value.

    Args:
        rack (list): The AI's current rack.
        remaining_tiles_dict (dict): Dictionary of unseen tiles (bag + opponent).
        bag_count (int): Number of tiles currently in the bag.

    Returns:
        tuple: (list_of_tiles_to_exchange, best_estimated_value)
               Returns ([], -float('inf')) if no valid exchange is possible or beneficial.
    """
    best_overall_exchange_tiles = []
    best_overall_estimated_value = -float('inf') # Keep track of the best value found across all numbers of tiles exchanged

    if not rack:
        return [], -float('inf')

    pool_analysis = analyze_unseen_pool(remaining_tiles_dict)

    # Iterate through exchanging k=1 to min(7, len(rack)) tiles
    for k in range(1, min(len(rack), 7) + 1):
        # Check if there are enough tiles in the bag to perform this exchange
        if bag_count < k:
            continue # Cannot exchange k tiles if fewer than k are in the bag

        best_leave_for_k = []
        best_leave_score_for_k = -float('inf')
        current_exchange_tiles_for_k = []

        # Find the combination of k tiles to *remove* that leaves the best rack
        # This is equivalent to finding the best combination of (len(rack) - k) tiles to *keep*
        num_to_keep = len(rack) - k
        if num_to_keep < 0: continue # Should not happen, but safety check

        if num_to_keep == 0: # Exchanging all tiles
             leave_score = 0 # No leave value if keeping zero tiles
             kept_tiles = []
             exchanged_tiles = rack[:]
        else:
            best_kept_subset = []
            # Iterate through all combinations of tiles to KEEP
            for kept_subset_tuple in itertools.combinations(rack, num_to_keep):
                kept_subset_list = list(kept_subset_tuple)
                current_leave_score = evaluate_leave(kept_subset_list) # Evaluate the potential leave

                if current_leave_score > best_leave_score_for_k:
                    best_leave_score_for_k = current_leave_score
                    best_kept_subset = kept_subset_list

            # Determine which tiles were exchanged to achieve this best leave
            temp_rack_counts = Counter(rack)
            temp_kept_counts = Counter(best_kept_subset)
            temp_rack_counts.subtract(temp_kept_counts)
            exchanged_tiles = list(temp_rack_counts.elements())
            kept_tiles = best_kept_subset
            leave_score = best_leave_score_for_k # The score of the best leave found for keeping num_to_keep tiles

        # Calculate the total estimated value for exchanging these 'k' tiles
        estimated_value_of_draw = estimate_draw_value(k, pool_analysis)
        total_estimated_value = leave_score + estimated_value_of_draw

        # print(f"DEBUG Exchange Option (k={k}): Exchanging {exchanged_tiles}, Keeping {kept_tiles}, Leave Score: {leave_score}, Draw Est: {estimated_value_of_draw:.1f}, Total Est: {total_estimated_value:.1f}") # Optional debug

        # Update the overall best exchange found so far
        if total_estimated_value > best_overall_estimated_value:
            best_overall_estimated_value = total_estimated_value
            best_overall_exchange_tiles = exchanged_tiles

    # print(f"DEBUG Best Exchange Found: Tiles={best_overall_exchange_tiles}, Est Value={best_overall_estimated_value:.1f}") # Optional debug
    return best_overall_exchange_tiles, best_overall_estimated_value



# --- NEW Endgame Solver Functions ---

def get_rack_value(rack):
    """Calculates the sum of tile values in a rack."""
    return sum(TILE_DISTRIBUTION[tile][1] for tile in rack if tile != ' ')

def calculate_endgame_score_diff(player_rack, opponent_rack, current_score_diff):
    """
    Calculates the final score difference from the perspective of the player
    whose turn it *would* be, assuming the game just ended.
    """
    player_val = get_rack_value(player_rack)
    opponent_val = get_rack_value(opponent_rack)

    if not player_rack: # Player went out
        # Player gains opponent's remaining value, opponent loses nothing extra relative to player
        final_diff = current_score_diff + opponent_val
    elif not opponent_rack: # Opponent went out
        # Player loses their own remaining value, opponent gains nothing extra relative to player
        final_diff = current_score_diff - player_val
    else: # Game ended via passes or other stalemate (treat as simultaneous deduction)
        final_diff = current_score_diff - player_val + opponent_val

    return final_diff

def format_move_for_debug(move, rack_before):
    """ Creates a concise string representation of a move for debug output. """
    if move == "PASS":
        return "PASS"
    elif isinstance(move, dict):
        word = move.get('word_with_blanks', move.get('word', '?'))
        score = move.get('score', 0)
        coord = get_coord(move.get('start', (0,0)), move.get('direction', '?'))
        leave = move.get('leave', [])
        leave_str = "".join(sorted(l if l != ' ' else '?' for l in leave))
        # Rack before might be useful context, but keep it concise
        # rack_before_str = "".join(sorted(l if l != ' ' else '?' for l in rack_before))
        # return f"[{word} {coord} ({score:+}) L:{leave_str} R:{rack_before_str}]"
        return f"[{word} {coord} ({score:+}) L:{leave_str}]"
    else:
        return "[INVALID MOVE]"

# --- NEW Helper Function for Drawing Indicator ---
def draw_endgame_solving_indicator():
    """Draws the 'AI Solving Endgame...' text."""
    # Ensure necessary globals like screen, dialog_font, RED, WINDOW_WIDTH are accessible
    # Or pass them as arguments if preferred
    solve_text = "AI Solving Endgame..."
    solve_surf = dialog_font.render(solve_text, True, RED)
    # Position it somewhere noticeable, e.g., top center
    solve_rect = solve_surf.get_rect(centerx=WINDOW_WIDTH // 2, top=10)
    # Optional: Add a semi-transparent background
    bg_rect = solve_rect.inflate(20, 10)
    bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
    bg_surf.fill((200, 200, 200, 180)) # Grayish background
    screen.blit(bg_surf, bg_rect)
    screen.blit(solve_surf, solve_rect)



def draw_specify_rack_dialog(p1_name, p2_name, input_texts, active_input_index, original_racks_display):
    """Draws the dialog for specifying player racks."""
    dialog_width, dialog_height = 450, 250 # Increased width slightly
    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2

    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
    pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

    title_text = dialog_font.render("Specify Racks", True, BLACK)
    screen.blit(title_text, (dialog_x + 10, dialog_y + 10))

    label_x = dialog_x + 10
    input_x = dialog_x + 130 # Adjusted for longer names potentially
    input_width = 180 # Width for 7 tiles + padding
    reset_x = input_x + input_width + 10
    reset_width = 80

    # Player 1 Row
    p1_label = ui_font.render(f"{p1_name}:", True, BLACK)
    screen.blit(p1_label, (label_x, dialog_y + 55))
    p1_input_rect = pygame.Rect(input_x, dialog_y + 50, input_width, 30)
    pygame.draw.rect(screen, WHITE, p1_input_rect)
    pygame.draw.rect(screen, BLACK, p1_input_rect, 1 if active_input_index != 0 else 2)
    p1_text_surf = ui_font.render(input_texts[0].upper(), True, BLACK) # Display uppercase
    screen.blit(p1_text_surf, (p1_input_rect.x + 5, p1_input_rect.y + 5))
    if active_input_index == 0 and int(time.time() * 2) % 2 == 0: # Blinking cursor
        cursor_x_pos = p1_input_rect.x + 5 + p1_text_surf.get_width()
        pygame.draw.line(screen, BLACK, (cursor_x_pos, p1_input_rect.y + 5), (cursor_x_pos, p1_input_rect.bottom - 5), 1)
    p1_reset_rect = pygame.Rect(reset_x, dialog_y + 50, reset_width, 30)
    pygame.draw.rect(screen, BUTTON_COLOR, p1_reset_rect)
    p1_reset_text = button_font.render("Reset", True, BLACK)
    screen.blit(p1_reset_text, p1_reset_text.get_rect(center=p1_reset_rect.center))

    # Player 2 Row
    p2_label = ui_font.render(f"{p2_name}:", True, BLACK)
    screen.blit(p2_label, (label_x, dialog_y + 105))
    p2_input_rect = pygame.Rect(input_x, dialog_y + 100, input_width, 30)
    pygame.draw.rect(screen, WHITE, p2_input_rect)
    pygame.draw.rect(screen, BLACK, p2_input_rect, 1 if active_input_index != 1 else 2)
    p2_text_surf = ui_font.render(input_texts[1].upper(), True, BLACK) # Display uppercase
    screen.blit(p2_text_surf, (p2_input_rect.x + 5, p2_input_rect.y + 5))
    if active_input_index == 1 and int(time.time() * 2) % 2 == 0: # Blinking cursor
        cursor_x_pos = p2_input_rect.x + 5 + p2_text_surf.get_width()
        pygame.draw.line(screen, BLACK, (cursor_x_pos, p2_input_rect.y + 5), (cursor_x_pos, p2_input_rect.bottom - 5), 1)
    p2_reset_rect = pygame.Rect(reset_x, dialog_y + 100, reset_width, 30)
    pygame.draw.rect(screen, BUTTON_COLOR, p2_reset_rect)
    p2_reset_text = button_font.render("Reset", True, BLACK)
    screen.blit(p2_reset_text, p2_reset_text.get_rect(center=p2_reset_rect.center))

    # Bottom Buttons
    button_y = dialog_y + dialog_height - BUTTON_HEIGHT - 20
    confirm_rect = pygame.Rect(dialog_x + dialog_width // 2 - BUTTON_WIDTH - BUTTON_GAP // 2, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    cancel_rect = pygame.Rect(dialog_x + dialog_width // 2 + BUTTON_GAP // 2, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, BUTTON_COLOR, confirm_rect)
    pygame.draw.rect(screen, BUTTON_COLOR, cancel_rect)
    confirm_text = button_font.render("Confirm", True, BLACK)
    cancel_text = button_font.render("Cancel", True, BLACK)
    screen.blit(confirm_text, confirm_text.get_rect(center=confirm_rect.center))
    screen.blit(cancel_text, cancel_text.get_rect(center=cancel_rect.center))

    return p1_input_rect, p2_input_rect, p1_reset_rect, p2_reset_rect, confirm_rect, cancel_rect



def draw_override_confirmation_dialog():
    """Draws the dialog asking the user to override bag constraints."""
    dialog_width, dialog_height = 400, 150
    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2

    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
    pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

    message_line1 = "Specified tiles not available in bag."
    message_line2 = "Override bag constraints?"
    msg1_surf = ui_font.render(message_line1, True, BLACK)
    msg2_surf = ui_font.render(message_line2, True, BLACK)
    screen.blit(msg1_surf, (dialog_x + (dialog_width - msg1_surf.get_width()) // 2, dialog_y + 20))
    screen.blit(msg2_surf, (dialog_x + (dialog_width - msg2_surf.get_width()) // 2, dialog_y + 50))

    button_y = dialog_y + dialog_height - BUTTON_HEIGHT - 20
    go_back_rect = pygame.Rect(dialog_x + dialog_width // 2 - BUTTON_WIDTH - BUTTON_GAP // 2, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    override_rect = pygame.Rect(dialog_x + dialog_width // 2 + BUTTON_GAP // 2, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, BUTTON_COLOR, go_back_rect)
    pygame.draw.rect(screen, BUTTON_COLOR, override_rect)
    go_back_text = button_font.render("Go Back", True, BLACK)
    override_text = button_font.render("Override", True, BLACK)
    screen.blit(go_back_text, go_back_text.get_rect(center=go_back_rect.center))
    screen.blit(override_text, override_text.get_rect(center=override_rect.center))

    return go_back_rect, override_rect




# --- NEW Heuristic Evaluation Function ---
def evaluate_endgame_heuristic(rack_player, rack_opponent, current_score_diff):
    """
    Estimates the final score difference at a search depth limit.
    A simple heuristic: assumes the game ends now and calculates score diff.
    More complex heuristics could consider tile values left, etc.
    """
    # For now, use the same logic as the terminal calculation.
    # This assumes the player whose turn it is *might* go out or deductions happen.
    # It's not perfect but provides a baseline evaluation.
    return calculate_endgame_score_diff(rack_player, rack_opponent, current_score_diff)



# --- MODIFIED Endgame Solver Function (with Depth Limit & Heuristic) ---
def negamax_endgame(rack_player, rack_opponent, tiles, blanks, board,
                    current_score_diff, alpha, beta, depth, pass_count,
                    max_depth, search_depth_limit): # Added search_depth_limit
    """
    Negamax solver for the endgame (empty bag) with depth limit.
    Returns (best_score_diff_for_player, best_move_sequence).
    Score difference is from the perspective of the player whose turn it is.
    """
    # --- Debug Print: Function Entry ---
    if depth < 4: # Limit debug print depth
        print(f"{'  ' * depth}DEBUG Depth {depth}: Enter Negamax. PlayerRack={''.join(sorted(rack_player))}, OppRack={''.join(sorted(rack_opponent))}, ScoreDiff={current_score_diff:.1f}, Alpha={alpha:.1f}, Beta={beta:.1f}, PassCount={pass_count}")

    # --- Base Cases: Game Over OR Depth Limit Reached ---
    # MODIFICATION: Check depth limit first
    if depth >= search_depth_limit:
        heuristic_score = evaluate_endgame_heuristic(rack_player, rack_opponent, current_score_diff)
        if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: Depth Limit Reached. HeuristicScore={heuristic_score:.1f}")
        return heuristic_score, [] # Return heuristic score and empty sequence

    if not rack_player or not rack_opponent or pass_count >= 6:
        final_diff = calculate_endgame_score_diff(rack_player, rack_opponent, current_score_diff)
        if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: Terminal Node. FinalDiff={final_diff:.1f}")
        return final_diff, []

    # --- Generate Moves ---
    possible_moves = generate_all_moves_gaddag(rack_player, tiles, board, blanks, GADDAG_STRUCTURE.root)
    if possible_moves is None: possible_moves = []

    # --- Include Pass as an option ---
    can_pass = True

    # --- Evaluate Moves ---
    best_value = -float('inf')
    best_sequence = None

    # Heuristic move ordering
    playout_moves = [m for m in possible_moves if len(m.get('leave', rack_player)) == 0]
    other_moves = [m for m in possible_moves if len(m.get('leave', rack_player)) > 0]
    other_moves.sort(key=lambda m: m.get('score', 0), reverse=True)

    ordered_moves = playout_moves + other_moves
    if can_pass:
        ordered_moves.append("PASS")

    if not ordered_moves: # No plays and cannot pass
        final_diff = calculate_endgame_score_diff(rack_player, rack_opponent, current_score_diff)
        if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: No moves possible. Terminal. FinalDiff={final_diff:.1f}")
        return final_diff, []

    # --- Iterate Through Moves ---
    for move_index, move in enumerate(ordered_moves):
        # Create copies for simulation
        sim_tiles = copy.deepcopy(tiles)
        sim_blanks = blanks.copy()
        sim_rack_player = rack_player[:]
        sim_rack_opponent = rack_opponent[:]
        sim_score_diff = current_score_diff
        sim_pass_count = pass_count
        current_move_details = move

        # --- Debug Print: Trying a Move ---
        if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: Trying move {move_index+1}/{len(ordered_moves)}: {format_move_for_debug(move, rack_player)}")

        if move == "PASS":
            sim_pass_count += 1
            # Recursive call for opponent's turn, passing depth limit
            value, subsequent_sequence = negamax_endgame(
                sim_rack_opponent, sim_rack_player, sim_tiles, sim_blanks, board,
                -sim_score_diff, -beta, -alpha, depth + 1, sim_pass_count, max_depth, search_depth_limit
            )
            value = -value # Flip score back
        else: # It's a play move (dictionary)
            sim_pass_count = 0
            newly_placed_details = move.get('newly_placed', [])
            move_blanks = move.get('blanks', set())
            move_score = move.get('score', 0)

            # Simulate placing tiles and updating rack
            temp_rack = sim_rack_player[:]
            valid_placement = True
            for r, c, letter in newly_placed_details:
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    sim_tiles[r][c] = letter
                    if (r, c) in move_blanks:
                        sim_blanks.add((r, c))
                        if ' ' in temp_rack: temp_rack.remove(' ')
                        else: valid_placement = False; break
                    else:
                        if letter in temp_rack: temp_rack.remove(letter)
                        else: valid_placement = False; break
                else: valid_placement = False; break

            if not valid_placement:
                 if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: Skipping move due to simulation error.")
                 continue # Skip if simulation failed

            sim_rack_player = temp_rack
            sim_score_diff += move_score

            # Recursive call for opponent, passing depth limit
            value, subsequent_sequence = negamax_endgame(
                sim_rack_opponent, sim_rack_player, sim_tiles, sim_blanks, board,
                -sim_score_diff, -beta, -alpha, depth + 1, sim_pass_count, max_depth, search_depth_limit
            )
            value = -value # Flip score back

        # --- Debug Print: Value Returned from Recursion ---
        if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: Move {format_move_for_debug(move, rack_player)} resulted in value {value:.1f}")

        # --- Update Best Sequence ---
        if value > best_value:
            if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: New best value! {value:.1f} > {best_value:.1f}")
            best_value = value
            best_sequence = [current_move_details] + subsequent_sequence

        # --- Alpha-Beta Pruning ---
        alpha = max(alpha, value)
        if alpha >= beta:
            if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: Pruning (alpha={alpha:.1f} >= beta={beta:.1f})")
            break # Prune

    # If no moves were possible at all
    if best_sequence is None:
         final_diff = calculate_endgame_score_diff(rack_player, rack_opponent, current_score_diff)
         if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: No valid sequence found from this node. Returning terminal diff {final_diff:.1f}")
         return final_diff, []

    # --- Debug Print: Returning from Function ---
    if depth < 4: print(f"{'  ' * depth}DEBUG Depth {depth}: Returning Best Value={best_value:.1f}, Seq Start={format_move_for_debug(best_sequence[0], rack_player) if best_sequence else 'None'}")

    return best_value, best_sequence


# --- MODIFIED Top-Level Solver Function (Passes Depth Limit) ---
def solve_endgame(rack_player, rack_opponent, tiles, blanks, board, current_score_diff):
    """
    Top-level function to initiate the endgame solver with a depth limit.
    Calls negamax once and prints the single best sequence found.
    Returns the best first move to make.
    """
    global is_solving_endgame, endgame_start_time
    print("--- Starting Endgame Solver ---")
    is_solving_endgame = True
    endgame_start_time = time.time()

    # Make deep copies to avoid modifying original game state during search
    tiles_copy = copy.deepcopy(tiles)
    blanks_copy = blanks.copy()
    rack_player_copy = rack_player[:]
    rack_opponent_copy = rack_opponent[:]

    # Define and pass search depth limit ---
    # Adjust this value based on performance vs. strength trade-off
    # 6 or 8 is often a reasonable starting point for endgames
    search_depth_limit = 6
    max_possible_depth = len(rack_player_copy) + len(rack_opponent_copy)
    actual_search_depth = min(search_depth_limit, max_possible_depth) # Don't search deeper than possible
    print(f"--- Endgame Search Depth Limit: {actual_search_depth} ---")
    


    # Initial call to negamax, passing the depth limit
    best_score_diff, best_move_sequence = negamax_endgame(
        rack_player_copy, rack_opponent_copy, tiles_copy, blanks_copy, board,
        current_score_diff, -float('inf'), float('inf'), 0, 0,
        max_possible_depth, actual_search_depth # Pass both max possible and actual limit
    )


    solve_duration = time.time() - endgame_start_time
    print(f"--- Endgame Solver Finished ({solve_duration:.2f}s) ---")
    is_solving_endgame = False # Reset flag

    # Print the single best sequence found
    if best_move_sequence:
        # Use the score returned by the limited search (might be heuristic)
        print(f"Best Endgame Sequence Found (Score: {best_score_diff:.0f} at depth {actual_search_depth}):")
        current_player_idx = 0 # 0 for initial player, 1 for opponent
        turn_num = 1
        temp_rack_p = rack_player[:] # Use copies for printing debug sequence
        temp_rack_o = rack_opponent[:]
        # Only print the sequence up to the search depth limit if desired,
        # or the full sequence returned (which might end with heuristic moves)
        # For simplicity, print the whole sequence returned.
        for i, move in enumerate(best_move_sequence):
            player_indicator = "P1►" if current_player_idx == 0 else "P2►"
            rack_to_use = temp_rack_p if current_player_idx == 0 else temp_rack_o
            print(f"  {turn_num}. {player_indicator} {format_move_for_debug(move, rack_to_use)}")

            # Simulate rack change for next debug line (approximate)
            if move != "PASS":
                 if isinstance(move, dict):
                     # Estimate leave based on newly_placed count
                     newly_placed_count = len(move.get('newly_placed', []))
                     if current_player_idx == 0:
                         temp_rack_p = temp_rack_p[newly_placed_count:]
                     else:
                         temp_rack_o = temp_rack_o[newly_placed_count:]
                 else: # Pass move doesn't change rack
                     pass

            current_player_idx = 1 - current_player_idx # Switch player
            if current_player_idx == 0: turn_num += 1 # Increment turn number after P2 moves

        return best_move_sequence[0] # Return only the first move to make
    else:
        print("Endgame Solver: No optimal sequence found (e.g., immediate game end).")
        return "PASS" # Default to passing if no sequence found


def draw_simulation_config_dialog(input_texts, active_input_index):
    """Draws the dialog for configuring simulation parameters."""
    dialog_width, dialog_height = 450, 280 # Slightly taller
    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2

    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
    pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

    title_text = dialog_font.render("Configure Simulation", True, BLACK)
    screen.blit(title_text, (dialog_x + 10, dialog_y + 10))

    labels = [
        "Initial AI Move Candidates:",
        "Opponent Rack Sims / AI Move:",
        "Post-Sim Candidates:"
        # "Ply Depth:" # Not currently used by run_ai_simulation
    ]
    input_rects = []
    input_y_start = dialog_y + 50
    input_height = 30
    input_gap = 15
    label_x = dialog_x + 10
    input_x = dialog_x + 300 # Align inputs to the right
    input_width = 100

    for i, label in enumerate(labels):
        y_pos = input_y_start + i * (input_height + input_gap)
        label_surf = ui_font.render(label, True, BLACK)
        screen.blit(label_surf, (label_x, y_pos + 5))

        rect = pygame.Rect(input_x, y_pos, input_width, input_height)
        input_rects.append(rect)
        pygame.draw.rect(screen, WHITE, rect)
        pygame.draw.rect(screen, BLACK, rect, 1 if active_input_index != i else 2)
        text_surf = ui_font.render(input_texts[i], True, BLACK)
        screen.blit(text_surf, (rect.x + 5, rect.y + 5))

        if active_input_index == i and int(time.time() * 2) % 2 == 0: # Blinking cursor
            cursor_x_pos = rect.x + 5 + text_surf.get_width()
            pygame.draw.line(screen, BLACK, (cursor_x_pos, rect.y + 5), (cursor_x_pos, rect.bottom - 5), 1)

    # Bottom Buttons
    button_y = dialog_y + dialog_height - BUTTON_HEIGHT - 20
    simulate_rect = pygame.Rect(dialog_x + dialog_width // 2 - BUTTON_WIDTH - BUTTON_GAP // 2, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    cancel_rect = pygame.Rect(dialog_x + dialog_width // 2 + BUTTON_GAP // 2, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    pygame.draw.rect(screen, BUTTON_COLOR, simulate_rect)
    pygame.draw.rect(screen, BUTTON_COLOR, cancel_rect)
    simulate_text = button_font.render("Simulate", True, BLACK)
    cancel_text = button_font.render("Cancel", True, BLACK)
    screen.blit(simulate_text, simulate_text.get_rect(center=simulate_rect.center))
    screen.blit(cancel_text, cancel_text.get_rect(center=cancel_rect.center))

    return input_rects, simulate_rect, cancel_rect



# python
# MODIFIED: Removed visual feedback updates for speed optimization
def run_ai_simulation(ai_rack, opponent_rack_len, tiles, blanks, board, bag, gaddag_root, is_first_play,
                      board_tile_counts, # Added board_tile_counts
                      # Parameters use global defaults defined earlier
                      num_ai_candidates=DEFAULT_AI_CANDIDATES,
                      num_opponent_sims=DEFAULT_OPPONENT_SIMULATIONS,
                      num_post_sim_candidates=DEFAULT_POST_SIM_CANDIDATES):
    """
    Performs a 2-ply simulation to find the best AI move.
    Simulates top N AI moves, estimates opponent's best response M times for each,
    then evaluates the top K results using the LEAVE_LOOKUP_TABLE (float values).
    Uses board_tile_counts for pool calculation. Includes debug prints.
    Removed visual feedback for performance.

    Args:
        ai_rack (list): The AI's current rack.
        opponent_rack_len (int): The number of tiles on the opponent's rack.
        tiles (list[list[str]]): Current board state.
        blanks (set): Current blanks on board.
        board (list[list[tuple]]): Board layout with multipliers.
        bag (list): Current tile bag.
        gaddag_root (GaddagNode): The root of the GADDAG structure.
        is_first_play (bool): Whether it's the first play of the game.
        board_tile_counts (Counter): Counter of tiles currently on the board. # Added doc
        num_ai_candidates (int): How many top raw-scoring AI moves to consider initially.
        num_opponent_sims (int): How many opponent racks to simulate for each AI move.
        num_post_sim_candidates (int): How many moves to evaluate with leave after simulation.

    Returns:
        list: A list of dictionaries, each containing {'move': move_dict, 'final_score': float},
              sorted by 'final_score' descending. Returns empty list if no moves or simulation fails.
    """
    import time # Ensure time module is available
    print("--- Running AI 2-Ply Simulation ---") # DEBUG START
    print(f"  Params: AI Cands={num_ai_candidates}, Opp Sims={num_opponent_sims}, Post Sims={num_post_sim_candidates}")
    print(f"  AI Rack: {''.join(sorted(ai_rack))}, Opp Rack Len: {opponent_rack_len}, Bag: {len(bag)}")

    # --- Simulation Indicator Setup REMOVED ---
    # sim_font = pygame.font.SysFont("Arial", 30, bold=True)
    # sim_text_surf = sim_font.render("Simulating...", True, RED, PALE_YELLOW)
    # sim_text_rect = sim_text_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
    # last_update_time = time.time()
    # --- End Indicator Setup REMOVED ---

    print("  Generating initial AI moves...") # DEBUG
    all_ai_moves = generate_all_moves_gaddag(ai_rack, tiles, board, blanks, gaddag_root)
    print(f"  Generated {len(all_ai_moves) if all_ai_moves else 0} initial AI moves.") # DEBUG

    if not all_ai_moves:
        print(f"  Simulation: No initial AI moves found.")
        return []


    # 1. Select Top N AI moves based on raw score
    all_ai_moves.sort(key=lambda m: m.get('score', 0), reverse=True)
    # --- USE PARAMETER ---
    top_ai_moves_candidates = all_ai_moves[:num_ai_candidates]
    print(f"  Simulating top {len(top_ai_moves_candidates)} AI moves...")
    # --- END USE PARAMETER ---

    simulation_results = [] # Store tuples: (ai_move, avg_opponent_score)
    # --- USE PARAMETER ---
    num_simulations_per_move = num_opponent_sims
    # --- END USE PARAMETER ---

    # --- Pre-calculate unseen tiles for opponent rack simulation ---
    # --- MODIFICATION: Use board_tile_counts ---
    remaining_dict = get_remaining_tiles(ai_rack, board_tile_counts)
    # --- END MODIFICATION ---
    unseen_tiles_pool = []
    for tile, count in remaining_dict.items():
        unseen_tiles_pool.extend([tile] * count)
    print(f"  Unseen Pool Size: {len(unseen_tiles_pool)}") # DEBUG

    # --- USE PARAMETER ---
    for i, ai_move in enumerate(top_ai_moves_candidates):
        # --- Periodic Update REMOVED ---
        # current_time = time.time()
        # if current_time - last_update_time > 0.5: # Update roughly 2 times/sec
        #     screen.blit(sim_text_surf, sim_text_rect)
        #     pygame.display.flip()
        #     pygame.event.pump() # Process internal events (like QUIT)
        #     last_update_time = current_time
        #     # Check for quit events explicitly
        #     for event in pygame.event.get(pygame.QUIT):
        #          print("--- Simulation interrupted by QUIT event ---")
        #          return [] # Abort simulation
        # --- End Update REMOVED ---

        # Minimal console output during simulation loop
        # --- DEBUG PRINT ---
        move_word = ai_move.get('word_with_blanks', ai_move.get('word', '?'))
        move_score = ai_move.get('score', 0)
        print(f"\n  Simulating AI move {i+1}/{len(top_ai_moves_candidates)}: '{move_word}' ({move_score} pts)")
        # --- END DEBUG PRINT ---

        total_opponent_score_for_this_move = 0
        ai_score = ai_move.get('score', 0)
        newly_placed_count = len(ai_move.get('newly_placed', []))

        # --- Simulate the AI move once to get the state *after* the move ---
        sim_tiles_after_ai = copy.deepcopy(tiles)
        sim_blanks_after_ai = blanks.copy()
        sim_rack_after_ai = ai_rack[:]
        sim_bag_after_ai = bag[:]
        move_blanks = ai_move.get('blanks', set())
        valid_placement = True
        for r, c, letter in ai_move.get('newly_placed', []):
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                sim_tiles_after_ai[r][c] = letter
                if (r, c) in move_blanks:
                    sim_blanks_after_ai.add((r, c))
                    if ' ' in sim_rack_after_ai: sim_rack_after_ai.remove(' ')
                    else: valid_placement = False; break
                else:
                    if letter in sim_rack_after_ai: sim_rack_after_ai.remove(letter)
                    else: valid_placement = False; break
            else: valid_placement = False; break
        if not valid_placement:
            print("    Skipping AI move due to invalid placement simulation.") # DEBUG
            continue
        num_to_draw_ai = newly_placed_count
        drawn_ai = []
        for _ in range(num_to_draw_ai):
            if sim_bag_after_ai: drawn_ai.append(sim_bag_after_ai.pop())
        sim_rack_after_ai.extend(drawn_ai)
        # --- End AI move simulation ---
        print(f"    State after AI move: Bag={len(sim_bag_after_ai)}") # DEBUG

        # --- Loop M times for opponent simulation ---
        print(f"    Starting {num_simulations_per_move} opponent simulations...") # DEBUG
        for sim_run in range(num_simulations_per_move):
            # --- Periodic Update REMOVED ---
            # if sim_run % 20 == 0: # Update less frequently within inner loop too
            #      current_time_inner = time.time()
            #      if current_time_inner - last_update_time > 0.5: # Update ~2 times/sec
            #          screen.blit(sim_text_surf, sim_text_rect)
            #          pygame.display.flip()
            #          pygame.event.pump() # Process internal events
            #          last_update_time = current_time_inner
            #          # Check for quit events explicitly
            #          for event in pygame.event.get(pygame.QUIT):
            #               print("--- Simulation interrupted by QUIT event (inner loop) ---")
            #               return [] # Abort simulation
            # --- End Update REMOVED ---

            opponent_available_pool = unseen_tiles_pool[:]
            temp_drawn_ai_counts = Counter(drawn_ai)
            temp_pool_copy = opponent_available_pool[:]
            # --- DEBUG: Check pool adjustment ---
            # print(f"        Pool before adj: {len(opponent_available_pool)}, Drawn AI: {''.join(sorted(drawn_ai))}")
            for tile in temp_pool_copy:
                 if temp_drawn_ai_counts.get(tile, 0) > 0:
                      temp_drawn_ai_counts[tile] -= 1
                      try:
                          opponent_available_pool.remove(tile)
                      except ValueError:
                          # This might happen if drawn_ai had duplicates not fully represented in the initial pool calculation
                          # (e.g., if board_tile_counts was slightly off) - print a warning
                          print(f"        WARNING: Could not remove tile '{tile}' from opponent pool during adjustment.")
            # print(f"        Pool after adj: {len(opponent_available_pool)}")
            # --- END DEBUG ---

            random.shuffle(opponent_available_pool)
            actual_opponent_rack_len = min(opponent_rack_len, len(opponent_available_pool))
            sim_opponent_rack = opponent_available_pool[:actual_opponent_rack_len]
            # --- DEBUG PRINT ---
            # print(f"        Sim Opp Rack: {''.join(sorted(sim_opponent_rack))}")
            # --- END DEBUG PRINT ---

            # --- DEBUG: Check if generate_all_moves_gaddag hangs here ---
            # print(f"        Generating opponent moves...")
            opponent_moves = generate_all_moves_gaddag(
                sim_opponent_rack, sim_tiles_after_ai, board, sim_blanks_after_ai, gaddag_root
            )
            # print(f"        Generated {len(opponent_moves) if opponent_moves else 0} opponent moves.")
            # --- END DEBUG ---

            best_opponent_score = 0
            if opponent_moves:
                opponent_moves.sort(key=lambda m: m.get('score', 0), reverse=True)
                best_opponent_score = opponent_moves[0].get('score', 0)
            total_opponent_score_for_this_move += best_opponent_score
        # --- End opponent simulation loop ---
        print(f"    Finished opponent simulations for AI move {i+1}.") # DEBUG

        average_opponent_score = total_opponent_score_for_this_move / num_simulations_per_move
        # --- Store avg_opp_score directly with the move for later use ---
        ai_move['avg_opp_score'] = average_opponent_score # Add score to the dict
        simulation_results.append(ai_move) # Store the augmented move dict
        # --- End store avg_opp_score ---


    if not simulation_results:
         print(f"  Simulation: No results generated. Falling back.")
         return top_ai_moves_candidates if top_ai_moves_candidates else []


    # 2. Select Top K based on (AI Score - Avg Opponent Score)
    simulation_results.sort(key=lambda r: r.get('score', 0) - r.get('avg_opp_score', 0.0), reverse=True)
    # --- USE PARAMETER ---
    top_sim_results = simulation_results[:num_post_sim_candidates]
    # --- END USE PARAMETER ---

    # 3. Apply Leave Evaluation (Lookup) to Top K
    print("--- Evaluating Top Simulation Results with Leave Lookup ---")
    final_evaluated_moves = []
    for move_result in top_sim_results:
        ai_move = move_result # Already contains avg_opp_score
        avg_opp_score = ai_move.get('avg_opp_score', 0.0)
        ai_raw_score = ai_move.get('score', 0)
        leave = ai_move.get('leave', [])
        # --- Use the new evaluate_leave (lookup, returns float) ---
        leave_value = evaluate_leave(leave) # Returns float
        # Update console output format ---
        leave_str_eval = "".join(sorted(['?' if tile == ' ' else tile for tile in leave])) # Use '?' for display
        word_eval = ai_move.get('word_with_blanks', ai_move.get('word', '?'))
        print(f"  Evaluating: {word_eval} ({ai_raw_score} pts), Leave: '{leave_str_eval}', Lookup Value: {leave_value:.2f}") # Format as float

        final_eval_score = float(ai_raw_score) + leave_value - float(avg_opp_score) # Ensure float arithmetic
        final_evaluated_moves.append({'move': ai_move, 'final_score': final_eval_score})
    print("-" * 20)

    # 4. Sort the final evaluated moves
    final_evaluated_moves.sort(key=lambda m: m['final_score'], reverse=True)

    # --- Print Top 5 Choices (Debug/Info) ---
    print("--- Simulation Top 5 Choices (AI Score + LeaveLookup - Avg Opponent Score) ---")
    for i, evaluated_move in enumerate(final_evaluated_moves[:5]):
        move = evaluated_move['move']
        final_score = evaluated_move['final_score']
        word = move.get('word_with_blanks', move.get('word', '?'))
        coord = get_coord(move.get('start', (0,0)), move.get('direction', '?'))
        raw_score = move.get('score', 0)
        leave_list = move.get('leave', [])
        leave_str = "".join(sorted(l if l != ' ' else '?' for l in leave_list))
        avg_opp = move.get('avg_opp_score', 0.0) # Get stored score
        # Get leave value again for printing (it was calculated above)
        leave_val = evaluate_leave(leave_list) # Returns float
        # Update print format ---
        print(f"  {i+1}. {word} at {coord} ({raw_score}) L:'{leave_str}' ({leave_val:.2f}) OppAvg:{avg_opp:.1f} -> Final:{final_score:.1f}") # Format leave_val and final_score

    print("-" * 20)
    # --- End Print Top 5 ---

    # sim_duration = time.time() - start_time # Calculate duration # Removed duration calculation/print
    print(f"--- AI Simulation Complete ---") # Removed duration print

    # --- MODIFIED RETURN ---
    return final_evaluated_moves # Return the full sorted list of evaluated moves








# Function to Replace: handle_mouse_down_event (Complete)
# REASON: Correct Suggest button logic AGAIN for 8-Letter Bingo mode to use
#         pre-calculated practice_target_moves instead of regenerating.

def handle_mouse_down_event(event, state, drawn_rects): # Added drawn_rects parameter
        """
        Handles MOUSEBUTTONDOWN events for the main game loop.
        Checks GADDAG status AND structure existence before Suggest/Simulate actions.
        Adds validation for "Only Fives" mode when playing from dialogs.
        Corrects 8-Letter Bingo hint play logic.
        Ensures cursor state variables (current_r, current_c) are always initialized.
        Corrects Suggest button move generation/usage.
        Uses drawn_rects dictionary for UI element collision checks.
        Fixes bug where hint play history was attributed to the wrong player.
        Fixes bug where practice mode plays from dialogs were recorded twice by returning early.
        Adds debug prints for invalid typed play issue.
        Reads global gaddag_loading_status directly for AI actions.
        Passes board_tile_counts to play_hint_move and run_ai_simulation.
        Calls find_best_exchange_option after simulation for human hints.
        Corrects Suggest logic AGAIN for 8-Letter Bingo mode.
        """
        # --- Access global directly for GADDAG status check ---
        global gaddag_loading_status, GADDAG_STRUCTURE # Need GADDAG_STRUCTURE too

        # Unpack necessary state variables from the dictionary
        turn = state['turn']; dropdown_open = state['dropdown_open']; bag_count = state['bag_count']; is_batch_running = state['is_batch_running']; replay_mode = state['replay_mode']; game_over_state = state['game_over_state']; practice_mode = state['practice_mode']; exchanging = state['exchanging']; hinting = state['hinting']; showing_all_words = state['showing_all_words']; specifying_rack = state['specifying_rack']; showing_simulation_config = state['showing_simulation_config']; showing_practice_end_dialog = state['showing_practice_end_dialog']; confirming_override = state['confirming_override']; final_scores = state['final_scores']; player_names = state['player_names']; move_history = state['move_history']; initial_racks = state['initial_racks']; showing_stats = state['showing_stats']; stats_dialog_x = state['stats_dialog_x']; stats_dialog_y = state['stats_dialog_y']; dialog_x = state['dialog_x']; dialog_y = state['dialog_y']; current_replay_turn = state['current_replay_turn']; selected_tiles = state['selected_tiles']; is_ai = state['is_ai']; specify_rack_original_racks = state['specify_rack_original_racks']; specify_rack_inputs = state['specify_rack_inputs']; specify_rack_active_input = state['specify_rack_active_input']; specify_rack_proposed_racks = state['specify_rack_proposed_racks']; racks = state['racks']; bag = state['bag']; blanks = state['blanks']; tiles = state['tiles']; scores = state['scores']; paused_for_power_tile = state['paused_for_power_tile']; paused_for_bingo_practice = state['paused_for_bingo_practice']; practice_best_move = state['practice_best_move']; all_moves = state['all_moves']; current_power_tile = state['current_power_tile']; number_checks = state['number_checks']; board = state['board']; current_turn_pool_quality_score = state['current_turn_pool_quality_score']; first_play = state['first_play']; pass_count = state['pass_count']; exchange_count = state['exchange_count']; consecutive_zero_point_turns = state['consecutive_zero_point_turns']; last_played_highlight_coords = state['last_played_highlight_coords'];
        practice_solved = state['practice_solved']; # RE-ADD unpacking
        practice_end_message = state['practice_end_message']; simulation_config_inputs = state['simulation_config_inputs']; simulation_config_active_input = state['simulation_config_active_input']; hint_moves = state['hint_moves']; selected_hint_index = state['selected_hint_index']; preview_score_enabled = state['preview_score_enabled']; dragged_tile = state['dragged_tile']; drag_pos = state['drag_pos']; drag_offset = state['drag_offset']; typing = state['typing']; word_positions = state['word_positions']; original_tiles = state['original_tiles']; original_rack = state['original_rack']; selected_square = state['selected_square']; last_left_click_time = state['last_left_click_time']; last_left_click_pos = state['last_left_click_pos']; stats_dialog_dragging = state['stats_dialog_dragging']; dragging = state['dragging']; letter_checks = state['letter_checks']
        # gaddag_loading_status = state['gaddag_loading_status'] # REMOVED - Use global directly
        stats_scroll_offset = state['stats_scroll_offset'] # Unpack stats scroll offset
        stats_dialog_drag_offset = state['stats_dialog_drag_offset'] # Unpack stats drag offset
        all_words_scroll_offset = state['all_words_scroll_offset'] # Unpack all words scroll offset
        restart_practice_mode = state['restart_practice_mode'] # Unpack flag
        stats_total_content_height = state.get('stats_total_content_height', 0) # Use .get() for safety
        board_tile_counts = state['board_tile_counts'] # Unpack the counter
        practice_target_moves = state['practice_target_moves'] # <<< Unpack practice_target_moves
        # --- Initialize local cursor variables from state ---
        current_r = state.get('current_r')
        current_c = state.get('current_c')
        typing_direction = state.get('typing_direction')
        typing_start = state.get('typing_start')
        # --- End Initialize local cursor variables ---
        # --- Unpack exchange hint info ---
        best_exchange_for_hint = state.get('best_exchange_for_hint') # Use .get for safety
        best_exchange_score_for_hint = state.get('best_exchange_score_for_hint', -float('inf')) # Use .get for safety

        # --- Get Rects from the drawn_rects dictionary ---
        practice_play_again_rect = drawn_rects.get('practice_play_again_rect')
        practice_main_menu_rect = drawn_rects.get('practice_main_menu_rect')
        practice_quit_rect = drawn_rects.get('practice_quit_rect')
        sim_input_rects = drawn_rects.get('sim_input_rects', [])
        sim_simulate_rect = drawn_rects.get('sim_simulate_rect')
        sim_cancel_rect = drawn_rects.get('sim_cancel_rect')
        go_back_rect_ov = drawn_rects.get('go_back_rect_ov')
        override_rect_ov = drawn_rects.get('override_rect_ov')
        p1_input_rect_sr = drawn_rects.get('p1_input_rect_sr')
        p2_input_rect_sr = drawn_rects.get('p2_input_rect_sr')
        p1_reset_rect_sr = drawn_rects.get('p1_reset_rect_sr')
        p2_reset_rect_sr = drawn_rects.get('p2_reset_rect_sr')
        confirm_rect_sr = drawn_rects.get('confirm_rect_sr')
        cancel_rect_sr = drawn_rects.get('cancel_rect_sr')
        options_rect_base = drawn_rects.get('options_rect_base')
        dropdown_rects_base = drawn_rects.get('dropdown_rects_base', [])
        save_rect = drawn_rects.get('save_rect')
        quit_rect = drawn_rects.get('quit_rect')
        replay_rect = drawn_rects.get('replay_rect')
        play_again_rect = drawn_rects.get('play_again_rect')
        stats_rect = drawn_rects.get('stats_rect')
        stats_ok_button_rect = drawn_rects.get('stats_ok_button_rect')
        replay_start_rect = state['replay_start_rect'] # These are fixed, get from state
        replay_prev_rect = state['replay_prev_rect']
        replay_next_rect = state['replay_next_rect']
        replay_end_rect = state['replay_end_rect']
        suggest_rect_base = drawn_rects.get('suggest_rect_base')
        simulate_button_rect = drawn_rects.get('simulate_button_rect')
        preview_checkbox_rect = drawn_rects.get('preview_checkbox_rect') # Get from drawn_rects
        p1_alpha_rect = drawn_rects.get('p1_alpha_rect')
        p1_rand_rect = drawn_rects.get('p1_rand_rect')
        p2_alpha_rect = drawn_rects.get('p2_alpha_rect')
        p2_rand_rect = drawn_rects.get('p2_rand_rect')
        tile_rects = drawn_rects.get('tile_rects', [])
        exchange_button_rect = drawn_rects.get('exchange_button_rect')
        cancel_button_rect = drawn_rects.get('cancel_button_rect')
        play_button_rect = drawn_rects.get('play_button_rect')
        ok_button_rect = drawn_rects.get('ok_button_rect')
        all_words_button_rect = drawn_rects.get('all_words_button_rect')
        hint_rects = drawn_rects.get('hint_rects', [])
        all_words_rects = drawn_rects.get('all_words_rects', [])
        all_words_play_rect = drawn_rects.get('all_words_play_rect')
        all_words_ok_rect = drawn_rects.get('all_words_ok_rect')
        # --- End Get Rects ---


        updated_state = {}
        running_inner = True; return_to_mode_selection = False; batch_stop_requested = False; human_played = False
        x, y = event.pos
        # --- Initialize variables for hint dialog ---
        # best_exchange_for_hint = None # Already unpacked
        # best_exchange_score_for_hint = -float('inf') # Already unpacked

        # --- Practice End Dialog Handling ---
        if showing_practice_end_dialog:
             if event.button == 1:
                 if practice_play_again_rect and practice_play_again_rect.collidepoint(x,y):
                     restart_practice_mode = True
                     showing_practice_end_dialog = False
                 elif practice_main_menu_rect and practice_main_menu_rect.collidepoint(x,y):
                     running_inner = False; return_to_mode_selection = True; # batch_stop_requested = True # Let main handle this
                 elif practice_quit_rect and practice_quit_rect.collidepoint(x,y):
                     running_inner = False; # batch_stop_requested = True # Let main handle this
             updated_state.update({'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection, 'restart_practice_mode': restart_practice_mode, 'showing_practice_end_dialog': showing_practice_end_dialog})
             return updated_state
        # Simulation Config Dialog Handling ---
        if showing_simulation_config:
            if event.button == 1:
                clicked_input = False
                for i, rect in enumerate(sim_input_rects):
                    if rect.collidepoint(x, y): simulation_config_active_input = i; clicked_input = True; break
                if not clicked_input: simulation_config_active_input = None
                if sim_cancel_rect and sim_cancel_rect.collidepoint(x, y): showing_simulation_config = False; simulation_config_active_input = None
                elif sim_simulate_rect and sim_simulate_rect.collidepoint(x, y):
                    try:
                        num_ai_cand = int(simulation_config_inputs[0]); num_opp_sim = int(simulation_config_inputs[1]); num_post_sim = int(simulation_config_inputs[2])
                        if num_ai_cand <= 0 or num_opp_sim <= 0 or num_post_sim <= 0: raise ValueError("Values must be positive.")
                        print(f"--- Running Human Turn Simulation with Params: AI Cands={num_ai_cand}, Opp Sims={num_opp_sim}, Post Sims={num_post_sim} ---")
                        showing_simulation_config = False; simulation_config_active_input = None
                        # *** Check GLOBAL status AND GADDAG_STRUCTURE directly ***
                        if gaddag_loading_status != 'loaded' or GADDAG_STRUCTURE is None:
                            show_message_dialog("Cannot simulate: AI data (GADDAG) is not loaded or available.", "Error")
                        else:
                            player_idx = turn - 1; opponent_idx = 1 - player_idx; opponent_rack_len = len(racks[opponent_idx]) if opponent_idx < len(racks) else 7
                            # --- ADD DEBUG PRINT ---
                            print("  DEBUG: About to call run_ai_simulation...")
                            # --- MODIFICATION: Pass board_tile_counts ---
                            simulation_results = run_ai_simulation(racks[player_idx], opponent_rack_len, tiles, blanks, board, bag, GADDAG_STRUCTURE.root, first_play, board_tile_counts, num_ai_candidates=num_ai_cand, num_opponent_sims=num_opp_sim, num_post_sim_candidates=num_post_sim)
                            # --- END MODIFICATION ---
                            print("  DEBUG: run_ai_simulation finished.") # DEBUG
                            if simulation_results: top_sim_move = simulation_results[0]['move']; top_sim_score = simulation_results[0]['final_score']; print(f"  Simulate Button Top Sim Result: Play '{top_sim_move.get('word','N/A')}' (Sim Score: {top_sim_score:.1f})")
                            else: print("  Simulate Button: No valid simulation results found.")
                            hint_moves = simulation_results # Store simulation results

                            # --- ADDED: Evaluate exchange option ---
                            # --- ADD DEBUG ---
                            print(f"  DEBUG: *** BEFORE Exchange Check. bag_count = {bag_count} ***") # NEW DEBUG
                            if bag_count >= 1: # Only evaluate if exchange is possible
                                print("  DEBUG: Evaluating exchange option for hint dialog...") # DEBUG 1
                                current_player_rack = racks[player_idx]
                                # Need remaining tiles dict based on current state
                                remaining_dict_for_exchange = get_remaining_tiles(current_player_rack, board_tile_counts)
                                best_exchange_for_hint, best_exchange_score_for_hint = find_best_exchange_option(current_player_rack, remaining_dict_for_exchange, bag_count)
                                # --- DEBUG 2: Print result always ---
                                if best_exchange_for_hint:
                                    print(f"  DEBUG: Best Exchange Option Found: {''.join(sorted(best_exchange_for_hint))} (Eval: {best_exchange_score_for_hint:.1f})")
                                else:
                                    print(f"  DEBUG: No beneficial exchange option found (Result: {best_exchange_score_for_hint}).") # Modified print
                                # --- END DEBUG 2 ---
                            else:
                                print("  DEBUG: Skipping exchange evaluation (bag empty or < 1 tile).") # DEBUG 3
                                best_exchange_for_hint = None
                                best_exchange_score_for_hint = -float('inf')
                            print("  DEBUG: *** AFTER Exchange Check. ***") # NEW DEBUG
                            # --- END ADDED ---

                            hinting = True; selected_hint_index = 0 if hint_moves else None
                            print(f"  DEBUG: Setting hinting = {hinting}") # NEW DEBUG
                    except ValueError as e: show_message_dialog(f"Invalid input: {e}\nPlease enter positive numbers.", "Input Error")
            # --- MODIFICATION: Remove premature return ---
            # updated_state.update({...}) # Update happens later now
            # return updated_state # REMOVED
        # --- Specify Rack Dialog Handling ---
        if specifying_rack:
            if confirming_override:
                if event.button == 1:
                    if go_back_rect_ov and go_back_rect_ov.collidepoint(x, y): confirming_override = False; specify_rack_proposed_racks = [[], []]
                    elif override_rect_ov and override_rect_ov.collidepoint(x, y):
                        print("Overriding bag constraints and setting racks."); racks[0] = specify_rack_proposed_racks[0][:]; racks[1] = specify_rack_proposed_racks[1][:]
                        if not is_ai[0]: racks[0].sort()
                        if not is_ai[1]: racks[1].sort()
                        # --- REMOVED automatic move regeneration ---
                        all_moves = [] # Explicitly clear moves after setting rack
                        specifying_rack = False; confirming_override = False; specify_rack_inputs = ["", ""]; specify_rack_active_input = None; specify_rack_original_racks = [[], []]; specify_rack_proposed_racks = [[], []]; dropdown_open = False
                updated_state.update({'racks': racks, 'all_moves': all_moves, 'specifying_rack': specifying_rack, 'confirming_override': confirming_override, 'specify_rack_inputs': specify_rack_inputs, 'specify_rack_active_input': specify_rack_active_input, 'specify_rack_original_racks': specify_rack_original_racks, 'specify_rack_proposed_racks': specify_rack_proposed_racks, 'dropdown_open': dropdown_open})
                return updated_state
            if event.button == 1:
                if p1_input_rect_sr and p1_input_rect_sr.collidepoint(x, y): specify_rack_active_input = 0
                elif p2_input_rect_sr and p2_input_rect_sr.collidepoint(x, y): specify_rack_active_input = 1
                elif p1_reset_rect_sr and p1_reset_rect_sr.collidepoint(x, y): specify_rack_inputs[0] = "".join(['?' if t == ' ' else t for t in specify_rack_original_racks[0]])
                elif p2_reset_rect_sr and p2_reset_rect_sr.collidepoint(x, y): specify_rack_inputs[1] = "".join(['?' if t == ' ' else t for t in specify_rack_original_racks[1]])
                elif cancel_rect_sr and cancel_rect_sr.collidepoint(x, y): specifying_rack = False; specify_rack_inputs = ["", ""]; specify_rack_active_input = None; specify_rack_original_racks = [[], []]
                elif confirm_rect_sr and confirm_rect_sr.collidepoint(x, y):
                    valid_input = True; proposed_racks_temp = [[], []]; error_message = None
                    for i in range(2):
                        input_str = specify_rack_inputs[i].upper()
                        if not (0 <= len(input_str) <= 7): error_message = f"Player {i+1} rack must have 0 to 7 tiles."; valid_input = False; break # Allow 0 tiles
                        current_proposed_rack = []
                        for char in input_str:
                            if 'A' <= char <= 'Z': current_proposed_rack.append(char)
                            elif char == '?' or char == ' ': current_proposed_rack.append(' ')
                            else: error_message = f"Invalid character '{char}' in Player {i+1} rack."; valid_input = False; break
                        if not valid_input: break
                        proposed_racks_temp[i] = current_proposed_rack
                    if not valid_input:
                        if error_message: show_message_dialog(error_message, "Input Error")
                    else:
                        bag_counts = Counter(bag); needs_override = False; combined_original_counts = Counter(specify_rack_original_racks[0]) + Counter(specify_rack_original_racks[1]); combined_proposed_counts = Counter(proposed_racks_temp[0]) + Counter(proposed_racks_temp[1]); net_change = combined_proposed_counts - combined_original_counts
                        if any(bag_counts[tile] < count for tile, count in net_change.items()): needs_override = True
                        if needs_override: print("Specified tiles require override."); specify_rack_proposed_racks = [r[:] for r in proposed_racks_temp]; confirming_override = True
                        else:
                            print("Specified racks are valid or don't require bag tiles. Setting racks."); racks[0] = proposed_racks_temp[0][:]; racks[1] = proposed_racks_temp[1][:]
                            if not is_ai[0]: racks[0].sort()
                            if not is_ai[1]: racks[1].sort()
                            # --- REMOVED automatic move regeneration ---
                            all_moves = [] # Explicitly clear moves after setting rack
                            specifying_rack = False; specify_rack_inputs = ["", ""]; specify_rack_active_input = None; specify_rack_original_racks = [[], []]; dropdown_open = False
                else: specify_rack_active_input = None
            updated_state.update({'racks': racks, 'all_moves': all_moves, 'specifying_rack': specifying_rack, 'confirming_override': confirming_override, 'specify_rack_inputs': specify_rack_inputs, 'specify_rack_active_input': specify_rack_active_input, 'specify_rack_original_racks': specify_rack_original_racks, 'specify_rack_proposed_racks': specify_rack_proposed_racks, 'dropdown_open': dropdown_open})
            return updated_state
        # --- Options Menu Click ---
        if options_rect_base and options_rect_base.collidepoint(x, y):
            dropdown_open = not dropdown_open; updated_state['dropdown_open'] = dropdown_open; return updated_state
        # --- Dropdown Item Click ---
        elif dropdown_open:
            clicked_dropdown_item = False; # ... (determine current_options_list) ...
            if is_batch_running: current_options_list = ["Stop Batch", "Quit"]
            elif replay_mode or game_over_state: current_options_list = ["Main", "Quit"]
            elif practice_mode == "eight_letter": current_options_list = ["Give Up", "Main", "Quit"]
            else: current_options_list = ["Pass", "Exchange", "Specify Rack", "Main", "Quit"]
            drawn_options = current_options_list
            for i, rect in enumerate(dropdown_rects_base):
                if rect and rect.collidepoint(x, y):
                    if i < len(drawn_options):
                        selected_option = drawn_options[i]; clicked_dropdown_item = True; dropdown_open = False
                        # ... (handle selected_option actions as before) ...
                        if selected_option == "Stop Batch": print("--- Batch Run Aborted by User ---"); running_inner = False; return_to_mode_selection = True # Set flags
                        elif selected_option == "Pass": move_rack = racks[turn-1][:]; consecutive_zero_point_turns += 1; pass_count += 1; exchange_count = 0; print(f"Player {turn} passed"); human_played = True; paused_for_power_tile = False; paused_for_bingo_practice = False; move_history.append({'player': turn, 'move_type': 'pass', 'rack': move_rack, 'score': 0, 'word': '', 'coord': '', 'blanks': set(), 'positions': [], 'drawn': [], 'is_bingo': False, 'word_with_blanks': '', 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': 0.0}); current_replay_turn = len(move_history); turn = 3 - turn; last_played_highlight_coords = set()
                        elif selected_option == "Exchange":
                            if bag_count >= 7: exchanging = True; selected_tiles.clear() # Use bag_count variable
                            else: show_message_dialog("Cannot exchange, less than 7 tiles in bag.", "Exchange Error")
                        elif selected_option == "Specify Rack":
                            is_human_turn = not is_ai[turn-1]; allowed_mode = game_mode in [MODE_HVH, MODE_HVA]
                            if is_human_turn and allowed_mode:
                                print("Specify Rack selected."); specifying_rack = True; specify_rack_original_racks = [racks[0][:], racks[1][:]]; specify_rack_inputs[0] = "".join(['?' if t == ' ' else t for t in racks[0]]); specify_rack_inputs[1] = "".join(['?' if t == ' ' else t for t in racks[1]]); specify_rack_active_input = None; confirming_override = False
                                typing = False; word_positions = []; selected_square = None; current_r = None; current_c = None # Reset cursor state
                                if original_tiles and original_rack:
                                    for r_wp, c_wp, _ in word_positions: tiles[r_wp][c_wp] = original_tiles[r_wp][c_wp]
                                    racks[turn-1] = original_rack[:];
                                    if not is_ai[turn-1]: racks[turn-1].sort()
                                    blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in word_positions if (r_wp, c_wp) in blanks); blanks.difference_update(blanks_to_remove); original_tiles = None; original_rack = None
                            else: show_message_dialog("Specify Rack only available on Human turn in HvH/HvA modes.", "Action Unavailable")
                        elif selected_option == "Give Up":
                            if practice_mode == "eight_letter":
                                practice_end_message = f"Best: {practice_best_move['word_with_blanks']} ({practice_best_move['score']} pts)" if practice_best_move else "No best move found."; practice_solved = True; showing_practice_end_dialog = True
                        elif selected_option == "Main": running_inner = False; return_to_mode_selection = True # Set flags
                        elif selected_option == "Quit":
                            if confirm_quit(): running_inner = False; return_to_mode_selection = False # Set flags
                        break
            if clicked_dropdown_item:
                 updated_state.update({'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection, 'dropdown_open': dropdown_open, 'exchanging': exchanging, 'selected_tiles': selected_tiles, 'specifying_rack': specifying_rack, 'specify_rack_original_racks': specify_rack_original_racks, 'specify_rack_inputs': specify_rack_inputs, 'specify_rack_active_input': specify_rack_active_input, 'confirming_override': confirming_override, 'typing': typing, 'word_positions': word_positions, 'selected_square': selected_square, 'original_tiles': original_tiles, 'original_rack': original_rack, 'blanks': blanks, 'practice_end_message': practice_end_message, 'practice_solved': practice_solved, 'showing_practice_end_dialog': showing_practice_end_dialog, 'consecutive_zero_point_turns': consecutive_zero_point_turns, 'pass_count': pass_count, 'exchange_count': exchange_count, 'human_played': human_played, 'paused_for_power_tile': paused_for_power_tile, 'paused_for_bingo_practice': paused_for_bingo_practice, 'move_history': move_history, 'current_replay_turn': current_replay_turn, 'turn': turn, 'last_played_highlight_coords': last_played_highlight_coords, 'racks': racks, 'bag': bag, 'current_r': current_r, 'current_c': current_c}) # Pack cursor state
                 return updated_state
            else: dropdown_open = False; updated_state['dropdown_open'] = dropdown_open

        # --- Process other MOUSEBUTTONDOWN only if NOT in batch mode ---
        if not is_batch_running:
            current_time = pygame.time.get_ticks()
            if event.button == 1: # Left Click
                # --- Game Over Event Handling ---
                if game_over_state:
                    # ... (game over button handling - unchanged, uses rects from drawn_rects) ...
                    if save_rect and save_rect.collidepoint(x, y):
                        if final_scores and player_names and move_history and initial_racks:
                            gcg_content = save_game_to_gcg(player_names, move_history, initial_racks, final_scores); now = datetime.datetime.now(); date_str = now.strftime("%d%b%y").upper(); time_str = now.strftime("%H%M"); seq_num = 1; max_existing_num = 0
                            try:
                                for filename in os.listdir('.'):
                                    if filename.startswith(f"{date_str}-") and filename.endswith(".gcg") and "-GAME-" in filename:
                                        parts = filename[:-4].split('-');
                                        if len(parts) >= 4 and parts[2] == "GAME":
                                            if parts[-1].isdigit(): num = int(parts[-1]); max_existing_num = max(max_existing_num, num)
                                seq_num = max_existing_num + 1
                            except OSError as e: print(f"Error listing directory for save sequence number: {e}")
                            save_filename = f"{date_str}-{time_str}-GAME-{seq_num}.gcg"
                            try:
                                with open(save_filename, "w") as f: f.write(gcg_content); print(f"Game saved to {save_filename}"); show_message_dialog(f"Game saved to:\n{save_filename}", "Game Saved")
                            except IOError as e: print(f"Error saving game to {save_filename}: {e}"); show_message_dialog(f"Error saving game: {e}", "Save Error")
                        else: print("Error: Missing data required for saving."); show_message_dialog("Could not save game: Missing data.", "Save Error")
                    elif quit_rect and quit_rect.collidepoint(x, y): running_inner = False; return_to_mode_selection = False # Set flags
                    elif replay_rect and replay_rect.collidepoint(x, y):
                        if move_history: print("Entering Replay Mode..."); replay_mode = True; current_replay_turn = 0; game_over_state = False; showing_stats = False; last_played_highlight_coords = set()
                        else: print("Cannot enter replay: No move history found.")
                    elif play_again_rect and play_again_rect.collidepoint(x, y): running_inner = False; return_to_mode_selection = True # Set flags
                    elif stats_rect and stats_rect.collidepoint(x, y): showing_stats = True; stats_dialog_x = (WINDOW_WIDTH - 480) // 2; stats_dialog_y = (WINDOW_HEIGHT - 600) // 2; stats_scroll_offset = 0; stats_dialog_dragging = False
                    elif showing_stats and stats_ok_button_rect and stats_ok_button_rect.collidepoint(x, y): showing_stats = False
                    elif showing_stats:
                        title_bar_height = 40; stats_title_rect = pygame.Rect(stats_dialog_x, stats_dialog_y, 480, title_bar_height)
                        if stats_title_rect.collidepoint(x, y): stats_dialog_dragging = True; stats_dialog_drag_offset = (x - stats_dialog_x, y - stats_dialog_y)
                    elif not showing_stats:
                        dialog_rect = pygame.Rect(dialog_x, dialog_y, DIALOG_WIDTH, DIALOG_HEIGHT)
                        if dialog_rect.collidepoint(x, y): dragging = True; drag_offset = (x - dialog_x, y - dialog_y)
                    updated_state.update({'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection, 'replay_mode': replay_mode, 'current_replay_turn': current_replay_turn, 'game_over_state': game_over_state, 'showing_stats': showing_stats, 'stats_dialog_x': stats_dialog_x, 'stats_dialog_y': stats_dialog_y, 'stats_dialog_dragging': stats_dialog_dragging, 'dragging': dragging, 'drag_offset': drag_offset, 'last_played_highlight_coords': last_played_highlight_coords})
                    return updated_state

                # --- Active Game / Replay Event Handling ---
                if not specifying_rack and not showing_simulation_config:
                    if replay_mode: # Replay buttons
                        if replay_start_rect.collidepoint(x, y): current_replay_turn = 0; last_played_highlight_coords = set()
                        elif replay_prev_rect.collidepoint(x, y) and current_replay_turn > 0: current_replay_turn -= 1; last_played_highlight_coords = set()
                        elif replay_next_rect.collidepoint(x, y) and current_replay_turn < len(move_history): current_replay_turn += 1; last_played_highlight_coords = set()
                        elif replay_end_rect.collidepoint(x, y): current_replay_turn = len(move_history); last_played_highlight_coords = set()
                    elif not (exchanging or hinting or showing_all_words): # Active game clicks
                        is_human_turn_or_paused_practice = not is_ai[turn-1] or paused_for_power_tile or paused_for_bingo_practice
                        # --- Suggest Button Click ---
                        if suggest_rect_base and suggest_rect_base.collidepoint(x, y) and is_human_turn_or_paused_practice:
                            # --- MODIFICATION START: Correct 8-Letter Logic ---
                            if practice_mode == "eight_letter":
                                print(f"DEBUG: Suggest clicked (8-Letter). Using practice_target_moves (length: {len(practice_target_moves)}).")
                                moves_to_hint = practice_target_moves if practice_target_moves else []
                                all_moves = practice_target_moves # Ensure all_moves state is also correct
                            # --- END MODIFICATION ---
                            else: # Standard game or other practice modes
                                # *** Check GLOBAL status AND GADDAG_STRUCTURE directly ***
                                if gaddag_loading_status != 'loaded' or GADDAG_STRUCTURE is None:
                                    show_message_dialog("Cannot suggest moves: AI data is not loaded or available.", "Loading")
                                    moves_to_hint = [] # Ensure it's empty if GADDAG not ready
                                    all_moves = []
                                else:
                                    current_player_rack = racks[turn-1]
                                    print(f"DEBUG: Suggest clicked (Standard/Other Practice). Regenerating moves for Player {turn}, Rack: {''.join(sorted(current_player_rack))}")
                                    tiles_copy = [row[:] for row in tiles]; blanks_copy = blanks.copy(); board_copy = board
                                    # *** Generate into temporary variable ***
                                    all_moves_generated_for_hint = generate_all_moves_gaddag(current_player_rack, tiles_copy, board_copy, blanks_copy, GADDAG_STRUCTURE.root)
                                    if all_moves_generated_for_hint is None: all_moves_generated_for_hint = []; print("DEBUG: generate_all_moves_gaddag returned None for Suggest.")
                                    else: print(f"DEBUG: generate_all_moves_gaddag returned {len(all_moves_generated_for_hint)} moves for Suggest.")

                                    # Filter for other practice modes if needed
                                    if practice_mode == "power_tiles" and paused_for_power_tile and current_power_tile: power_moves_hint = [m for m in all_moves_generated_for_hint if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)]; moves_to_hint = sorted(power_moves_hint, key=lambda m: m['score'], reverse=True)
                                    elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice: bingo_moves_hint = [m for m in all_moves_generated_for_hint if m.get('is_bingo', False)]; moves_to_hint = sorted(bingo_moves_hint, key=lambda m: m['score'], reverse=True)
                                    else: moves_to_hint = all_moves_generated_for_hint # Use generated moves for standard game
                                    # *** Store the generated moves back into the main state's all_moves ***
                                    all_moves = all_moves_generated_for_hint # Update the state variable

                            # --- Common logic for setting hint state ---
                            hint_moves = moves_to_hint[:5]; hinting = True; selected_hint_index = 0 if hint_moves else None
                            updated_state['all_moves'] = all_moves # Ensure it's packed

                        # --- Simulate Button Click ---
                        elif simulate_button_rect and simulate_button_rect.collidepoint(x, y) and is_human_turn_or_paused_practice:
                            # *** Check GLOBAL status AND GADDAG_STRUCTURE directly ***
                            if gaddag_loading_status != 'loaded' or GADDAG_STRUCTURE is None:
                                show_message_dialog("Cannot simulate: AI data is not loaded or available.", "Loading")
                            else:
                                print("Simulate button clicked."); showing_simulation_config = True; simulation_config_inputs = [str(DEFAULT_AI_CANDIDATES), str(DEFAULT_OPPONENT_SIMULATIONS), str(DEFAULT_POST_SIM_CANDIDATES)]; simulation_config_active_input = None
                                typing = False; word_positions = []; selected_square = None; current_r = None; current_c = None # Reset cursor
                                if original_tiles and original_rack:
                                    for r_wp, c_wp, _ in word_positions: tiles[r_wp][c_wp] = original_tiles[r_wp][c_wp]
                                    racks[turn-1] = original_rack[:];
                                    if not is_ai[turn-1]: racks[turn-1].sort()
                                    blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in word_positions if (r_wp, c_wp) in blanks); blanks.difference_update(blanks_to_remove); original_tiles = None; original_rack = None
                        # --- Preview Checkbox Click ---
                        elif preview_checkbox_rect and preview_checkbox_rect.collidepoint(x, y): # Use the rect from drawn_rects
                            preview_score_enabled = not preview_score_enabled
                        # --- Rack Button Clicks ---
                        current_player_idx = turn - 1
                        if 0 <= current_player_idx < len(is_ai) and is_human_turn_or_paused_practice:
                             if turn == 1:
                                  if p1_alpha_rect and p1_alpha_rect.collidepoint(x, y): racks[0].sort()
                                  elif p1_rand_rect and p1_rand_rect.collidepoint(x, y): random.shuffle(racks[0])
                             elif turn == 2 and practice_mode != "eight_letter":
                                  if p2_alpha_rect and p2_alpha_rect.collidepoint(x, y): racks[1].sort()
                                  elif p2_rand_rect and p2_rand_rect.collidepoint(x, y): random.shuffle(racks[1])
                        # --- Drag Start ---
                        rack_y_drag = BOARD_SIZE + 80 if turn == 1 else BOARD_SIZE + 150; rack_width_calc = 7 * (TILE_WIDTH + TILE_GAP) - TILE_GAP; replay_area_end_x = 10 + 4 * (REPLAY_BUTTON_WIDTH + REPLAY_BUTTON_GAP); min_rack_start_x = replay_area_end_x + BUTTON_GAP + 20; rack_start_x_calc = max(min_rack_start_x, (BOARD_SIZE - rack_width_calc) // 2)
                        if 0 <= current_player_idx < len(racks) and 0 <= current_player_idx < len(is_ai):
                            rack_len = len(racks[current_player_idx]); tile_idx = get_tile_under_mouse(x, y, rack_start_x_calc, rack_y_drag, rack_len)
                            if tile_idx is not None and not dragged_tile and is_human_turn_or_paused_practice:
                                dragged_tile = (turn, tile_idx); drag_pos = (x, y);
                                tile_abs_x = rack_start_x_calc + tile_idx * (TILE_WIDTH + TILE_GAP); tile_center_x = tile_abs_x + TILE_WIDTH // 2; tile_center_y = rack_y_drag + TILE_HEIGHT // 2; drag_offset = (x - tile_center_x, y - tile_center_y)
                        # --- Board Click ---
                        if not dragged_tile and is_human_turn_or_paused_practice:
                            col = (x - 40) // SQUARE_SIZE; row = (y - 40) // SQUARE_SIZE
                            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and not tiles[row][col]:
                                is_double_click = (last_left_click_pos == (row, col) and current_time - last_left_click_time < DOUBLE_CLICK_TIME)
                                if is_double_click:
                                     selected_square = None; typing = False; current_r = None; current_c = None # Reset cursor
                                     if word_positions and original_tiles and original_rack:
                                          for r_wp, c_wp, _ in word_positions: tiles[r_wp][c_wp] = original_tiles[r_wp][c_wp]
                                          racks[turn-1] = original_rack[:];
                                          if not is_ai[turn-1]: racks[turn-1].sort()
                                          blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in word_positions if (r_wp, c_wp) in blanks); blanks.difference_update(blanks_to_remove); word_positions = []; original_tiles = None; original_rack = None
                                elif selected_square is None or selected_square[:2] != (row, col): selected_square = (row, col, "right"); typing = False; word_positions = []; current_r = None; current_c = None # Reset cursor
                                elif selected_square[2] == "right": selected_square = (row, col, "down")
                                elif selected_square[2] == "down": selected_square = None; current_r = None; current_c = None # Reset cursor
                                last_left_click_pos = (row, col); last_left_click_time = current_time
                            else: selected_square = None; current_r = None; current_c = None # Reset cursor

                    # --- Handle clicks within dialogs (Exchange, Hint, All Words) ---
                    elif exchanging:
                        clicked_tile = False
                        for i, rect in enumerate(tile_rects):
                            if rect.collidepoint(x, y): selected_tiles.add(i) if i not in selected_tiles else selected_tiles.remove(i); clicked_tile = True; break
                        if not clicked_tile:
                            if exchange_button_rect and exchange_button_rect.collidepoint(x, y):
                                if selected_tiles:
                                    tiles_to_exchange = [racks[turn-1][i] for i in selected_tiles]; print(f"Player {turn} exchanging {len(tiles_to_exchange)} tiles: {''.join(sorted(tiles_to_exchange))}"); move_rack = racks[turn-1][:]
                                    new_rack = [tile for i, tile in enumerate(racks[turn-1]) if i not in selected_tiles]; num_to_draw = len(tiles_to_exchange); drawn_tiles = [bag.pop() for _ in range(num_to_draw) if bag]; new_rack.extend(drawn_tiles); racks[turn-1] = new_rack
                                    if not is_ai[turn-1]: racks[turn-1].sort(); bag.extend(tiles_to_exchange); random.shuffle(bag)
                                    luck_factor = 0.0
                                    if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                    move_history.append({'player': turn, 'move_type': 'exchange', 'rack': move_rack, 'exchanged_tiles': tiles_to_exchange, 'drawn': drawn_tiles, 'score': 0, 'word': '', 'coord': '', 'blanks': set(), 'positions': [], 'is_bingo': False, 'word_with_blanks': '', 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history)
                                    exchanging = False; selected_tiles.clear(); consecutive_zero_point_turns += 1; exchange_count += 1; pass_count = 0; human_played = True; paused_for_power_tile = False; paused_for_bingo_practice = False; turn = 3 - turn; last_played_highlight_coords = set()
                                else: show_message_dialog("No tiles selected for exchange.", "Exchange Error")
                            elif cancel_button_rect and cancel_button_rect.collidepoint(x, y): exchanging = False; selected_tiles.clear()
                    elif hinting:
                        clicked_in_dialog = False
                        if play_button_rect and play_button_rect.collidepoint(x, y) and selected_hint_index is not None and selected_hint_index < len(hint_moves):
                            clicked_in_dialog = True; selected_move = None; is_simulation_result = bool(hint_moves and isinstance(hint_moves[0], dict) and 'final_score' in hint_moves[0])
                            if is_simulation_result: selected_item = hint_moves[selected_hint_index]; selected_move = selected_item.get('move')
                            else: selected_move = hint_moves[selected_hint_index]

                            if selected_move and isinstance(selected_move, dict):
                                player_who_played = turn # Store player number BEFORE turn changes
                                move_rack = racks[player_who_played-1][:]; valid_for_practice = True
                                # Check for "Only Fives" practice
                                if practice_mode == "only_fives":
                                    if not does_move_form_five_letter_word(selected_move, tiles, blanks): show_message_dialog("At least one 5-letter word must be formed.", "Invalid Play"); valid_for_practice = False
                                # --- 8-LETTER BINGO HINT PLAY LOGIC ---
                                elif practice_mode == "eight_letter":
                                    if practice_best_move: # Ensure best move is known
                                        selected_score = selected_move.get('score', -1); max_score_8l = practice_best_move.get('score', 0)
                                        if selected_score >= max_score_8l and max_score_8l > 0: # Correct play
                                            print(f"8-Letter Hint Play CORRECT! Played: '{selected_move.get('word_with_blanks')}' ({selected_score} pts)");
                                            # --- MODIFICATION: Pass board_tile_counts ---
                                            _next_turn, _drawn, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                            human_played = True; hinting = False; practice_solved = True; showing_practice_end_dialog = True; practice_end_message = f"Correct! You found the highest scoring bingo:\n{selected_move.get('word_with_blanks','')} ({selected_score} pts)"; last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', []))
                                        else: print(f"8-Letter Hint Play INCORRECT. Score: {selected_score}, Target: {max_score_8l}"); show_message_dialog(f"Try again. The highest score is {max_score_8l}.", "8-Letter Bingo"); hinting = False # Close hint dialog
                                    else: print("Error: Cannot validate 8-letter hint play, best move unknown."); show_message_dialog("Error: Best move data missing for validation.", "Internal Error"); hinting = False
                                    valid_for_practice = False # Prevent falling through
                                # --- END 8-LETTER BINGO HINT PLAY LOGIC ---
                                # --- Power Tile Practice Hint Play Logic ---
                                elif paused_for_power_tile:
                                    power_moves_filtered = [ m for m in all_moves if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks) ]; max_power_score_filtered = max(m['score'] for m in power_moves_filtered) if power_moves_filtered else 0
                                    if selected_move.get('score', -1) >= max_power_score_filtered:
                                        # --- MODIFICATION: Pass board_tile_counts ---
                                        next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                        human_played = True; hinting = False; paused_for_power_tile = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0; luck_factor = 0.0;
                                        if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                        move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                                        # Pack state and return immediately
                                        updated_state.update({'turn': turn, 'human_played': human_played, 'hinting': hinting, 'paused_for_power_tile': paused_for_power_tile, 'consecutive_zero_point_turns': consecutive_zero_point_turns, 'pass_count': pass_count, 'exchange_count': exchange_count, 'move_history': move_history, 'current_replay_turn': current_replay_turn, 'last_played_highlight_coords': last_played_highlight_coords, 'racks': racks, 'bag': bag, 'tiles': tiles, 'blanks': blanks, 'scores': scores, 'board_tile_counts': board_tile_counts}) # Pack counter
                                        return updated_state # Return early
                                    else: show_message_dialog(f"This is not the highest scoring move with {current_power_tile} matching the selected lengths!", "Incorrect Move"); valid_for_practice = False # Prevent finalization
                                # --- Bingo Bango Bongo Hint Play Logic ---
                                elif paused_for_bingo_practice:
                                    bingo_moves = [m for m in all_moves if m.get('is_bingo', False)]; max_bingo_score = max(m['score'] for m in bingo_moves) if bingo_moves else 0
                                    if selected_move.get('is_bingo', False) and selected_move.get('score', -1) >= max_bingo_score:
                                        # --- MODIFICATION: Pass board_tile_counts ---
                                        next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                        human_played = True; hinting = False; paused_for_bingo_practice = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0; luck_factor = 0.0;
                                        if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                        # *** USE player_who_played FOR HISTORY ***
                                        move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                                        # Pack state and return immediately
                                        updated_state.update({'turn': turn, 'human_played': human_played, 'hinting': hinting, 'paused_for_bingo_practice': paused_for_bingo_practice, 'consecutive_zero_point_turns': consecutive_zero_point_turns, 'pass_count': pass_count, 'exchange_count': exchange_count, 'move_history': move_history, 'current_replay_turn': current_replay_turn, 'last_played_highlight_coords': last_played_highlight_coords, 'racks': racks, 'bag': bag, 'tiles': tiles, 'blanks': blanks, 'scores': scores, 'board_tile_counts': board_tile_counts}) # Pack counter
                                        return updated_state # Return early
                                    else: show_message_dialog(f"This is not the highest scoring bingo! Max score is {max_bingo_score}.", "Incorrect Move"); valid_for_practice = False # Prevent finalization
                                # --- Standard Play or Valid "Only Fives" Finalization ---
                                if valid_for_practice and practice_mode not in ["eight_letter"] and not paused_for_power_tile and not paused_for_bingo_practice:
                                    # --- MODIFICATION: Pass board_tile_counts ---
                                    next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                    human_played = True; hinting = False; paused_for_power_tile = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0
                                    luck_factor = 0.0;
                                    if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                    # *** USE player_who_played FOR HISTORY ***
                                    move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                            elif is_simulation_result and not selected_move: show_message_dialog("Error retrieving move data from selected simulation result.", "Internal Error")
                        elif ok_button_rect and ok_button_rect.collidepoint(x, y): clicked_in_dialog = True; hinting = False
                        elif all_words_button_rect and all_words_button_rect.collidepoint(x, y):
                            clicked_in_dialog = True; hinting = False; showing_all_words = True;
                            # *** Use the updated all_moves from state ***
                            current_all_moves = all_moves # Use the version potentially updated by Suggest
                            if practice_mode == "eight_letter": moves_for_all = practice_target_moves
                            elif practice_mode == "power_tiles" and paused_for_power_tile: moves_for_all = sorted([m for m in current_all_moves if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)], key=lambda m: m['score'], reverse=True)
                            elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice: moves_for_all = sorted([m for m in current_all_moves if m.get('is_bingo', False)], key=lambda m: m['score'], reverse=True)
                            else: moves_for_all = current_all_moves # Use potentially updated all_moves
                            selected_hint_index = 0 if moves_for_all else None; all_words_scroll_offset = 0
                        elif hint_rects:
                            for i, rect in enumerate(hint_rects):
                                if rect.collidepoint(x, y) and i < len(hint_moves): clicked_in_dialog = True; selected_hint_index = i; break
                        dialog_width_hint, dialog_height_hint = 400, 280 # Adjusted height
                        dialog_rect_hint = pygame.Rect((WINDOW_WIDTH - dialog_width_hint) // 2, (WINDOW_HEIGHT - dialog_height_hint) // 2, dialog_width_hint, dialog_height_hint)
                        if dialog_rect_hint.collidepoint(x,y) and not clicked_in_dialog: pass
                    elif showing_all_words:
                        clicked_in_dialog = False;
                        # *** Use the updated all_moves from state ***
                        current_all_moves = all_moves
                        if practice_mode == "eight_letter": moves_for_all = practice_target_moves
                        elif practice_mode == "power_tiles" and paused_for_power_tile: moves_for_all = sorted([m for m in current_all_moves if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)], key=lambda m: m['score'], reverse=True)
                        elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice: moves_for_all = sorted([m for m in current_all_moves if m.get('is_bingo', False)], key=lambda m: m['score'], reverse=True)
                        else: moves_for_all = current_all_moves # Use potentially updated all_moves
                        if all_words_play_rect and all_words_play_rect.collidepoint(x, y) and selected_hint_index is not None and selected_hint_index < len(moves_for_all):
                            clicked_in_dialog = True; selected_move = moves_for_all[selected_hint_index];
                            player_who_played = turn # Store player number BEFORE turn changes
                            move_rack = racks[player_who_played-1][:];
                            valid_for_practice = True
                            # --- ADD "Only Fives" Check ---
                            if practice_mode == "only_fives":
                                if not does_move_form_five_letter_word(selected_move, tiles, blanks): show_message_dialog("At least one 5-letter word must be formed.", "Invalid Play"); valid_for_practice = False
                            # --- ADD 8-Letter Check (Similar to hint dialog) ---
                            elif practice_mode == "eight_letter":
                                 if practice_best_move:
                                     selected_score = selected_move.get('score', -1); max_score_8l = practice_best_move.get('score', 0)
                                     if selected_score >= max_score_8l and max_score_8l > 0: # Correct
                                         print(f"8-Letter All Words Play CORRECT! Played: '{selected_move.get('word_with_blanks')}' ({selected_score} pts)");
                                         # --- MODIFICATION: Pass board_tile_counts ---
                                         _next_turn, _drawn, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                         human_played = True; showing_all_words = False; practice_solved = True; showing_practice_end_dialog = True; practice_end_message = f"Correct! You found the highest scoring bingo:\n{selected_move.get('word_with_blanks','')} ({selected_score} pts)"; last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', []))
                                     else: print(f"8-Letter All Words Play INCORRECT. Score: {selected_score}, Target: {max_score_8l}"); show_message_dialog(f"Try again. The highest score is {max_score_8l}.", "8-Letter Bingo"); showing_all_words = False # Close dialog
                                 else: print("Error: Cannot validate 8-letter all words play, best move unknown."); show_message_dialog("Error: Best move data missing for validation.", "Internal Error"); showing_all_words = False
                                 valid_for_practice = False # Prevent standard finalization
                            # --- END 8-Letter Check ---
                            if valid_for_practice: # Proceed only if valid for current practice mode and not 8-letter
                                if paused_for_bingo_practice:
                                    bingo_moves = [m for m in all_moves if m.get('is_bingo', False)]; max_bingo_score = max(m['score'] for m in bingo_moves) if bingo_moves else 0
                                    if selected_move.get('is_bingo', False) and selected_move.get('score', -1) >= max_bingo_score:
                                        # --- MODIFICATION: Pass board_tile_counts ---
                                        next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                        human_played = True; showing_all_words = False; paused_for_bingo_practice = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0; luck_factor = 0.0;
                                        if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                        # *** USE player_who_played FOR HISTORY ***
                                        move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                                        # Pack state and return immediately
                                        updated_state.update({'turn': turn, 'human_played': human_played, 'showing_all_words': showing_all_words, 'paused_for_bingo_practice': paused_for_bingo_practice, 'consecutive_zero_point_turns': consecutive_zero_point_turns, 'pass_count': pass_count, 'exchange_count': exchange_count, 'move_history': move_history, 'current_replay_turn': current_replay_turn, 'last_played_highlight_coords': last_played_highlight_coords, 'racks': racks, 'bag': bag, 'tiles': tiles, 'blanks': blanks, 'scores': scores, 'board_tile_counts': board_tile_counts}) # Pack counter
                                        return updated_state # Return early
                                    else: show_message_dialog(f"This is not the highest scoring bingo! Max score is {max_bingo_score}.", "Incorrect Move")
                                else: # Standard play or valid "Only Fives" play
                                    # --- MODIFICATION: Pass board_tile_counts ---
                                    next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                    human_played = True; showing_all_words = False; paused_for_power_tile = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0
                                    luck_factor = 0.0;
                                    if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                    # *** USE player_who_played FOR HISTORY ***
                                    move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                        elif all_words_ok_rect and all_words_ok_rect.collidepoint(x, y): clicked_in_dialog = True; showing_all_words = False
                        elif all_words_rects:
                            for rect, idx in all_words_rects:
                                if rect.collidepoint(x, y): clicked_in_dialog = True; selected_hint_index = idx; break
                        dialog_rect_all = pygame.Rect((WINDOW_WIDTH - ALL_WORDS_DIALOG_WIDTH) // 2, (WINDOW_HEIGHT - ALL_WORDS_DIALOG_HEIGHT) // 2, ALL_WORDS_DIALOG_WIDTH, ALL_WORDS_DIALOG_HEIGHT)
                        if dialog_rect_all.collidepoint(x,y) and not clicked_in_dialog: pass

            # --- Right Click Handling ---
            elif event.button == 3:
                selected_square = None; current_r = None; current_c = None # Reset cursor
                if typing:
                    if original_tiles and original_rack:
                        for r_wp, c_wp, _ in word_positions: tiles[r_wp][c_wp] = original_tiles[r_wp][c_wp]
                        racks[turn-1] = original_rack[:];
                        if not is_ai[turn-1]: racks[turn-1].sort()
                        blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in word_positions if (r_wp, c_wp) in blanks); blanks.difference_update(blanks_to_remove)
                    typing = False; typing_start = None; typing_direction = None; word_positions = []; original_tiles = None; original_rack = None;



        # Pack updated state variables into the return dictionary
        updated_state.update({
            'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection,
            'dropdown_open': dropdown_open,
            'hinting': hinting, 'showing_all_words': showing_all_words, 'exchanging': exchanging,
            'typing': typing, 'selected_square': selected_square, 'dragged_tile': dragged_tile,
            'drag_pos': drag_pos, 'drag_offset': drag_offset, 'selected_hint_index': selected_hint_index,
            'game_over_state': game_over_state, 'showing_stats': showing_stats,
            'stats_dialog_dragging': stats_dialog_dragging, 'dragging': dragging,
            'specifying_rack': specifying_rack, 'specify_rack_active_input': specify_rack_active_input,
            'specify_rack_inputs': specify_rack_inputs, 'specify_rack_original_racks': specify_rack_original_racks,
            'specify_rack_proposed_racks': specify_rack_proposed_racks, 'confirming_override': confirming_override,
            'showing_simulation_config': showing_simulation_config, 'simulation_config_active_input': simulation_config_active_input,
            'turn': turn, 'pass_count': pass_count, 'exchange_count': exchange_count,
            'consecutive_zero_point_turns': consecutive_zero_point_turns, 'human_played': human_played,
            'paused_for_power_tile': paused_for_power_tile, 'paused_for_bingo_practice': paused_for_bingo_practice,
            'practice_solved': practice_solved, # RE-ADD packing
            'showing_practice_end_dialog': showing_practice_end_dialog,
            'replay_mode': replay_mode, 'current_replay_turn': current_replay_turn,
            'last_played_highlight_coords': last_played_highlight_coords, 'selected_tiles': selected_tiles,
            'word_positions': word_positions, 'original_tiles': original_tiles, 'original_rack': original_rack,
            'preview_score_enabled': preview_score_enabled, 'all_moves': all_moves, # Ensure all_moves is packed back
            'racks': racks, 'bag': bag, 'blanks': blanks, 'tiles': tiles, 'scores': scores, 'move_history': move_history,
            'last_left_click_pos': last_left_click_pos, 'last_left_click_time': last_left_click_time,
            'hint_moves': hint_moves,
            'stats_dialog_x': stats_dialog_x, 'stats_dialog_y': stats_dialog_y,
            'stats_scroll_offset': stats_scroll_offset,
            'stats_dialog_drag_offset': stats_dialog_drag_offset, 'stats_total_content_height': stats_total_content_height, # Pack stats total height
            'all_words_scroll_offset': all_words_scroll_offset,
            'simulation_config_inputs': simulation_config_inputs,
            'dialog_x': dialog_x, 'dialog_y': dialog_y,
            'current_power_tile': current_power_tile,
            'practice_end_message': practice_end_message,
            'letter_checks': letter_checks,
            'number_checks': number_checks,
            'final_scores': final_scores,
            'initial_racks': initial_racks,
            # 'gaddag_loading_status': gaddag_loading_status, # Don't pack back, use global
            'restart_practice_mode': restart_practice_mode, # Pack the new flag
            'current_r': current_r, 'current_c': current_c, # Pack cursor state
            'typing_direction': typing_direction, 'typing_start': typing_start, # Pack typing direction and start
            'board_tile_counts': board_tile_counts, # Pack updated counter
            # --- MODIFICATION: Pack exchange info ---
            'best_exchange_for_hint': best_exchange_for_hint,
            'best_exchange_score_for_hint': best_exchange_score_for_hint
        })

        return updated_state





# End of Part 6



# Part 7

def get_replay_state(turn_idx, initial_racks):
    """
    Recreate the game state up to turn_idx ITERATIVELY.
    Sorts the racks alphabetically before returning.
    """
    # Start with initial state (ensure deep copies)
    tiles_state = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    blanks_state = set()
    scores_state = [0, 0]

    if not isinstance(initial_racks, list) or len(initial_racks) != 2 or \
       not isinstance(initial_racks[0], list) or not isinstance(initial_racks[1], list):
        print(f"Error: Invalid initial_racks format in get_replay_state: {initial_racks}")
        return tiles_state, blanks_state, scores_state, [[], []]

    racks_state = [initial_racks[0][:], initial_racks[1][:]]

    # Apply moves sequentially up to turn_idx
    for i in range(turn_idx):
        if i >= len(move_history):
             print(f"Warning: Replay index {i} out of bounds for move_history (len {len(move_history)})")
             break
        move = move_history[i]
        player_idx = move['player'] - 1

        # Get the rack state *before* applying this move's changes
        current_rack_before_move = racks_state[player_idx][:]
        rack_after_move = current_rack_before_move[:] # Work with a copy for this move

        if move['move_type'] == 'place':
            positions = move.get('positions', [])
            blanks_in_move = move.get('blanks', set())

            # Apply board changes and update rack based on *this* move
            for r, c, letter in positions:
                # Check if this square was empty *before* this move (using current tiles_state)
                is_newly_placed = (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and not tiles_state[r][c])

                # Place tile on board state
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    tiles_state[r][c] = letter
                    if (r, c) in blanks_in_move:
                        blanks_state.add((r, c))
                else:
                    print(f"Replay Warning: Invalid position ({r},{c}) in move {i}")
                    continue # Skip processing this invalid position

                # If it was newly placed, remove the corresponding tile from the rack_after_move
                if is_newly_placed:
                    if (r,c) in blanks_in_move:
                        if ' ' in rack_after_move: rack_after_move.remove(' ')
                        else: print(f"Replay Warning: Tried to remove blank from rack for move {i}, but not found.")
                    else:
                        if letter in rack_after_move: rack_after_move.remove(letter)
                        else: print(f"Replay Warning: Tried to remove '{letter}' from rack for move {i}, but not found.")

            # Update score
            scores_state[player_idx] += move.get('score', 0)
            # Add drawn tiles to the rack for the *next* turn's state
            rack_after_move.extend(move.get('drawn', []))
            racks_state[player_idx] = rack_after_move

        elif move['move_type'] == 'exchange':
            # Update rack: remove exchanged, add drawn
            exchanged = move.get('exchanged_tiles', [])
            for tile in exchanged:
                 if tile in rack_after_move: rack_after_move.remove(tile)
                 else: print(f"Replay Warning: Tried to remove exchanged tile '{tile}' from rack for move {i}, but not found.")
            rack_after_move.extend(move.get('drawn', []))
            racks_state[player_idx] = rack_after_move
            # Score doesn't change

        elif move['move_type'] == 'pass':
            # No change to board, score, or rack content needed for the state
            racks_state[player_idx] = rack_after_move # Rack state carries over

    # Sort racks alphabetically before returning for replay display
    for rack in racks_state:
        rack.sort() # In-place sort

    # Return the state *after* turn_idx moves have been applied
    return tiles_state, blanks_state, scores_state, racks_state





# Function to Replace: ai_turn
# REASON: Fix NameError by calling find_all_words_formed_cython instead of
#         the non-existent find_all_words_formed in the "Only Fives" filter.

def ai_turn(turn, racks, tiles, board, blanks, scores, bag, first_play, pass_count, exchange_count, consecutive_zero_point_turns, player_names, board_tile_counts, dropdown_open=False, hinting=False, showing_all_words=False, letter_checks=None): # Added board_tile_counts
    """
    Handles the AI's turn... Checks GADDAG status before generating moves or solving endgame.
    Restored simpler Power Tile pause logic.
    Uses board_tile_counts for evaluations.
    Reduced printing for batch mode.
    Calls correct Cython helper for "Only Fives" check.
    """
    global last_word, last_score, last_start, last_direction, move_history, current_replay_turn, practice_mode, GADDAG_STRUCTURE, last_played_highlight_coords
    global is_solving_endgame, USE_ENDGAME_SOLVER, USE_AI_SIMULATION, paused_for_bingo_practice
    global gaddag_loading_status # Access status flag
    # global practiced_power_tiles_this_turn # REMOVED - Not needed for this approach

    start_turn_time = time.time()
    player_idx = turn - 1
    opponent_idx = 1 - player_idx
    current_rack = racks[player_idx][:]
    bag_count = len(bag)
    # --- COMMENTED OUT ---
    # print(f"AI Player {turn} turn started. Rack: {''.join(sorted(current_rack))}, Bag: {bag_count}, Practice: {practice_mode}, Sim: {USE_AI_SIMULATION}")

    # --- Check GADDAG Status Early ---
    if gaddag_loading_status != 'loaded':
        action_chosen = 'pass' # Default to pass if GADDAG not ready
        if gaddag_loading_status == 'loading':
            print(f"AI {turn}: GADDAG still loading, passing turn.") # Keep important status prints
        elif gaddag_loading_status == 'error':
            print(f"AI {turn}: GADDAG failed to load, passing turn.") # Keep important status prints
        else:
             print(f"AI {turn}: GADDAG status unknown ('{gaddag_loading_status}'), passing turn.") # Keep important status prints

        # --- Execute forced pass ---
        move_rack_before = racks[player_idx][:]; exchanged_tiles_for_history = []; next_turn = turn; drawn_tiles = []; newly_placed = []; move_type = ''; score = 0; word = ''; positions = []; blanks_used = set(); coord = ''; word_with_blanks = ''; is_bingo = False; luck_factor = 0.0
        move_type = 'pass'; score = 0; consecutive_zero_point_turns += 1; pass_count += 1; exchange_count = 0; next_turn = 3 - turn; last_played_highlight_coords = set()
        end_turn_time = time.time(); turn_duration = end_turn_time - start_turn_time
        # Calculate pool quality before draw (even though it's a pass)
        remaining_tiles_dict_for_eval = get_remaining_tiles(racks[player_idx], board_tile_counts)
        pool_analysis_for_eval = analyze_unseen_pool(remaining_tiles_dict_for_eval)
        pool_quality_score = pool_analysis_for_eval.get('quality_score', 0.0)
        move_data = { 'player': turn, 'move_type': move_type, 'rack': move_rack_before, 'score': score, 'word': word, 'positions': positions, 'blanks': blanks_used, 'drawn': drawn_tiles, 'coord': coord, 'word_with_blanks': word_with_blanks, 'is_bingo': is_bingo, 'turn_duration': turn_duration, 'pool_quality_before_draw': pool_quality_score, 'luck_factor': luck_factor }
        move_history.append(move_data); current_replay_turn = len(move_history)
        return next_turn, first_play, pass_count, exchange_count, consecutive_zero_point_turns, [], dropdown_open, hinting, showing_all_words, False, None, False, set()
    # --- End GADDAG Status Check ---

    # --- Calculate current pool quality ---
    remaining_tiles_dict_for_eval = get_remaining_tiles(racks[player_idx], board_tile_counts)
    pool_analysis_for_eval = analyze_unseen_pool(remaining_tiles_dict_for_eval)
    pool_quality_score = pool_analysis_for_eval.get('quality_score', 0.0)
    # --- COMMENTED OUT ---
    # print(f"  Current Unseen Pool Quality Score: {pool_quality_score:.2f}")

    # --- Endgame Solver Check (GADDAG must be loaded here) ---
    if USE_ENDGAME_SOLVER and bag_count == 0 and practice_mode != "eight_letter" and not is_solving_endgame:
        print("AI: Bag empty and solver enabled, entering endgame solver...") # Keep this important message
        opponent_rack = racks[opponent_idx][:]; current_score_diff = scores[player_idx] - scores[opponent_idx]; is_solving_endgame = True
        # --- Drawing for solver indicator (keep if desired, but adds overhead) ---
        # screen.fill(WHITE)
        # ... (drawing code) ...
        # draw_endgame_solving_indicator()
        # pygame.display.flip()
        # --- End Drawing ---
        best_first_move = solve_endgame(current_rack, opponent_rack, tiles, blanks, board, current_score_diff)
        action_chosen = 'pass'; best_play_move = None
        if best_first_move == "PASS": action_chosen = 'pass'
        elif isinstance(best_first_move, dict): action_chosen = 'play'; best_play_move = best_first_move
        else: action_chosen = 'pass'
        move_rack_before = racks[player_idx][:]; exchanged_tiles_for_history = []; next_turn = turn; drawn_tiles = []; newly_placed = []; move_type = ''; score = 0; word = ''; positions = []; blanks_used = set(); coord = ''; word_with_blanks = ''; is_bingo = False
        if action_chosen == 'play' and best_play_move:
            next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(best_play_move, tiles, racks, blanks, scores, turn, bag, board, board_tile_counts)
            move_type = 'place'; score = best_play_move.get('score', 0); word = best_play_move.get('word', 'N/A'); positions = best_play_move.get('positions', []); blanks_used = best_play_move.get('blanks', set()); start = best_play_move.get('start', (0,0)); direction = best_play_move.get('direction', 'right'); coord = get_coord(start, direction); word_with_blanks = best_play_move.get('word_with_blanks', ''); is_bingo = best_play_move.get('is_bingo', False)
            first_play = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0; last_played_highlight_coords = set((pos[0], pos[1]) for pos in positions)
        elif action_chosen == 'pass':
            move_type = 'pass'; score = 0; consecutive_zero_point_turns += 1; pass_count += 1; exchange_count = 0; next_turn = 3 - turn; last_played_highlight_coords = set()
        end_turn_time = time.time(); turn_duration = end_turn_time - start_turn_time
        # --- COMMENTED OUT ---
        # print(f"AI turn took {turn_duration:.2f} seconds.")
        move_data = { 'player': turn, 'move_type': move_type, 'rack': move_rack_before, 'score': score, 'word': word, 'positions': positions, 'blanks': blanks_used, 'drawn': drawn_tiles, 'coord': coord, 'word_with_blanks': word_with_blanks, 'is_bingo': is_bingo, 'turn_duration': turn_duration, 'pool_quality_before_draw': pool_quality_score, 'luck_factor': 0.0 }
        if move_type == 'place' and best_play_move: move_data['newly_placed'] = best_play_move.get('newly_placed', newly_placed)
        move_history.append(move_data); current_replay_turn = len(move_history)
        return next_turn, first_play, pass_count, exchange_count, consecutive_zero_point_turns, [], dropdown_open, hinting, showing_all_words, False, None, False, set()


    # --- Normal Turn Logic (GADDAG must be loaded here) ---
    paused_for_power_tile = False
    paused_for_bingo_practice = False
    current_power_tile = None
    all_moves = []
    best_play_move = None
    action_chosen = 'pass'

    # Generate moves (GADDAG is loaded if we reached here)
    all_moves = generate_all_moves_gaddag(racks[player_idx], tiles, board, blanks, GADDAG_STRUCTURE.root)
    if all_moves is None: all_moves = [] # Should not happen if GADDAG loaded, but safety
    # --- COMMENTED OUT ---
    # print(f"AI Player {turn} generated {len(all_moves)} raw moves.")

    # Filter moves for "Only Fives"
    if practice_mode == "only_fives":
        original_tiles_copy = [row[:] for row in tiles]; original_blanks_copy = blanks.copy(); filtered_moves = []
        for move in all_moves:
            temp_tiles = [row[:] for row in original_tiles_copy]; temp_blanks = original_blanks_copy.copy(); newly_placed_details = move.get('newly_placed', [])
            if not newly_placed_details: continue
            for r, c, letter in newly_placed_details:
                 if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                     temp_tiles[r][c] = letter
                     if (r, c) in move.get('blanks', set()): temp_blanks.add((r, c))
            # --- FIX: Call Cython version ---
            words_formed_details = find_all_words_formed_cython(newly_placed_details, temp_tiles)
            # --- END FIX ---
            if any(len("".join(t[2] for t in word_detail)) == 5 for word_detail in words_formed_details): filtered_moves.append(move)
        all_moves = filtered_moves;
        # --- COMMENTED OUT ---
        # print(f"AI Player {turn} filtered moves for 'Only Fives'. Remaining: {len(all_moves)}")

    # --- Practice Mode Pause Checks ---
    if practice_mode == "power_tiles" and letter_checks:
        checked_power_tiles = {letter for i, letter in enumerate(['J', 'Q', 'X', 'Z']) if letter_checks[i]}
        power_tiles_on_rack = sorted([tile for tile in current_rack if tile in checked_power_tiles])
        if power_tiles_on_rack:
            current_power_tile = power_tiles_on_rack[0]
            paused_for_power_tile = True
            print(f"AI turn paused for power tile practice. Target: {current_power_tile}") # Keep pause message
            end_turn_time = time.time(); turn_duration = end_turn_time - start_turn_time
            # --- COMMENTED OUT ---
            # print(f"AI turn took {turn_duration:.2f} seconds (paused).")
            return turn, first_play, pass_count, exchange_count, consecutive_zero_point_turns, all_moves, dropdown_open, hinting, showing_all_words, paused_for_power_tile, current_power_tile, False, set()
    elif practice_mode == "bingo_bango_bongo":
        found_bingo = any(move.get('is_bingo', False) for move in all_moves)
        if found_bingo:
            paused_for_bingo_practice = True
            print(f"AI turn paused for Bingo, Bango, Bongo practice. Bingo found.") # Keep pause message
            end_turn_time = time.time(); turn_duration = end_turn_time - start_turn_time
            # --- COMMENTED OUT ---
            # print(f"AI turn took {turn_duration:.2f} seconds (paused).")
            return turn, first_play, pass_count, exchange_count, consecutive_zero_point_turns, all_moves, dropdown_open, hinting, showing_all_words, False, None, paused_for_bingo_practice, set()

    # --- AI Action Decision Logic ---
    move_rack_before = racks[player_idx][:]; exchanged_tiles_for_history = []; best_play_evaluation = -float('inf'); best_exchange_tiles = []; best_exchange_evaluation = -float('inf')
    can_play = bool(all_moves); can_exchange_proactively = bag_count >= 7
    run_simulation = (USE_AI_SIMULATION and game_mode in [MODE_HVA, MODE_AVA] and practice_mode is None and can_play)

    if run_simulation:
        opponent_rack_len_sim = len(racks[opponent_idx]) if opponent_idx < len(racks) else 7
        simulation_results = run_ai_simulation(current_rack, opponent_rack_len_sim, tiles, blanks, board, bag, GADDAG_STRUCTURE.root, first_play, board_tile_counts)
        if simulation_results:
             best_play_move = simulation_results[0]['move']
             best_play_evaluation = simulation_results[0]['final_score']
             # --- COMMENTED OUT ---
             # print(f"  (Simulation provided play candidate: '{best_play_move.get('word','N/A')}', Sim Score: {best_play_evaluation:.1f})")
        else:
             # --- COMMENTED OUT ---
             # print("  Simulation returned no valid play. Falling back to standard evaluation.")
             run_simulation = False; best_play_move = None

    if not run_simulation and can_play:
        # --- COMMENTED OUT ---
        # print(f"AI {turn} evaluating {len(all_moves)} plays (Standard Eval)...")
        current_top_raw_score = -float('inf'); top_raw_score_move = None
        for move in all_moves:
            evaluated_score = evaluate_single_move(move, evaluate_leave); immediate_score = move.get('score', 0)
            if evaluated_score > best_play_evaluation: best_play_evaluation = evaluated_score; best_play_move = move
            if immediate_score > current_top_raw_score: current_top_raw_score = immediate_score; top_raw_score_move = move
        # --- COMMENTED OUT ---
        # print("-" * 20); print(f"AI {turn} Standard Play Evaluation Complete:")
        # if top_raw_score_move: print(f"  Top Raw Score Move : '{top_raw_score_move.get('word','N/A')}' ({top_raw_score_move.get('score',0)} pts)")
        # else: print("  Top Raw Score Move : None Found")
        # if best_play_move:
        #     chosen_score = best_play_move.get('score', 0); chosen_leave = best_play_move.get('leave', []); chosen_leave_str = ''.join(sorted(chosen_leave)); print(f"  Best Evaluated Play: '{best_play_move.get('word','N/A')}' (Raw Score: {chosen_score})"); print(f"    Leave Tiles: '{chosen_leave_str}'")
        #     leave_adjust = evaluate_leave(chosen_leave, verbose=False); print(f"    --------------------"); print(f"    Final Play Eval Score: {float(chosen_score + leave_adjust):.2f} (Raw: {chosen_score}, Leave Adj: {leave_adjust})")
        # else: print("  Best Evaluated Play: None Found")
        # print("-" * 20)

    if not can_play or can_exchange_proactively:
        # --- COMMENTED OUT ---
        # print(f"AI {turn} evaluating exchange options...")
        best_exchange_tiles, best_exchange_evaluation = find_best_exchange_option(racks[player_idx], remaining_tiles_dict_for_eval, bag_count)
        # --- COMMENTED OUT ---
        # if best_exchange_tiles: print(f"  Best Exchange Option Found: Exchange {len(best_exchange_tiles)} tiles ({''.join(sorted(best_exchange_tiles))}), Est Value: {best_exchange_evaluation:.1f}")
        # else: print("  No beneficial exchange option found.")

    if can_play and best_play_move:
        action_chosen = 'play'
        if can_exchange_proactively and best_exchange_tiles:
            current_play_eval = best_play_evaluation
            if best_exchange_evaluation > current_play_eval:
                # --- COMMENTED OUT ---
                # print(f"AI {turn}: DECISION -> Proactive Exchange (Exchange eval {best_exchange_evaluation:.1f} > Play eval {current_play_eval:.1f})")
                action_chosen = 'exchange'
            # else: # --- COMMENTED OUT ---
                # print(f"AI {turn}: DECISION -> Play ('{best_play_move.get('word','N/A')}') (Exchange eval {best_exchange_evaluation:.1f} <= Play eval {current_play_eval:.1f})")
    elif best_exchange_tiles:
        # --- COMMENTED OUT ---
        # print(f"AI {turn}: DECISION -> Mandatory Exchange (No plays possible, exchange evaluated as beneficial)")
        action_chosen = 'exchange'
    # else: # --- COMMENTED OUT ---
        # print(f"AI {turn}: DECISION -> Pass (No plays, no viable/beneficial exchange)")
        action_chosen = 'pass'

    # --- Execute Action ---
    next_turn = turn; drawn_tiles = []; newly_placed = []; move_type = ''; score = 0; word = ''; positions = []; blanks_used = set(); coord = ''; word_with_blanks = ''; is_bingo = False; luck_factor = 0.0
    if action_chosen == 'play':
        if best_play_move:
            word_to_print = best_play_move.get('word', 'N/A'); start_pos_print = best_play_move.get('start', (0,0)); direction_print = best_play_move.get('direction', 'right'); coord_print = get_coord(start_pos_print, direction_print);
            # --- COMMENTED OUT ---
            # print(f"AI {turn} playing move: '{word_to_print}' at {coord_print}")
            next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(best_play_move, tiles, racks, blanks, scores, turn, bag, board, board_tile_counts)
            move_type = 'place'; score = best_play_move.get('score', 0); word = word_to_print; positions = best_play_move.get('positions', []); blanks_used = best_play_move.get('blanks', set()); coord = coord_print; word_with_blanks = best_play_move.get('word_with_blanks', ''); is_bingo = best_play_move.get('is_bingo', False)
            first_play = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0; last_played_highlight_coords = set((pos[0], pos[1]) for pos in positions)
        else: print(f"AI {turn} Error: Action was 'play' but best_play_move is None. Passing."); action_chosen = 'pass' # Keep error message
    if action_chosen == 'exchange':
        if best_exchange_tiles:
            # --- COMMENTED OUT ---
            # print(f"AI {turn} exchanging {len(best_exchange_tiles)} tiles: {''.join(sorted(best_exchange_tiles))}")
            exchanged_tiles_for_history = best_exchange_tiles[:]; current_rack_list = racks[player_idx]; temp_rack_after_exchange = []; exchange_counts = Counter(exchanged_tiles_for_history)
            for tile in current_rack_list:
                if exchange_counts.get(tile, 0) > 0: exchange_counts[tile] -= 1
                else: temp_rack_after_exchange.append(tile)
            num_to_draw = len(exchanged_tiles_for_history); drawn_tiles = [bag.pop() for _ in range(num_to_draw) if bag]; temp_rack_after_exchange.extend(drawn_tiles); racks[player_idx] = temp_rack_after_exchange
            if not is_ai[player_idx]: racks[player_idx].sort(); bag.extend(exchanged_tiles_for_history); random.shuffle(bag)
            move_type = 'exchange'; score = 0; consecutive_zero_point_turns += 1; exchange_count += 1; pass_count = 0; next_turn = 3 - turn; last_played_highlight_coords = set()
        else: print(f"AI {turn} Error: Action was 'exchange' but best_exchange_tiles is empty. Passing."); action_chosen = 'pass' # Keep error message
    if action_chosen == 'pass':
        # --- COMMENTED OUT ---
        # print(f"AI {turn} passing.")
        move_type = 'pass'; score = 0; consecutive_zero_point_turns += 1; pass_count += 1; exchange_count = 0; next_turn = 3 - turn; last_played_highlight_coords = set()

    if drawn_tiles:
        drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple;
        # --- COMMENTED OUT ---
        # drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")

    end_turn_time = time.time(); turn_duration = end_turn_time - start_turn_time
    # --- COMMENTED OUT ---
    # print(f"AI turn took {turn_duration:.2f} seconds.")
    move_data = { 'player': turn, 'move_type': move_type, 'rack': move_rack_before, 'score': score, 'word': word, 'positions': positions, 'blanks': blanks_used, 'drawn': drawn_tiles, 'coord': coord, 'word_with_blanks': word_with_blanks, 'is_bingo': is_bingo, 'turn_duration': turn_duration, 'pool_quality_before_draw': pool_quality_score, 'luck_factor': luck_factor }
    if move_type == 'exchange': move_data['exchanged_tiles'] = exchanged_tiles_for_history
    if move_type == 'place' and best_play_move: move_data['newly_placed'] = best_play_move.get('newly_placed', newly_placed)
    move_history.append(move_data); current_replay_turn = len(move_history)

    return next_turn, first_play, pass_count, exchange_count, consecutive_zero_point_turns, [], dropdown_open, hinting, showing_all_words, paused_for_power_tile, current_power_tile, paused_for_bingo_practice, set()









#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
***************************************************************
***************************************************************
*******************NEW EXTRACTED FUNCTIONS*********************
***************************************************************
***************************************************************
***************************************************************
'''
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$




# Function to Replace: initialize_game
# REASON: Add debug print after resetting practice_target_moves for 8-letter mode.

def initialize_game(selected_mode_result, return_data, main_called_flag):
    """
    Initializes the game state based on the selected mode and data.
    Starts background GADDAG loading if not already loaded/loading.
    Returns the current GADDAG loading status along with other state variables.
    Initializes and populates board_tile_counts.
    Unpacks visualize_batch setting for batch mode.
    Adds debug print for 8-letter practice init.
    """
    global GADDAG_STRUCTURE, gaddag_loading_status, gaddag_load_thread # Allow modification

    print("--- initialize_game() entered ---")

    # --- Start Background GADDAG Load (Only Once) ---
    if not main_called_flag and gaddag_loading_status == 'idle':
        print("--- initialize_game(): Starting GADDAG background load... ---")
        gaddag_loading_status = 'loading'
        GADDAG_STRUCTURE = None # Ensure it's None while loading
        gaddag_load_thread = threading.Thread(target=_load_gaddag_background, daemon=True) # Use daemon thread
        gaddag_load_thread.start()
    elif gaddag_loading_status == 'loading':
        print("--- initialize_game(): GADDAG is already loading in the background. ---")
    elif gaddag_loading_status == 'loaded':
        print("--- initialize_game(): GADDAG is already loaded. ---")
    elif gaddag_loading_status == 'error':
        print("--- initialize_game(): GADDAG failed to load previously. AI features disabled. ---")
        GADDAG_STRUCTURE = None # Ensure it remains None on error


    # --- Initialize variables with defaults ---
    game_mode = None
    is_loaded_game = False
    player_names = ["Player 1", "Player 2"]
    move_history = []
    final_scores = None
    replay_initial_shuffled_bag = None
    board, _, tiles = create_board() # Initialize board and tiles grid
    scores = [0, 0]
    blanks = set()
    racks = [[], []]
    bag = []
    replay_mode = False
    current_replay_turn = 0
    practice_mode = None
    is_ai = [False, False]
    human_player = 1
    first_play = True
    initial_racks = [[], []]
    number_checks = [True] * 6 # Default
    letter_checks = [True] * 4 # Default
    USE_ENDGAME_SOLVER = False
    USE_AI_SIMULATION = False
    is_batch_running = False
    total_batch_games = 0
    current_batch_game_num = 0
    batch_results = []
    initial_game_config = {}
    practice_target_moves = []
    practice_best_move = None
    all_moves = []
    turn = 1
    pass_count = 0
    exchange_count = 0
    consecutive_zero_point_turns = 0
    last_played_highlight_coords = set()
    is_solving_endgame = False
    board_tile_counts = Counter() # Initialize the new counter
    visualize_batch = False # Default visualize_batch setting

    # --- Game State Initialization based on selected mode ---
    print(f"--- initialize_game(): Starting game state initialization for mode: {selected_mode_result} ---")
    if selected_mode_result == "LOADED_GAME":
        # ... (LOADED_GAME logic - unchanged) ...
        print("--- initialize_game(): Handling LOADED_GAME setup ---")
        game_mode = "LOADED_GAME"; is_loaded_game = True
        player_names, loaded_history, final_scores_loaded = return_data
        move_history = loaded_history; final_scores = final_scores_loaded
        base_bag = create_standard_bag(); random.shuffle(base_bag); replay_initial_shuffled_bag = base_bag[:]
        scores = [0, 0]; blanks = set(); racks = [[], []]; bag = []
        replay_mode = True; current_replay_turn = 0; practice_mode = None; is_ai = [False, False]; human_player = 1; first_play = False
        initial_racks = [[], []]
        number_checks = [True] * 6
        USE_ENDGAME_SOLVER = True
        USE_AI_SIMULATION = False
        tiles_loaded, _, _, _ = simulate_game_up_to(0, move_history, replay_initial_shuffled_bag)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if tiles_loaded[r][c]:
                    board_tile_counts[tiles_loaded[r][c]] += 1
        print(f"--- initialize_game(): Loaded Game Setup Complete. Players: {player_names}, Moves: {len(move_history)} ---")

    elif selected_mode_result == "BATCH_MODE":
        is_batch_running = True
        # --- MODIFICATION: Unpack visualize_batch_checked ---
        try:
            # Expect 7 items now
            game_mode, player_names, human_player, use_endgame_solver_checked, use_ai_simulation_checked, num_games, visualize_batch_checked = return_data
            visualize_batch = visualize_batch_checked # Store the setting
        except ValueError:
            print("Error: Incorrect number of values unpacked for BATCH_MODE setup. Using defaults.")
            # Handle error case - maybe default visualize_batch to False or True?
            game_mode, player_names, human_player, use_endgame_solver_checked, use_ai_simulation_checked, num_games = return_data # Unpack original 6
            visualize_batch = False # Default if unpacking failed
        # --- END MODIFICATION ---

        total_batch_games = num_games
        current_batch_game_num = 1
        batch_results = []
        USE_ENDGAME_SOLVER = use_endgame_solver_checked
        USE_AI_SIMULATION = use_ai_simulation_checked
        practice_mode = None
        letter_checks = [True]*4
        number_checks = [True]*6

        batch_now = datetime.datetime.now()
        batch_date_str = batch_now.strftime("%d%b%y").upper()
        batch_time_str = batch_now.strftime("%H%M")
        batch_seq_num = 1; max_existing_batch_num = 0
        try:
            for filename in os.listdir('.'):
                if filename.startswith(f"{batch_date_str}-") and filename.endswith(".txt") and "-BATCH-" in filename:
                    parts = filename[:-4].split('-');
                    if len(parts) == 4 and parts[2] == "BATCH" and parts[-1].isdigit():
                        num = int(parts[-1]); max_existing_batch_num = max(max_existing_batch_num, num)
            batch_seq_num = max_existing_batch_num + 1
        except OSError as e: print(f"Warning: Error listing directory for batch sequence number: {e}. Using sequence 1.")
        batch_base_filename_prefix = f"{batch_date_str}-{batch_time_str}-BATCH-{batch_seq_num}"

        # --- MODIFICATION: Add visualize_batch to initial_game_config ---
        initial_game_config = {
            'game_mode': game_mode, 'player_names': player_names, 'human_player': human_player,
            'use_endgame_solver': USE_ENDGAME_SOLVER, 'use_ai_simulation': USE_AI_SIMULATION,
            'batch_filename_prefix': batch_base_filename_prefix,
            'visualize_batch': visualize_batch # Store the setting here
        }
        # --- END MODIFICATION ---
        is_ai = [False, False]
        if initial_game_config['game_mode'] == MODE_HVA: is_ai[2 - initial_game_config['human_player']] = True
        elif initial_game_config['game_mode'] == MODE_AVA: is_ai = [True, True]
        initial_game_config['is_ai'] = is_ai

        reset_result = reset_game_state(initial_game_config)
        if reset_result is None:
             print("FATAL: Could not initialize first batch game.")
             pygame.quit(); sys.exit()

        (board, tiles, racks, blanks, scores, turn, first_play, bag,
         move_history, pass_count, exchange_count, consecutive_zero_point_turns,
         last_played_highlight_coords, is_solving_endgame, board_tile_counts) = reset_result # Unpack counter
        initial_racks = [r[:] for r in racks]
        print(f"--- initialize_game(): Batch Mode Setup Complete. Running {total_batch_games} games. Visualize: {visualize_batch}. Base Filename Prefix: {batch_base_filename_prefix} ---")

    elif selected_mode_result is not None: # Normal New Game or Practice
        print(f"--- initialize_game(): Handling New Game Setup ({selected_mode_result}) ---")
        game_mode = selected_mode_result; is_loaded_game = False; replay_initial_shuffled_bag = None
        # --- MODIFICATION: Unpack visualize_batch (though unused here) ---
        try:
            # Expect 9 items now
            player_names, human_player, practice_mode, letter_checks, number_checks, use_endgame_solver_checked, use_ai_simulation_checked, practice_state, visualize_batch_checked = return_data
            # visualize_batch = visualize_batch_checked # Store if needed later
        except ValueError:
             print("Error: Incorrect number of values unpacked for New Game setup. Using defaults.")
             player_names, human_player, practice_mode, letter_checks, number_checks, use_endgame_solver_checked, use_ai_simulation_checked, practice_state = return_data # Original 8
             # visualize_batch = False # Default
        # --- END MODIFICATION ---

        USE_ENDGAME_SOLVER = use_endgame_solver_checked
        USE_AI_SIMULATION = use_ai_simulation_checked
        print(f"--- initialize_game(): Use Endgame Solver set to: {USE_ENDGAME_SOLVER} ---")
        print(f"--- initialize_game(): Use AI Simulation set to: {USE_AI_SIMULATION} ---")

        if practice_state and practice_mode == "eight_letter":
            # ... (8-letter practice setup - unchanged) ...
            print("Loading state from 8-letter practice...");
            board, tiles, racks, blanks, bag, scores, turn, first_play = practice_state["board"], practice_state["tiles"], practice_state["racks"], practice_state["blanks"], practice_state["bag"], practice_state["scores"], practice_state["turn"], practice_state["first_play"]; is_ai = [False, False];
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if tiles[r][c]:
                        board_tile_counts[tiles[r][c]] += 1
            print("--- initialize_game(): Loaded state from 8-letter practice. ---")
            # --- ADDED DEBUG PRINT ---
            practice_target_moves = [] # Reset here
            print(f"  DEBUG initialize_game (8-letter): Reset practice_target_moves. Length is now: {len(practice_target_moves)}")
            # --- END ADDED DEBUG PRINT ---
            practice_best_move = None
            all_moves = []
        elif practice_state: # Assuming other practice modes might also have initial state
            # ... (Other practice setup - unchanged) ...
            print("Loading state from other practice mode...");
            board, tiles, racks, blanks, bag, scores, turn, first_play = practice_state["board"], practice_state["tiles"], practice_state["racks"], practice_state["blanks"], practice_state["bag"], practice_state["scores"], practice_state["turn"], practice_state["first_play"]; is_ai = [False, False];
            if game_mode == MODE_HVA: is_ai[2 - human_player] = True
            elif game_mode == MODE_AVA or practice_mode == "power_tiles" or practice_mode == "bingo_bango_bongo": is_ai = [True, True];
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if tiles[r][c]:
                        board_tile_counts[tiles[r][c]] += 1
            print(f"--- initialize_game(): Loaded state from other practice mode. is_ai: {is_ai} ---")
        else: # Standard new game
            # ... (Standard new game setup - unchanged) ...
            print("Performing standard game initialization...");
            bag = create_standard_bag(); random.shuffle(bag); racks = [[], []]; scores = [0, 0]; turn = 1; blanks = set(); first_play = True
            board_tile_counts = Counter() # Ensure it's empty for standard start
            try: racks[0] = [bag.pop() for _ in range(7)]; racks[1] = [bag.pop() for _ in range(7)]
            except IndexError: print("Error: Not enough tiles in bag."); pygame.quit(); sys.exit()
            is_ai = [False, False]
            if game_mode == MODE_HVA: is_ai[2 - human_player] = True
            elif game_mode == MODE_AVA: is_ai = [True, True];
            print(f"--- initialize_game(): Initialized is_ai: {is_ai} ---")
            for i, rack in enumerate(racks):
                if 0 <= i < len(is_ai) and not is_ai[i]: rack.sort()
            print(f"--- initialize_game(): Performed standard game initialization. is_ai: {is_ai} ---")

        initial_racks = [rack[:] for rack in racks]

        # --- MOVED 8-letter specific init here ---
        # if practice_mode == "eight_letter":
        #     print("--- initialize_game(): Performing 8-letter practice specific init (move gen deferred)... ---")
        #     practice_target_moves = []
        #     practice_best_move = None
        #     all_moves = []
        # --- END MOVED ---

    elif selected_mode_result is None:
        print("--- initialize_game(): Mode selection returned None. Exiting. ---")
        pygame.quit(); sys.exit()
        return None

    print("--- initialize_game(): Initialization complete. Returning state. ---")
    # --- MODIFICATION: Add visualize_batch to the return tuple ---
    return (game_mode, is_loaded_game, player_names, move_history, final_scores,
            replay_initial_shuffled_bag, board, tiles, scores, blanks, racks, bag,
            replay_mode, current_replay_turn, practice_mode, is_ai, human_player,
            first_play, initial_racks, number_checks, USE_ENDGAME_SOLVER,
            USE_AI_SIMULATION, is_batch_running, total_batch_games,
            current_batch_game_num, batch_results, initial_game_config,
            GADDAG_STRUCTURE, practice_target_moves, practice_best_move, all_moves,
            letter_checks, turn, pass_count, exchange_count, consecutive_zero_point_turns,
            last_played_highlight_coords, is_solving_endgame, gaddag_loading_status,
            board_tile_counts, visualize_batch) # Added visualize_batch here







def draw_board_labels(screen, ui_font):
    """Draws the row (1-15) and column (A-O) labels around the board."""
    # Draw Row Labels (1-15)
    for r in range(GRID_SIZE):
        row_label = ui_font.render(str(r + 1), True, BLACK)
        screen.blit(row_label, (10, 40 + r * SQUARE_SIZE + (SQUARE_SIZE // 2 - row_label.get_height() // 2)))
    # Draw Column Labels (A-O)
    for c in range(GRID_SIZE):
        col_label = ui_font.render(LETTERS[c], True, BLACK)
        screen.blit(col_label, (40 + c * SQUARE_SIZE + (SQUARE_SIZE // 2 - col_label.get_width() // 2), 10))



def draw_player_racks(screen, racks_to_display, scores_to_display, turn_to_display, player_names, dragged_tile, drag_pos, practice_mode):
    """
    Draws the racks for both players (conditionally for P2 in practice mode).

    Args:
        screen: The Pygame surface to draw on.
        racks_to_display: List containing the racks [[P1_rack], [P2_rack]].
        scores_to_display: List of scores [P1_score, P2_score].
        turn_to_display: The current turn number (1 or 2).
        player_names: List of player names ["P1_name", "P2_name"].
        dragged_tile: Tuple (player, index) if a tile is being dragged, else None.
        drag_pos: Tuple (x, y) of the dragged tile's position, else None.
        practice_mode: String indicating the current practice mode, or None.

    Returns:
        tuple: (p1_alpha_rect, p1_rand_rect, p2_alpha_rect, p2_rand_rect)
               Contains the pygame.Rect objects for the alphabetize and randomize
               buttons for both players (P2 rects will be None if not drawn).
    """
    # Draw Player 1 Rack
    p1_rack_to_draw = racks_to_display[0] if len(racks_to_display) > 0 else []
    p1_drag_info = dragged_tile if dragged_tile and dragged_tile[0] == 1 else None
    p1_alpha_rect, p1_rand_rect = draw_rack(1, p1_rack_to_draw, scores_to_display, turn_to_display, player_names, p1_drag_info, drag_pos)

    # Draw Player 2 Rack (conditionally)
    p2_alpha_rect, p2_rand_rect = None, None
    if practice_mode != "eight_letter":
        p2_rack_to_draw = racks_to_display[1] if len(racks_to_display) > 1 else []
        p2_drag_info = dragged_tile if dragged_tile and dragged_tile[0] == 2 else None
        # Note: draw_rack handles the case where p2_rack_to_draw might be empty
        p2_alpha_rect, p2_rand_rect = draw_rack(2, p2_rack_to_draw, scores_to_display, turn_to_display, player_names, p2_drag_info, drag_pos)

    return p1_alpha_rect, p1_rand_rect, p2_alpha_rect, p2_rand_rect




# Function to Replace: handle_mouse_down_event (Complete)
# REASON: Add debug print immediately before the practice_mode check
#         within the Suggest button handler.

def handle_mouse_down_event(event, state, drawn_rects): # Added drawn_rects parameter
        """
        Handles MOUSEBUTTONDOWN events for the main game loop.
        Checks GADDAG status AND structure existence before Suggest/Simulate actions.
        Adds validation for "Only Fives" mode when playing from dialogs.
        Corrects 8-Letter Bingo hint play logic.
        Ensures cursor state variables (current_r, current_c) are always initialized.
        Corrects Suggest button move generation/usage.
        Uses drawn_rects dictionary for UI element collision checks.
        Fixes bug where hint play history was attributed to the wrong player.
        Fixes bug where practice mode plays from dialogs were recorded twice by returning early.
        Adds debug prints for invalid typed play issue.
        Reads global gaddag_loading_status directly for AI actions.
        Passes board_tile_counts to play_hint_move and run_ai_simulation.
        Calls find_best_exchange_option after simulation for human hints.
        Corrects Suggest logic AGAIN for 8-Letter Bingo mode.
        Adds debug print before practice_mode check in Suggest handler.
        """
        # --- Access global directly for GADDAG status check ---
        global gaddag_loading_status, GADDAG_STRUCTURE # Need GADDAG_STRUCTURE too

        # Unpack necessary state variables from the dictionary
        turn = state['turn']; dropdown_open = state['dropdown_open']; bag_count = state['bag_count']; is_batch_running = state['is_batch_running']; replay_mode = state['replay_mode']; game_over_state = state['game_over_state']; practice_mode = state['practice_mode']; exchanging = state['exchanging']; hinting = state['hinting']; showing_all_words = state['showing_all_words']; specifying_rack = state['specifying_rack']; showing_simulation_config = state['showing_simulation_config']; showing_practice_end_dialog = state['showing_practice_end_dialog']; confirming_override = state['confirming_override']; final_scores = state['final_scores']; player_names = state['player_names']; move_history = state['move_history']; initial_racks = state['initial_racks']; showing_stats = state['showing_stats']; stats_dialog_x = state['stats_dialog_x']; stats_dialog_y = state['stats_dialog_y']; dialog_x = state['dialog_x']; dialog_y = state['dialog_y']; current_replay_turn = state['current_replay_turn']; selected_tiles = state['selected_tiles']; is_ai = state['is_ai']; specify_rack_original_racks = state['specify_rack_original_racks']; specify_rack_inputs = state['specify_rack_inputs']; specify_rack_active_input = state['specify_rack_active_input']; specify_rack_proposed_racks = state['specify_rack_proposed_racks']; racks = state['racks']; bag = state['bag']; blanks = state['blanks']; tiles = state['tiles']; scores = state['scores']; paused_for_power_tile = state['paused_for_power_tile']; paused_for_bingo_practice = state['paused_for_bingo_practice']; practice_best_move = state['practice_best_move']; all_moves = state['all_moves']; current_power_tile = state['current_power_tile']; number_checks = state['number_checks']; board = state['board']; current_turn_pool_quality_score = state['current_turn_pool_quality_score']; first_play = state['first_play']; pass_count = state['pass_count']; exchange_count = state['exchange_count']; consecutive_zero_point_turns = state['consecutive_zero_point_turns']; last_played_highlight_coords = state['last_played_highlight_coords'];
        practice_solved = state['practice_solved']; # RE-ADD unpacking
        practice_end_message = state['practice_end_message']; simulation_config_inputs = state['simulation_config_inputs']; simulation_config_active_input = state['simulation_config_active_input']; hint_moves = state['hint_moves']; selected_hint_index = state['selected_hint_index']; preview_score_enabled = state['preview_score_enabled']; dragged_tile = state['dragged_tile']; drag_pos = state['drag_pos']; drag_offset = state['drag_offset']; typing = state['typing']; word_positions = state['word_positions']; original_tiles = state['original_tiles']; original_rack = state['original_rack']; selected_square = state['selected_square']; last_left_click_time = state['last_left_click_time']; last_left_click_pos = state['last_left_click_pos']; stats_dialog_dragging = state['stats_dialog_dragging']; dragging = state['dragging']; letter_checks = state['letter_checks']
        # gaddag_loading_status = state['gaddag_loading_status'] # REMOVED - Use global directly
        stats_scroll_offset = state['stats_scroll_offset'] # Unpack stats scroll offset
        stats_dialog_drag_offset = state['stats_dialog_drag_offset'] # Unpack stats drag offset
        all_words_scroll_offset = state['all_words_scroll_offset'] # Unpack all words scroll offset
        restart_practice_mode = state['restart_practice_mode'] # Unpack flag
        stats_total_content_height = state.get('stats_total_content_height', 0) # Use .get() for safety
        board_tile_counts = state['board_tile_counts'] # Unpack the counter
        practice_target_moves = state['practice_target_moves'] # <<< Unpack practice_target_moves
        # --- Initialize local cursor variables from state ---
        current_r = state.get('current_r')
        current_c = state.get('current_c')
        typing_direction = state.get('typing_direction')
        typing_start = state.get('typing_start')
        # --- End Initialize local cursor variables ---
        # --- Unpack exchange hint info ---
        best_exchange_for_hint = state.get('best_exchange_for_hint') # Use .get for safety
        best_exchange_score_for_hint = state.get('best_exchange_score_for_hint', -float('inf')) # Use .get for safety

        # --- Get Rects from the drawn_rects dictionary ---
        practice_play_again_rect = drawn_rects.get('practice_play_again_rect')
        practice_main_menu_rect = drawn_rects.get('practice_main_menu_rect')
        practice_quit_rect = drawn_rects.get('practice_quit_rect')
        sim_input_rects = drawn_rects.get('sim_input_rects', [])
        sim_simulate_rect = drawn_rects.get('sim_simulate_rect')
        sim_cancel_rect = drawn_rects.get('sim_cancel_rect')
        go_back_rect_ov = drawn_rects.get('go_back_rect_ov')
        override_rect_ov = drawn_rects.get('override_rect_ov')
        p1_input_rect_sr = drawn_rects.get('p1_input_rect_sr')
        p2_input_rect_sr = drawn_rects.get('p2_input_rect_sr')
        p1_reset_rect_sr = drawn_rects.get('p1_reset_rect_sr')
        p2_reset_rect_sr = drawn_rects.get('p2_reset_rect_sr')
        confirm_rect_sr = drawn_rects.get('confirm_rect_sr')
        cancel_rect_sr = drawn_rects.get('cancel_rect_sr')
        options_rect_base = drawn_rects.get('options_rect_base')
        dropdown_rects_base = drawn_rects.get('dropdown_rects_base', [])
        save_rect = drawn_rects.get('save_rect')
        quit_rect = drawn_rects.get('quit_rect')
        replay_rect = drawn_rects.get('replay_rect')
        play_again_rect = drawn_rects.get('play_again_rect')
        stats_rect = drawn_rects.get('stats_rect')
        stats_ok_button_rect = drawn_rects.get('stats_ok_button_rect')
        replay_start_rect = state['replay_start_rect'] # These are fixed, get from state
        replay_prev_rect = state['replay_prev_rect']
        replay_next_rect = state['replay_next_rect']
        replay_end_rect = state['replay_end_rect']
        suggest_rect_base = drawn_rects.get('suggest_rect_base')
        simulate_button_rect = drawn_rects.get('simulate_button_rect')
        preview_checkbox_rect = drawn_rects.get('preview_checkbox_rect') # Get from drawn_rects
        p1_alpha_rect = drawn_rects.get('p1_alpha_rect')
        p1_rand_rect = drawn_rects.get('p1_rand_rect')
        p2_alpha_rect = drawn_rects.get('p2_alpha_rect')
        p2_rand_rect = drawn_rects.get('p2_rand_rect')
        tile_rects = drawn_rects.get('tile_rects', [])
        exchange_button_rect = drawn_rects.get('exchange_button_rect')
        cancel_button_rect = drawn_rects.get('cancel_button_rect')
        play_button_rect = drawn_rects.get('play_button_rect')
        ok_button_rect = drawn_rects.get('ok_button_rect')
        all_words_button_rect = drawn_rects.get('all_words_button_rect')
        hint_rects = drawn_rects.get('hint_rects', [])
        all_words_rects = drawn_rects.get('all_words_rects', [])
        all_words_play_rect = drawn_rects.get('all_words_play_rect')
        all_words_ok_rect = drawn_rects.get('all_words_ok_rect')
        # --- End Get Rects ---


        updated_state = {}
        running_inner = True; return_to_mode_selection = False; batch_stop_requested = False; human_played = False
        x, y = event.pos
        # --- Initialize variables for hint dialog ---
        # best_exchange_for_hint = None # Already unpacked
        # best_exchange_score_for_hint = -float('inf') # Already unpacked

        # --- Practice End Dialog Handling ---
        if showing_practice_end_dialog:
             if event.button == 1:
                 if practice_play_again_rect and practice_play_again_rect.collidepoint(x,y):
                     restart_practice_mode = True
                     showing_practice_end_dialog = False
                 elif practice_main_menu_rect and practice_main_menu_rect.collidepoint(x,y):
                     running_inner = False; return_to_mode_selection = True; # batch_stop_requested = True # Let main handle this
                 elif practice_quit_rect and practice_quit_rect.collidepoint(x,y):
                     running_inner = False; # batch_stop_requested = True # Let main handle this
             updated_state.update({'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection, 'restart_practice_mode': restart_practice_mode, 'showing_practice_end_dialog': showing_practice_end_dialog})
             return updated_state
        # Simulation Config Dialog Handling ---
        if showing_simulation_config:
            if event.button == 1:
                clicked_input = False
                for i, rect in enumerate(sim_input_rects):
                    if rect.collidepoint(x, y): simulation_config_active_input = i; clicked_input = True; break
                if not clicked_input: simulation_config_active_input = None
                if sim_cancel_rect and sim_cancel_rect.collidepoint(x, y): showing_simulation_config = False; simulation_config_active_input = None
                elif sim_simulate_rect and sim_simulate_rect.collidepoint(x, y):
                    try:
                        num_ai_cand = int(simulation_config_inputs[0]); num_opp_sim = int(simulation_config_inputs[1]); num_post_sim = int(simulation_config_inputs[2])
                        if num_ai_cand <= 0 or num_opp_sim <= 0 or num_post_sim <= 0: raise ValueError("Values must be positive.")
                        print(f"--- Running Human Turn Simulation with Params: AI Cands={num_ai_cand}, Opp Sims={num_opp_sim}, Post Sims={num_post_sim} ---")
                        showing_simulation_config = False; simulation_config_active_input = None
                        # *** Check GLOBAL status AND GADDAG_STRUCTURE directly ***
                        if gaddag_loading_status != 'loaded' or GADDAG_STRUCTURE is None:
                            show_message_dialog("Cannot simulate: AI data (GADDAG) is not loaded or available.", "Error")
                        else:
                            player_idx = turn - 1; opponent_idx = 1 - player_idx; opponent_rack_len = len(racks[opponent_idx]) if opponent_idx < len(racks) else 7
                            # --- ADD DEBUG PRINT ---
                            print("  DEBUG: About to call run_ai_simulation...")
                            # --- MODIFICATION: Pass board_tile_counts ---
                            simulation_results = run_ai_simulation(racks[player_idx], opponent_rack_len, tiles, blanks, board, bag, GADDAG_STRUCTURE.root, first_play, board_tile_counts, num_ai_candidates=num_ai_cand, num_opponent_sims=num_opp_sim, num_post_sim_candidates=num_post_sim)
                            # --- END MODIFICATION ---
                            print("  DEBUG: run_ai_simulation finished.") # DEBUG
                            if simulation_results: top_sim_move = simulation_results[0]['move']; top_sim_score = simulation_results[0]['final_score']; print(f"  Simulate Button Top Sim Result: Play '{top_sim_move.get('word','N/A')}' (Sim Score: {top_sim_score:.1f})")
                            else: print("  Simulate Button: No valid simulation results found.")
                            hint_moves = simulation_results # Store simulation results

                            # --- ADDED: Evaluate exchange option ---
                            # --- ADD DEBUG ---
                            print(f"  DEBUG: *** BEFORE Exchange Check. bag_count = {bag_count} ***") # NEW DEBUG
                            if bag_count >= 1: # Only evaluate if exchange is possible
                                print("  DEBUG: Evaluating exchange option for hint dialog...") # DEBUG 1
                                current_player_rack = racks[player_idx]
                                # Need remaining tiles dict based on current state
                                remaining_dict_for_exchange = get_remaining_tiles(current_player_rack, board_tile_counts)
                                best_exchange_for_hint, best_exchange_score_for_hint = find_best_exchange_option(current_player_rack, remaining_dict_for_exchange, bag_count)
                                # --- DEBUG 2: Print result always ---
                                if best_exchange_for_hint:
                                    print(f"  DEBUG: Best Exchange Option Found: {''.join(sorted(best_exchange_for_hint))} (Eval: {best_exchange_score_for_hint:.1f})")
                                else:
                                    print(f"  DEBUG: No beneficial exchange option found (Result: {best_exchange_score_for_hint}).") # Modified print
                                # --- END DEBUG 2 ---
                            else:
                                print("  DEBUG: Skipping exchange evaluation (bag empty or < 1 tile).") # DEBUG 3
                                best_exchange_for_hint = None
                                best_exchange_score_for_hint = -float('inf')
                            print("  DEBUG: *** AFTER Exchange Check. ***") # NEW DEBUG
                            # --- END ADDED ---

                            hinting = True; selected_hint_index = 0 if hint_moves else None
                            print(f"  DEBUG: Setting hinting = {hinting}") # NEW DEBUG
                    except ValueError as e: show_message_dialog(f"Invalid input: {e}\nPlease enter positive numbers.", "Input Error")
            # --- MODIFICATION: Remove premature return ---
            # updated_state.update({...}) # Update happens later now
            # return updated_state # REMOVED
        # --- Specify Rack Dialog Handling ---
        if specifying_rack:
            if confirming_override:
                if event.button == 1:
                    if go_back_rect_ov and go_back_rect_ov.collidepoint(x, y): confirming_override = False; specify_rack_proposed_racks = [[], []]
                    elif override_rect_ov and override_rect_ov.collidepoint(x, y):
                        print("Overriding bag constraints and setting racks."); racks[0] = specify_rack_proposed_racks[0][:]; racks[1] = specify_rack_proposed_racks[1][:]
                        if not is_ai[0]: racks[0].sort()
                        if not is_ai[1]: racks[1].sort()
                        # --- REMOVED automatic move regeneration ---
                        all_moves = [] # Explicitly clear moves after setting rack
                        specifying_rack = False; confirming_override = False; specify_rack_inputs = ["", ""]; specify_rack_active_input = None; specify_rack_original_racks = [[], []]; specify_rack_proposed_racks = [[], []]; dropdown_open = False
                updated_state.update({'racks': racks, 'all_moves': all_moves, 'specifying_rack': specifying_rack, 'confirming_override': confirming_override, 'specify_rack_inputs': specify_rack_inputs, 'specify_rack_active_input': specify_rack_active_input, 'specify_rack_original_racks': specify_rack_original_racks, 'specify_rack_proposed_racks': specify_rack_proposed_racks, 'dropdown_open': dropdown_open})
                return updated_state
            if event.button == 1:
                if p1_input_rect_sr and p1_input_rect_sr.collidepoint(x, y): specify_rack_active_input = 0
                elif p2_input_rect_sr and p2_input_rect_sr.collidepoint(x, y): specify_rack_active_input = 1
                elif p1_reset_rect_sr and p1_reset_rect_sr.collidepoint(x, y): specify_rack_inputs[0] = "".join(['?' if t == ' ' else t for t in specify_rack_original_racks[0]])
                elif p2_reset_rect_sr and p2_reset_rect_sr.collidepoint(x, y): specify_rack_inputs[1] = "".join(['?' if t == ' ' else t for t in specify_rack_original_racks[1]])
                elif cancel_rect_sr and cancel_rect_sr.collidepoint(x, y): specifying_rack = False; specify_rack_inputs = ["", ""]; specify_rack_active_input = None; specify_rack_original_racks = [[], []]
                elif confirm_rect_sr and confirm_rect_sr.collidepoint(x, y):
                    valid_input = True; proposed_racks_temp = [[], []]; error_message = None
                    for i in range(2):
                        input_str = specify_rack_inputs[i].upper()
                        if not (0 <= len(input_str) <= 7): error_message = f"Player {i+1} rack must have 0 to 7 tiles."; valid_input = False; break # Allow 0 tiles
                        current_proposed_rack = []
                        for char in input_str:
                            if 'A' <= char <= 'Z': current_proposed_rack.append(char)
                            elif char == '?' or char == ' ': current_proposed_rack.append(' ')
                            else: error_message = f"Invalid character '{char}' in Player {i+1} rack."; valid_input = False; break
                        if not valid_input: break
                        proposed_racks_temp[i] = current_proposed_rack
                    if not valid_input:
                        if error_message: show_message_dialog(error_message, "Input Error")
                    else:
                        bag_counts = Counter(bag); needs_override = False; combined_original_counts = Counter(specify_rack_original_racks[0]) + Counter(specify_rack_original_racks[1]); combined_proposed_counts = Counter(proposed_racks_temp[0]) + Counter(proposed_racks_temp[1]); net_change = combined_proposed_counts - combined_original_counts
                        if any(bag_counts[tile] < count for tile, count in net_change.items()): needs_override = True
                        if needs_override: print("Specified tiles require override."); specify_rack_proposed_racks = [r[:] for r in proposed_racks_temp]; confirming_override = True
                        else:
                            print("Specified racks are valid or don't require bag tiles. Setting racks."); racks[0] = proposed_racks_temp[0][:]; racks[1] = proposed_racks_temp[1][:]
                            if not is_ai[0]: racks[0].sort()
                            if not is_ai[1]: racks[1].sort()
                            # --- REMOVED automatic move regeneration ---
                            all_moves = [] # Explicitly clear moves after setting rack
                            specifying_rack = False; specify_rack_inputs = ["", ""]; specify_rack_active_input = None; specify_rack_original_racks = [[], []]; dropdown_open = False
                else: specify_rack_active_input = None
            updated_state.update({'racks': racks, 'all_moves': all_moves, 'specifying_rack': specifying_rack, 'confirming_override': confirming_override, 'specify_rack_inputs': specify_rack_inputs, 'specify_rack_active_input': specify_rack_active_input, 'specify_rack_original_racks': specify_rack_original_racks, 'specify_rack_proposed_racks': specify_rack_proposed_racks, 'dropdown_open': dropdown_open})
            return updated_state
        # --- Options Menu Click ---
        if options_rect_base and options_rect_base.collidepoint(x, y):
            dropdown_open = not dropdown_open; updated_state['dropdown_open'] = dropdown_open; return updated_state
        # --- Dropdown Item Click ---
        elif dropdown_open:
            clicked_dropdown_item = False; # ... (determine current_options_list) ...
            if is_batch_running: current_options_list = ["Stop Batch", "Quit"]
            elif replay_mode or game_over_state: current_options_list = ["Main", "Quit"]
            elif practice_mode == "eight_letter": current_options_list = ["Give Up", "Main", "Quit"]
            else: current_options_list = ["Pass", "Exchange", "Specify Rack", "Main", "Quit"]
            drawn_options = current_options_list
            for i, rect in enumerate(dropdown_rects_base):
                if rect and rect.collidepoint(x, y):
                    if i < len(drawn_options):
                        selected_option = drawn_options[i]; clicked_dropdown_item = True; dropdown_open = False
                        # ... (handle selected_option actions as before) ...
                        if selected_option == "Stop Batch": print("--- Batch Run Aborted by User ---"); running_inner = False; return_to_mode_selection = True # Set flags
                        elif selected_option == "Pass": move_rack = racks[turn-1][:]; consecutive_zero_point_turns += 1; pass_count += 1; exchange_count = 0; print(f"Player {turn} passed"); human_played = True; paused_for_power_tile = False; paused_for_bingo_practice = False; move_history.append({'player': turn, 'move_type': 'pass', 'rack': move_rack, 'score': 0, 'word': '', 'coord': '', 'blanks': set(), 'positions': [], 'drawn': [], 'is_bingo': False, 'word_with_blanks': '', 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': 0.0}); current_replay_turn = len(move_history); turn = 3 - turn; last_played_highlight_coords = set()
                        elif selected_option == "Exchange":
                            if bag_count >= 7: exchanging = True; selected_tiles.clear() # Use bag_count variable
                            else: show_message_dialog("Cannot exchange, less than 7 tiles in bag.", "Exchange Error")
                        elif selected_option == "Specify Rack":
                            is_human_turn = not is_ai[turn-1]; allowed_mode = game_mode in [MODE_HVH, MODE_HVA]
                            if is_human_turn and allowed_mode:
                                print("Specify Rack selected."); specifying_rack = True; specify_rack_original_racks = [racks[0][:], racks[1][:]]; specify_rack_inputs[0] = "".join(['?' if t == ' ' else t for t in racks[0]]); specify_rack_inputs[1] = "".join(['?' if t == ' ' else t for t in racks[1]]); specify_rack_active_input = None; confirming_override = False
                                typing = False; word_positions = []; selected_square = None; current_r = None; current_c = None # Reset cursor state
                                if original_tiles and original_rack:
                                    for r_wp, c_wp, _ in word_positions: tiles[r_wp][c_wp] = original_tiles[r_wp][c_wp]
                                    racks[turn-1] = original_rack[:];
                                    if not is_ai[turn-1]: racks[turn-1].sort()
                                    blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in word_positions if (r_wp, c_wp) in blanks); blanks.difference_update(blanks_to_remove); original_tiles = None; original_rack = None
                            else: show_message_dialog("Specify Rack only available on Human turn in HvH/HvA modes.", "Action Unavailable")
                        elif selected_option == "Give Up":
                            if practice_mode == "eight_letter":
                                practice_end_message = f"Best: {practice_best_move['word_with_blanks']} ({practice_best_move['score']} pts)" if practice_best_move else "No best move found."; practice_solved = True; showing_practice_end_dialog = True
                        elif selected_option == "Main": running_inner = False; return_to_mode_selection = True # Set flags
                        elif selected_option == "Quit":
                            if confirm_quit(): running_inner = False; return_to_mode_selection = False # Set flags
                        break
            if clicked_dropdown_item:
                 updated_state.update({'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection, 'dropdown_open': dropdown_open, 'exchanging': exchanging, 'selected_tiles': selected_tiles, 'specifying_rack': specifying_rack, 'specify_rack_original_racks': specify_rack_original_racks, 'specify_rack_inputs': specify_rack_inputs, 'specify_rack_active_input': specify_rack_active_input, 'confirming_override': confirming_override, 'typing': typing, 'word_positions': word_positions, 'selected_square': selected_square, 'original_tiles': original_tiles, 'original_rack': original_rack, 'blanks': blanks, 'practice_end_message': practice_end_message, 'practice_solved': practice_solved, 'showing_practice_end_dialog': showing_practice_end_dialog, 'consecutive_zero_point_turns': consecutive_zero_point_turns, 'pass_count': pass_count, 'exchange_count': exchange_count, 'human_played': human_played, 'paused_for_power_tile': paused_for_power_tile, 'paused_for_bingo_practice': paused_for_bingo_practice, 'move_history': move_history, 'current_replay_turn': current_replay_turn, 'turn': turn, 'last_played_highlight_coords': last_played_highlight_coords, 'racks': racks, 'bag': bag, 'current_r': current_r, 'current_c': current_c}) # Pack cursor state
                 return updated_state
            else: dropdown_open = False; updated_state['dropdown_open'] = dropdown_open

        # --- Process other MOUSEBUTTONDOWN only if NOT in batch mode ---
        if not is_batch_running:
            current_time = pygame.time.get_ticks()
            if event.button == 1: # Left Click
                # --- Game Over Event Handling ---
                if game_over_state:
                    # ... (game over button handling - unchanged, uses rects from drawn_rects) ...
                    if save_rect and save_rect.collidepoint(x, y):
                        if final_scores and player_names and move_history and initial_racks:
                            gcg_content = save_game_to_gcg(player_names, move_history, initial_racks, final_scores); now = datetime.datetime.now(); date_str = now.strftime("%d%b%y").upper(); time_str = now.strftime("%H%M"); seq_num = 1; max_existing_num = 0
                            try:
                                for filename in os.listdir('.'):
                                    if filename.startswith(f"{date_str}-") and filename.endswith(".gcg") and "-GAME-" in filename:
                                        parts = filename[:-4].split('-');
                                        if len(parts) >= 4 and parts[2] == "GAME":
                                            if parts[-1].isdigit(): num = int(parts[-1]); max_existing_num = max(max_existing_num, num)
                                seq_num = max_existing_num + 1
                            except OSError as e: print(f"Error listing directory for save sequence number: {e}")
                            save_filename = f"{date_str}-{time_str}-GAME-{seq_num}.gcg"
                            try:
                                with open(save_filename, "w") as f: f.write(gcg_content); print(f"Game saved to {save_filename}"); show_message_dialog(f"Game saved to:\n{save_filename}", "Game Saved")
                            except IOError as e: print(f"Error saving game to {save_filename}: {e}"); show_message_dialog(f"Error saving game: {e}", "Save Error")
                        else: print("Error: Missing data required for saving."); show_message_dialog("Could not save game: Missing data.", "Save Error")
                    elif quit_rect and quit_rect.collidepoint(x, y): running_inner = False; return_to_mode_selection = False # Set flags
                    elif replay_rect and replay_rect.collidepoint(x, y):
                        if move_history: print("Entering Replay Mode..."); replay_mode = True; current_replay_turn = 0; game_over_state = False; showing_stats = False; last_played_highlight_coords = set()
                        else: print("Cannot enter replay: No move history found.")
                    elif play_again_rect and play_again_rect.collidepoint(x, y): running_inner = False; return_to_mode_selection = True # Set flags
                    elif stats_rect and stats_rect.collidepoint(x, y): showing_stats = True; stats_dialog_x = (WINDOW_WIDTH - 480) // 2; stats_dialog_y = (WINDOW_HEIGHT - 600) // 2; stats_scroll_offset = 0; stats_dialog_dragging = False
                    elif showing_stats and stats_ok_button_rect and stats_ok_button_rect.collidepoint(x, y): showing_stats = False
                    elif showing_stats:
                        title_bar_height = 40; stats_title_rect = pygame.Rect(stats_dialog_x, stats_dialog_y, 480, title_bar_height)
                        if stats_title_rect.collidepoint(x, y): stats_dialog_dragging = True; stats_dialog_drag_offset = (x - stats_dialog_x, y - stats_dialog_y)
                    elif not showing_stats:
                        dialog_rect = pygame.Rect(dialog_x, dialog_y, DIALOG_WIDTH, DIALOG_HEIGHT)
                        if dialog_rect.collidepoint(x, y): dragging = True; drag_offset = (x - dialog_x, y - dialog_y)
                    updated_state.update({'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection, 'replay_mode': replay_mode, 'current_replay_turn': current_replay_turn, 'game_over_state': game_over_state, 'showing_stats': showing_stats, 'stats_dialog_x': stats_dialog_x, 'stats_dialog_y': stats_dialog_y, 'stats_dialog_dragging': stats_dialog_dragging, 'dragging': dragging, 'drag_offset': drag_offset, 'last_played_highlight_coords': last_played_highlight_coords})
                    return updated_state

                # --- Active Game / Replay Event Handling ---
                if not specifying_rack and not showing_simulation_config:
                    if replay_mode: # Replay buttons
                        if replay_start_rect.collidepoint(x, y): current_replay_turn = 0; last_played_highlight_coords = set()
                        elif replay_prev_rect.collidepoint(x, y) and current_replay_turn > 0: current_replay_turn -= 1; last_played_highlight_coords = set()
                        elif replay_next_rect.collidepoint(x, y) and current_replay_turn < len(move_history): current_replay_turn += 1; last_played_highlight_coords = set()
                        elif replay_end_rect.collidepoint(x, y): current_replay_turn = len(move_history); last_played_highlight_coords = set()
                    elif not (exchanging or hinting or showing_all_words): # Active game clicks
                        is_human_turn_or_paused_practice = not is_ai[turn-1] or paused_for_power_tile or paused_for_bingo_practice
                        # --- Suggest Button Click ---
                        if suggest_rect_base and suggest_rect_base.collidepoint(x, y) and is_human_turn_or_paused_practice:
                            # --- ADD DEBUG PRINT HERE ---
                            print(f"DEBUG Suggest Handler: practice_mode = '{practice_mode}' (Type: {type(practice_mode)})")
                            # --- END DEBUG PRINT ---
                            # --- MODIFICATION START: Correct 8-Letter Logic ---
                            if practice_mode == "eight_letter":
                                print(f"DEBUG: Suggest clicked (8-Letter). Using practice_target_moves (length: {len(practice_target_moves)}).")
                                moves_to_hint = practice_target_moves if practice_target_moves else []
                                all_moves = practice_target_moves # Ensure all_moves state is also correct
                            # --- END MODIFICATION ---
                            else: # Standard game or other practice modes
                                # *** Check GLOBAL status AND GADDAG_STRUCTURE directly ***
                                if gaddag_loading_status != 'loaded' or GADDAG_STRUCTURE is None:
                                    show_message_dialog("Cannot suggest moves: AI data is not loaded or available.", "Loading")
                                    moves_to_hint = [] # Ensure it's empty if GADDAG not ready
                                    all_moves = []
                                else:
                                    current_player_rack = racks[turn-1]
                                    print(f"DEBUG: Suggest clicked (Standard/Other Practice). Regenerating moves for Player {turn}, Rack: {''.join(sorted(current_player_rack))}")
                                    tiles_copy = [row[:] for row in tiles]; blanks_copy = blanks.copy(); board_copy = board
                                    # *** Generate into temporary variable ***
                                    all_moves_generated_for_hint = generate_all_moves_gaddag(current_player_rack, tiles_copy, board_copy, blanks_copy, GADDAG_STRUCTURE.root)
                                    if all_moves_generated_for_hint is None: all_moves_generated_for_hint = []; print("DEBUG: generate_all_moves_gaddag returned None for Suggest.")
                                    else: print(f"DEBUG: generate_all_moves_gaddag returned {len(all_moves_generated_for_hint)} moves for Suggest.")

                                    # Filter for other practice modes if needed
                                    if practice_mode == "power_tiles" and paused_for_power_tile and current_power_tile: power_moves_hint = [m for m in all_moves_generated_for_hint if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)]; moves_to_hint = sorted(power_moves_hint, key=lambda m: m['score'], reverse=True)
                                    elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice: bingo_moves_hint = [m for m in all_moves_generated_for_hint if m.get('is_bingo', False)]; moves_to_hint = sorted(bingo_moves_hint, key=lambda m: m['score'], reverse=True)
                                    else: moves_to_hint = all_moves_generated_for_hint # Use generated moves for standard game
                                    # *** Store the generated moves back into the main state's all_moves ***
                                    all_moves = all_moves_generated_for_hint # Update the state variable

                            # --- Common logic for setting hint state ---
                            hint_moves = moves_to_hint[:5]; hinting = True; selected_hint_index = 0 if hint_moves else None
                            updated_state['all_moves'] = all_moves # Ensure it's packed

                        # --- Simulate Button Click ---
                        elif simulate_button_rect and simulate_button_rect.collidepoint(x, y) and is_human_turn_or_paused_practice:
                            # *** Check GLOBAL status AND GADDAG_STRUCTURE directly ***
                            if gaddag_loading_status != 'loaded' or GADDAG_STRUCTURE is None:
                                show_message_dialog("Cannot simulate: AI data is not loaded or available.", "Loading")
                            else:
                                print("Simulate button clicked."); showing_simulation_config = True; simulation_config_inputs = [str(DEFAULT_AI_CANDIDATES), str(DEFAULT_OPPONENT_SIMULATIONS), str(DEFAULT_POST_SIM_CANDIDATES)]; simulation_config_active_input = None
                                typing = False; word_positions = []; selected_square = None; current_r = None; current_c = None # Reset cursor
                                if original_tiles and original_rack:
                                    for r_wp, c_wp, _ in word_positions: tiles[r_wp][c_wp] = original_tiles[r_wp][c_wp]
                                    racks[turn-1] = original_rack[:];
                                    if not is_ai[turn-1]: racks[turn-1].sort()
                                    blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in word_positions if (r_wp, c_wp) in blanks); blanks.difference_update(blanks_to_remove); original_tiles = None; original_rack = None
                        # --- Preview Checkbox Click ---
                        elif preview_checkbox_rect and preview_checkbox_rect.collidepoint(x, y): # Use the rect from drawn_rects
                            preview_score_enabled = not preview_score_enabled
                        # --- Rack Button Clicks ---
                        current_player_idx = turn - 1
                        if 0 <= current_player_idx < len(is_ai) and is_human_turn_or_paused_practice:
                             if turn == 1:
                                  if p1_alpha_rect and p1_alpha_rect.collidepoint(x, y): racks[0].sort()
                                  elif p1_rand_rect and p1_rand_rect.collidepoint(x, y): random.shuffle(racks[0])
                             elif turn == 2 and practice_mode != "eight_letter":
                                  if p2_alpha_rect and p2_alpha_rect.collidepoint(x, y): racks[1].sort()
                                  elif p2_rand_rect and p2_rand_rect.collidepoint(x, y): random.shuffle(racks[1])
                        # --- Drag Start ---
                        rack_y_drag = BOARD_SIZE + 80 if turn == 1 else BOARD_SIZE + 150; rack_width_calc = 7 * (TILE_WIDTH + TILE_GAP) - TILE_GAP; replay_area_end_x = 10 + 4 * (REPLAY_BUTTON_WIDTH + REPLAY_BUTTON_GAP); min_rack_start_x = replay_area_end_x + BUTTON_GAP + 20; rack_start_x_calc = max(min_rack_start_x, (BOARD_SIZE - rack_width_calc) // 2)
                        if 0 <= current_player_idx < len(racks) and 0 <= current_player_idx < len(is_ai):
                            rack_len = len(racks[current_player_idx]); tile_idx = get_tile_under_mouse(x, y, rack_start_x_calc, rack_y_drag, rack_len)
                            if tile_idx is not None and not dragged_tile and is_human_turn_or_paused_practice:
                                dragged_tile = (turn, tile_idx); drag_pos = (x, y);
                                tile_abs_x = rack_start_x_calc + tile_idx * (TILE_WIDTH + TILE_GAP); tile_center_x = tile_abs_x + TILE_WIDTH // 2; tile_center_y = rack_y_drag + TILE_HEIGHT // 2; drag_offset = (x - tile_center_x, y - tile_center_y)
                        # --- Board Click ---
                        if not dragged_tile and is_human_turn_or_paused_practice:
                            col = (x - 40) // SQUARE_SIZE; row = (y - 40) // SQUARE_SIZE
                            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and not tiles[row][col]:
                                is_double_click = (last_left_click_pos == (row, col) and current_time - last_left_click_time < DOUBLE_CLICK_TIME)
                                if is_double_click:
                                     selected_square = None; typing = False; current_r = None; current_c = None # Reset cursor
                                     if word_positions and original_tiles and original_rack:
                                          for r_wp, c_wp, _ in word_positions: tiles[r_wp][c_wp] = original_tiles[r_wp][c_wp]
                                          racks[turn-1] = original_rack[:];
                                          if not is_ai[turn-1]: racks[turn-1].sort()
                                          blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in word_positions if (r_wp, c_wp) in blanks); blanks.difference_update(blanks_to_remove); word_positions = []; original_tiles = None; original_rack = None
                                elif selected_square is None or selected_square[:2] != (row, col): selected_square = (row, col, "right"); typing = False; word_positions = []; current_r = None; current_c = None # Reset cursor
                                elif selected_square[2] == "right": selected_square = (row, col, "down")
                                elif selected_square[2] == "down": selected_square = None; current_r = None; current_c = None # Reset cursor
                                last_left_click_pos = (row, col); last_left_click_time = current_time
                            else: selected_square = None; current_r = None; current_c = None # Reset cursor

                    # --- Handle clicks within dialogs (Exchange, Hint, All Words) ---
                    elif exchanging:
                        clicked_tile = False
                        for i, rect in enumerate(tile_rects):
                            if rect.collidepoint(x, y): selected_tiles.add(i) if i not in selected_tiles else selected_tiles.remove(i); clicked_tile = True; break
                        if not clicked_tile:
                            if exchange_button_rect and exchange_button_rect.collidepoint(x, y):
                                if selected_tiles:
                                    tiles_to_exchange = [racks[turn-1][i] for i in selected_tiles]; print(f"Player {turn} exchanging {len(tiles_to_exchange)} tiles: {''.join(sorted(tiles_to_exchange))}"); move_rack = racks[turn-1][:]
                                    new_rack = [tile for i, tile in enumerate(racks[turn-1]) if i not in selected_tiles]; num_to_draw = len(tiles_to_exchange); drawn_tiles = [bag.pop() for _ in range(num_to_draw) if bag]; new_rack.extend(drawn_tiles); racks[turn-1] = new_rack
                                    if not is_ai[turn-1]: racks[turn-1].sort(); bag.extend(tiles_to_exchange); random.shuffle(bag)
                                    luck_factor = 0.0
                                    if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                    move_history.append({'player': turn, 'move_type': 'exchange', 'rack': move_rack, 'exchanged_tiles': tiles_to_exchange, 'drawn': drawn_tiles, 'score': 0, 'word': '', 'coord': '', 'blanks': set(), 'positions': [], 'is_bingo': False, 'word_with_blanks': '', 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history)
                                    exchanging = False; selected_tiles.clear(); consecutive_zero_point_turns += 1; exchange_count += 1; pass_count = 0; human_played = True; paused_for_power_tile = False; paused_for_bingo_practice = False; turn = 3 - turn; last_played_highlight_coords = set()
                                else: show_message_dialog("No tiles selected for exchange.", "Exchange Error")
                            elif cancel_button_rect and cancel_button_rect.collidepoint(x, y): exchanging = False; selected_tiles.clear()
                    elif hinting:
                        clicked_in_dialog = False
                        if play_button_rect and play_button_rect.collidepoint(x, y) and selected_hint_index is not None and selected_hint_index < len(hint_moves):
                            clicked_in_dialog = True; selected_move = None; is_simulation_result = bool(hint_moves and isinstance(hint_moves[0], dict) and 'final_score' in hint_moves[0])
                            if is_simulation_result: selected_item = hint_moves[selected_hint_index]; selected_move = selected_item.get('move')
                            else: selected_move = hint_moves[selected_hint_index]

                            if selected_move and isinstance(selected_move, dict):
                                player_who_played = turn # Store player number BEFORE turn changes
                                move_rack = racks[player_who_played-1][:]; valid_for_practice = True
                                # Check for "Only Fives" practice
                                if practice_mode == "only_fives":
                                    if not does_move_form_five_letter_word(selected_move, tiles, blanks): show_message_dialog("At least one 5-letter word must be formed.", "Invalid Play"); valid_for_practice = False
                                # --- 8-LETTER BINGO HINT PLAY LOGIC ---
                                elif practice_mode == "eight_letter":
                                    if practice_best_move: # Ensure best move is known
                                        selected_score = selected_move.get('score', -1); max_score_8l = practice_best_move.get('score', 0)
                                        if selected_score >= max_score_8l and max_score_8l > 0: # Correct play
                                            print(f"8-Letter Hint Play CORRECT! Played: '{selected_move.get('word_with_blanks')}' ({selected_score} pts)");
                                            # --- MODIFICATION: Pass board_tile_counts ---
                                            _next_turn, _drawn, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                            human_played = True; hinting = False; practice_solved = True; showing_practice_end_dialog = True; practice_end_message = f"Correct! You found the highest scoring bingo:\n{selected_move.get('word_with_blanks','')} ({selected_score} pts)"; last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', []))
                                        else: print(f"8-Letter Hint Play INCORRECT. Score: {selected_score}, Target: {max_score_8l}"); show_message_dialog(f"Try again. The highest score is {max_score_8l}.", "8-Letter Bingo"); hinting = False # Close hint dialog
                                    else: print("Error: Cannot validate 8-letter hint play, best move unknown."); show_message_dialog("Error: Best move data missing for validation.", "Internal Error"); hinting = False
                                    valid_for_practice = False # Prevent falling through
                                # --- END 8-LETTER BINGO HINT PLAY LOGIC ---
                                # --- Power Tile Practice Hint Play Logic ---
                                elif paused_for_power_tile:
                                    power_moves_filtered = [ m for m in all_moves if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks) ]; max_power_score_filtered = max(m['score'] for m in power_moves_filtered) if power_moves_filtered else 0
                                    if selected_move.get('score', -1) >= max_power_score_filtered:
                                        # --- MODIFICATION: Pass board_tile_counts ---
                                        next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                        human_played = True; hinting = False; paused_for_power_tile = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0; luck_factor = 0.0;
                                        if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                        move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                                        # Pack state and return immediately
                                        updated_state.update({'turn': turn, 'human_played': human_played, 'hinting': hinting, 'paused_for_power_tile': paused_for_power_tile, 'consecutive_zero_point_turns': consecutive_zero_point_turns, 'pass_count': pass_count, 'exchange_count': exchange_count, 'move_history': move_history, 'current_replay_turn': current_replay_turn, 'last_played_highlight_coords': last_played_highlight_coords, 'racks': racks, 'bag': bag, 'tiles': tiles, 'blanks': blanks, 'scores': scores, 'board_tile_counts': board_tile_counts}) # Pack counter
                                        return updated_state # Return early
                                    else: show_message_dialog(f"This is not the highest scoring move with {current_power_tile} matching the selected lengths!", "Incorrect Move"); valid_for_practice = False # Prevent finalization
                                # --- Bingo Bango Bongo Hint Play Logic ---
                                elif paused_for_bingo_practice:
                                    bingo_moves = [m for m in all_moves if m.get('is_bingo', False)]; max_bingo_score = max(m['score'] for m in bingo_moves) if bingo_moves else 0
                                    if selected_move.get('is_bingo', False) and selected_move.get('score', -1) >= max_bingo_score:
                                        # --- MODIFICATION: Pass board_tile_counts ---
                                        next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                        human_played = True; hinting = False; paused_for_bingo_practice = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0; luck_factor = 0.0;
                                        if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                        # *** USE player_who_played FOR HISTORY ***
                                        move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                                        # Pack state and return immediately
                                        updated_state.update({'turn': turn, 'human_played': human_played, 'hinting': hinting, 'paused_for_bingo_practice': paused_for_bingo_practice, 'consecutive_zero_point_turns': consecutive_zero_point_turns, 'pass_count': pass_count, 'exchange_count': exchange_count, 'move_history': move_history, 'current_replay_turn': current_replay_turn, 'last_played_highlight_coords': last_played_highlight_coords, 'racks': racks, 'bag': bag, 'tiles': tiles, 'blanks': blanks, 'scores': scores, 'board_tile_counts': board_tile_counts}) # Pack counter
                                        return updated_state # Return early
                                    else: show_message_dialog(f"This is not the highest scoring bingo! Max score is {max_bingo_score}.", "Incorrect Move"); valid_for_practice = False # Prevent finalization
                                # --- Standard Play or Valid "Only Fives" Finalization ---
                                if valid_for_practice and practice_mode not in ["eight_letter"] and not paused_for_power_tile and not paused_for_bingo_practice:
                                    # --- MODIFICATION: Pass board_tile_counts ---
                                    next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                    human_played = True; hinting = False; paused_for_power_tile = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0
                                    luck_factor = 0.0;
                                    if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                    # *** USE player_who_played FOR HISTORY ***
                                    move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                            elif is_simulation_result and not selected_move: show_message_dialog("Error retrieving move data from selected simulation result.", "Internal Error")
                        elif ok_button_rect and ok_button_rect.collidepoint(x, y): clicked_in_dialog = True; hinting = False
                        elif all_words_button_rect and all_words_button_rect.collidepoint(x, y):
                            clicked_in_dialog = True; hinting = False; showing_all_words = True;
                            # *** Use the updated all_moves from state ***
                            current_all_moves = all_moves # Use the version potentially updated by Suggest
                            if practice_mode == "eight_letter": moves_for_all = practice_target_moves
                            elif practice_mode == "power_tiles" and paused_for_power_tile: moves_for_all = sorted([m for m in current_all_moves if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)], key=lambda m: m['score'], reverse=True)
                            elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice: moves_for_all = sorted([m for m in current_all_moves if m.get('is_bingo', False)], key=lambda m: m['score'], reverse=True)
                            else: moves_for_all = current_all_moves # Use potentially updated all_moves
                            selected_hint_index = 0 if moves_for_all else None; all_words_scroll_offset = 0
                        elif hint_rects:
                            for i, rect in enumerate(hint_rects):
                                if rect.collidepoint(x, y) and i < len(hint_moves): clicked_in_dialog = True; selected_hint_index = i; break
                        dialog_width_hint, dialog_height_hint = 400, 280 # Adjusted height
                        dialog_rect_hint = pygame.Rect((WINDOW_WIDTH - dialog_width_hint) // 2, (WINDOW_HEIGHT - dialog_height_hint) // 2, dialog_width_hint, dialog_height_hint)
                        if dialog_rect_hint.collidepoint(x,y) and not clicked_in_dialog: pass
                    elif showing_all_words:
                        clicked_in_dialog = False;
                        # *** Use the updated all_moves from state ***
                        current_all_moves = all_moves
                        if practice_mode == "eight_letter": moves_for_all = practice_target_moves
                        elif practice_mode == "power_tiles" and paused_for_power_tile: moves_for_all = sorted([m for m in current_all_moves if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)], key=lambda m: m['score'], reverse=True)
                        elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice: moves_for_all = sorted([m for m in current_all_moves if m.get('is_bingo', False)], key=lambda m: m['score'], reverse=True)
                        else: moves_for_all = current_all_moves # Use potentially updated all_moves
                        if all_words_play_rect and all_words_play_rect.collidepoint(x, y) and selected_hint_index is not None and selected_hint_index < len(moves_for_all):
                            clicked_in_dialog = True; selected_move = moves_for_all[selected_hint_index];
                            player_who_played = turn # Store player number BEFORE turn changes
                            move_rack = racks[player_who_played-1][:];
                            valid_for_practice = True
                            # --- ADD "Only Fives" Check ---
                            if practice_mode == "only_fives":
                                if not does_move_form_five_letter_word(selected_move, tiles, blanks): show_message_dialog("At least one 5-letter word must be formed.", "Invalid Play"); valid_for_practice = False
                            # --- ADD 8-Letter Check (Similar to hint dialog) ---
                            elif practice_mode == "eight_letter":
                                 if practice_best_move:
                                     selected_score = selected_move.get('score', -1); max_score_8l = practice_best_move.get('score', 0)
                                     if selected_score >= max_score_8l and max_score_8l > 0: # Correct
                                         print(f"8-Letter All Words Play CORRECT! Played: '{selected_move.get('word_with_blanks')}' ({selected_score} pts)");
                                         # --- MODIFICATION: Pass board_tile_counts ---
                                         _next_turn, _drawn, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                         human_played = True; showing_all_words = False; practice_solved = True; showing_practice_end_dialog = True; practice_end_message = f"Correct! You found the highest scoring bingo:\n{selected_move.get('word_with_blanks','')} ({selected_score} pts)"; last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', []))
                                     else: print(f"8-Letter All Words Play INCORRECT. Score: {selected_score}, Target: {max_score_8l}"); show_message_dialog(f"Try again. The highest score is {max_score_8l}.", "8-Letter Bingo"); showing_all_words = False # Close dialog
                                 else: print("Error: Cannot validate 8-letter all words play, best move unknown."); show_message_dialog("Error: Best move data missing for validation.", "Internal Error"); showing_all_words = False
                                 valid_for_practice = False # Prevent standard finalization
                            # --- END 8-Letter Check ---
                            if valid_for_practice: # Proceed only if valid for current practice mode and not 8-letter
                                if paused_for_bingo_practice:
                                    bingo_moves = [m for m in all_moves if m.get('is_bingo', False)]; max_bingo_score = max(m['score'] for m in bingo_moves) if bingo_moves else 0
                                    if selected_move.get('is_bingo', False) and selected_move.get('score', -1) >= max_bingo_score:
                                        # --- MODIFICATION: Pass board_tile_counts ---
                                        next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                        human_played = True; showing_all_words = False; paused_for_bingo_practice = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0; luck_factor = 0.0;
                                        if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                        # *** USE player_who_played FOR HISTORY ***
                                        move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                                        # Pack state and return immediately
                                        updated_state.update({'turn': turn, 'human_played': human_played, 'showing_all_words': showing_all_words, 'paused_for_bingo_practice': paused_for_bingo_practice, 'consecutive_zero_point_turns': consecutive_zero_point_turns, 'pass_count': pass_count, 'exchange_count': exchange_count, 'move_history': move_history, 'current_replay_turn': current_replay_turn, 'last_played_highlight_coords': last_played_highlight_coords, 'racks': racks, 'bag': bag, 'tiles': tiles, 'blanks': blanks, 'scores': scores, 'board_tile_counts': board_tile_counts}) # Pack counter
                                        return updated_state # Return early
                                    else: show_message_dialog(f"This is not the highest scoring bingo! Max score is {max_bingo_score}.", "Incorrect Move")
                                else: # Standard play or valid "Only Fives" play
                                    # --- MODIFICATION: Pass board_tile_counts ---
                                    next_turn, drawn_tiles, newly_placed, board_tile_counts = play_hint_move(selected_move, tiles, racks, blanks, scores, player_who_played, bag, board, board_tile_counts);
                                    human_played = True; showing_all_words = False; paused_for_power_tile = False; consecutive_zero_point_turns = 0; pass_count = 0; exchange_count = 0
                                    luck_factor = 0.0;
                                    if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = current_turn_pool_quality_score * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                    # *** USE player_who_played FOR HISTORY ***
                                    move_history.append({'player': player_who_played, 'move_type': 'place', 'rack': move_rack, 'positions': selected_move.get('positions',[]), 'blanks': selected_move.get('blanks',set()), 'score': selected_move.get('score',0), 'word': selected_move.get('word','N/A'), 'drawn': drawn_tiles, 'coord': get_coord(selected_move.get('start',(0,0)), selected_move.get('direction','right')), 'word_with_blanks': selected_move.get('word_with_blanks',''), 'is_bingo': selected_move.get('is_bingo',False), 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': luck_factor}); current_replay_turn = len(move_history); last_played_highlight_coords = set((pos[0], pos[1]) for pos in selected_move.get('positions', [])); turn = next_turn
                        elif all_words_ok_rect and all_words_ok_rect.collidepoint(x, y): clicked_in_dialog = True; showing_all_words = False
                        elif all_words_rects:
                            for rect, idx in all_words_rects:
                                if rect.collidepoint(x, y): clicked_in_dialog = True; selected_hint_index = idx; break
                        dialog_rect_all = pygame.Rect((WINDOW_WIDTH - ALL_WORDS_DIALOG_WIDTH) // 2, (WINDOW_HEIGHT - ALL_WORDS_DIALOG_HEIGHT) // 2, ALL_WORDS_DIALOG_WIDTH, ALL_WORDS_DIALOG_HEIGHT)
                        if dialog_rect_all.collidepoint(x,y) and not clicked_in_dialog: pass

            # --- Right Click Handling ---
            elif event.button == 3:
                selected_square = None; current_r = None; current_c = None # Reset cursor
                if typing:
                    if original_tiles and original_rack:
                        for r_wp, c_wp, _ in word_positions: tiles[r_wp][c_wp] = original_tiles[r_wp][c_wp]
                        racks[turn-1] = original_rack[:];
                        if not is_ai[turn-1]: racks[turn-1].sort()
                        blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in word_positions if (r_wp, c_wp) in blanks); blanks.difference_update(blanks_to_remove)
                    typing = False; typing_start = None; typing_direction = None; word_positions = []; original_tiles = None; original_rack = None;



        # Pack updated state variables into the return dictionary
        updated_state.update({
            'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection,
            'dropdown_open': dropdown_open,
            'hinting': hinting, 'showing_all_words': showing_all_words, 'exchanging': exchanging,
            'typing': typing, 'selected_square': selected_square, 'dragged_tile': dragged_tile,
            'drag_pos': drag_pos, 'drag_offset': drag_offset, 'selected_hint_index': selected_hint_index,
            'game_over_state': game_over_state, 'showing_stats': showing_stats,
            'stats_dialog_dragging': stats_dialog_dragging, 'dragging': dragging,
            'specifying_rack': specifying_rack, 'specify_rack_active_input': specify_rack_active_input,
            'specify_rack_inputs': specify_rack_inputs, 'specify_rack_original_racks': specify_rack_original_racks,
            'specify_rack_proposed_racks': specify_rack_proposed_racks, 'confirming_override': confirming_override,
            'showing_simulation_config': showing_simulation_config, 'simulation_config_active_input': simulation_config_active_input,
            'turn': turn, 'pass_count': pass_count, 'exchange_count': exchange_count,
            'consecutive_zero_point_turns': consecutive_zero_point_turns, 'human_played': human_played,
            'paused_for_power_tile': paused_for_power_tile, 'paused_for_bingo_practice': paused_for_bingo_practice,
            'practice_solved': practice_solved, # RE-ADD packing
            'showing_practice_end_dialog': showing_practice_end_dialog,
            'replay_mode': replay_mode, 'current_replay_turn': current_replay_turn,
            'last_played_highlight_coords': last_played_highlight_coords, 'selected_tiles': selected_tiles,
            'word_positions': word_positions, 'original_tiles': original_tiles, 'original_rack': original_rack,
            'preview_score_enabled': preview_score_enabled, 'all_moves': all_moves, # Ensure all_moves is packed back
            'racks': racks, 'bag': bag, 'blanks': blanks, 'tiles': tiles, 'scores': scores, 'move_history': move_history,
            'last_left_click_pos': last_left_click_pos, 'last_left_click_time': last_left_click_time,
            'hint_moves': hint_moves,
            'stats_dialog_x': stats_dialog_x, 'stats_dialog_y': stats_dialog_y,
            'stats_scroll_offset': stats_scroll_offset,
            'stats_dialog_drag_offset': stats_dialog_drag_offset, 'stats_total_content_height': stats_total_content_height, # Pack stats total height
            'all_words_scroll_offset': all_words_scroll_offset,
            'simulation_config_inputs': simulation_config_inputs,
            'dialog_x': dialog_x, 'dialog_y': dialog_y,
            'current_power_tile': current_power_tile,
            'practice_end_message': practice_end_message,
            'letter_checks': letter_checks,
            'number_checks': number_checks,
            'final_scores': final_scores,
            'initial_racks': initial_racks,
            # 'gaddag_loading_status': gaddag_loading_status, # Don't pack back, use global
            'restart_practice_mode': restart_practice_mode, # Pack the new flag
            'current_r': current_r, 'current_c': current_c, # Pack cursor state
            'typing_direction': typing_direction, 'typing_start': typing_start, # Pack typing direction and start
            'board_tile_counts': board_tile_counts, # Pack updated counter
            # --- MODIFICATION: Pack exchange info ---
            'best_exchange_for_hint': best_exchange_for_hint,
            'best_exchange_score_for_hint': best_exchange_score_for_hint
        })

        return updated_state




def does_move_form_five_letter_word(move, current_tiles, current_blanks):
    """
    Checks if a given move forms at least one 5-letter word.

    Args:
        move (dict): The move dictionary containing 'newly_placed' and 'blanks'.
        current_tiles (list[list[str]]): The current state of the board tiles.
        current_blanks (set): The current set of blank coordinates on the board.

    Returns:
        bool: True if the move forms at least one 5-letter word, False otherwise.
    """
    newly_placed_details = move.get('newly_placed', [])
    if not newly_placed_details:
        return False # Cannot form a word without placing tiles

    # Simulate the move on temporary copies
    temp_tiles = [row[:] for row in current_tiles]
    temp_blanks = current_blanks.copy()
    move_blanks_coords = move.get('blanks', set())

    for r, c, letter in newly_placed_details:
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            temp_tiles[r][c] = letter
            if (r, c) in move_blanks_coords:
                temp_blanks.add((r, c))
        else:
            # Invalid placement within the move data itself
            print(f"Warning (does_move_form_five_letter_word): Invalid position ({r},{c}) in move data.")
            return False

    # Find all words formed by this simulated placement
    words_formed_details = find_all_words_formed(newly_placed_details, temp_tiles)

    # Check if any formed word has length 5
    for word_detail in words_formed_details:
        word_len = len("".join(t[2] for t in word_detail))
        if word_len == 5:
            return True # Found at least one 5-letter word

    return False # No 5-letter words were formed





# Function to Replace: process_game_events (Complete)
# REASON: Add detailed debug prints before calling is_valid_play_cython
#         to diagnose the "Invalid play" error.

def process_game_events(state, drawn_rects): # Added drawn_rects parameter
    """
    Handles the main event loop, processing user input and system events.
    Calls Cython versions of helpers for typed play validation.
    Includes debug prints for invalid play diagnosis.
    """
    # Unpack frequently used control flow flags for slightly cleaner access inside loop
    running_inner = state['running_inner']
    return_to_mode_selection = state['return_to_mode_selection']
    # Unpack pyperclip availability
    pyperclip_available = state['pyperclip_available']
    pyperclip = state['pyperclip']

    # --- Initialize local cursor variables from state ---
    current_r = state.get('current_r')
    current_c = state.get('current_c')
    typing_direction = state.get('typing_direction')
    typing_start = state.get('typing_start')
    # --- End Initialization ---

    # --- Unpack board_tile_counts ---
    board_tile_counts = state['board_tile_counts']


    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running_inner = False
            return_to_mode_selection = False # Explicitly set to False on QUIT
            break # Exit event loop immediately

        # --- MOUSEBUTTONDOWN ---
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Pass the full state dictionary AND drawn_rects to the handler
            mouse_down_result = handle_mouse_down_event(event, state, drawn_rects) # Pass drawn_rects
            # Update the main state dictionary with changes from the handler
            state.update(mouse_down_result)
            # Update local control flags if they were changed by the handler
            running_inner = state['running_inner']
            return_to_mode_selection = state['return_to_mode_selection'] # Get updated value
            # --- Update local board_tile_counts if modified by handler ---
            board_tile_counts = state['board_tile_counts']

            # --- ADD RESYNC ---
            # Re-sync local cursor variables after potential modification in mouse handler
            current_r = state.get('current_r')
            current_c = state.get('current_c')
            typing_direction = state.get('typing_direction')
            typing_start = state.get('typing_start')
            # --- END RESYNC ---


        # --- MOUSEMOTION ---
        elif event.type == pygame.MOUSEMOTION and not state['is_batch_running']:
            # Handle dragging tiles
            if state['dragged_tile'] and state['drag_pos']:
                state['drag_pos'] = event.pos # Update position while dragging
            # Handle dragging game over dialog
            if state['game_over_state'] and state['dragging']:
                x, y = event.pos
                state['dialog_x'] = x - state['drag_offset'][0]
                state['dialog_y'] = y - state['drag_offset'][1]
                # Clamp dialog position to screen bounds
                state['dialog_x'] = max(0, min(state['dialog_x'], WINDOW_WIDTH - DIALOG_WIDTH))
                state['dialog_y'] = max(0, min(state['dialog_y'], WINDOW_HEIGHT - DIALOG_HEIGHT))
            # Handle dragging stats dialog
            if state['showing_stats'] and state['stats_dialog_dragging']:
                x, y = event.pos
                state['stats_dialog_x'] = x - state['stats_dialog_drag_offset'][0]
                state['stats_dialog_y'] = y - state['stats_dialog_drag_offset'][1]
                # Clamp dialog position
                state['stats_dialog_x'] = max(0, min(state['stats_dialog_x'], WINDOW_WIDTH - 480)) # Use stats dialog width
                state['stats_dialog_y'] = max(0, min(state['stats_dialog_y'], WINDOW_HEIGHT - 600)) # Use stats dialog height

        # --- MOUSEBUTTONUP ---
        elif event.type == pygame.MOUSEBUTTONUP and not state['is_batch_running']:
            x, y = event.pos
            if event.button == 1: # Left mouse button up
                # Stop dragging dialogs
                if state['game_over_state'] and state['dragging']:
                    state['dragging'] = False
                if state['showing_stats'] and state['stats_dialog_dragging']:
                    state['stats_dialog_dragging'] = False
                # Handle dropping a dragged tile
                elif state['dragged_tile'] and (0 <= state['dragged_tile'][0]-1 < len(state['is_ai']) and (not state['is_ai'][state['dragged_tile'][0]-1] or state['paused_for_power_tile'] or state['paused_for_bingo_practice'])) and not state['replay_mode']:
                    player_idx = state['dragged_tile'][0] - 1
                    rack_y = BOARD_SIZE + 80 if state['dragged_tile'][0] == 1 else BOARD_SIZE + 150
                    # Calculate rack start x dynamically (consistent with drawing)
                    rack_width_calc = 7 * (TILE_WIDTH + TILE_GAP) - TILE_GAP
                    replay_area_end_x = 10 + 4 * (REPLAY_BUTTON_WIDTH + REPLAY_BUTTON_GAP) # 250
                    min_rack_start_x = replay_area_end_x + BUTTON_GAP + 20 # 280
                    rack_start_x_calc = max(min_rack_start_x, (BOARD_SIZE - rack_width_calc) // 2)
                    rack_area_rect = pygame.Rect(rack_start_x_calc, rack_y, rack_width_calc, TILE_HEIGHT)

                    # Check if dropped on the rack area
                    if rack_area_rect.collidepoint(x, y):
                        # Reorder rack based on drop position
                        if 0 <= player_idx < len(state['racks']):
                            player_rack = state['racks'][player_idx]
                            rack_len = len(player_rack)
                            insert_idx_raw = get_insertion_index(x, rack_start_x_calc, rack_len)
                            original_tile_idx = state['dragged_tile'][1]

                            if 0 <= original_tile_idx < rack_len:
                                tile_to_move = player_rack.pop(original_tile_idx)
                                # Adjust insertion index if the removed tile was before the target index
                                insert_idx_adjusted = insert_idx_raw
                                if original_tile_idx < insert_idx_raw:
                                    insert_idx_adjusted -= 1
                                # Clamp index just in case
                                insert_idx_final = max(0, min(insert_idx_adjusted, len(player_rack)))
                                player_rack.insert(insert_idx_final, tile_to_move)
                                # state['racks'][player_idx] is already updated by modifying player_rack

                    # Always stop dragging
                    state['dragged_tile'] = None
                    state['drag_pos'] = None

        # --- MOUSEWHEEL ---
        elif event.type == pygame.MOUSEWHEEL and not state['is_batch_running']:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Scroll All Words Dialog
            if state['showing_all_words']:
                dialog_rect_all = pygame.Rect((WINDOW_WIDTH - ALL_WORDS_DIALOG_WIDTH) // 2, (WINDOW_HEIGHT - ALL_WORDS_DIALOG_HEIGHT) // 2, ALL_WORDS_DIALOG_WIDTH, ALL_WORDS_DIALOG_HEIGHT)
                if dialog_rect_all.collidepoint(mouse_x, mouse_y):
                    if state['practice_mode'] == "eight_letter": moves_for_scroll = state['practice_target_moves']
                    elif state['practice_mode'] == "power_tiles" and state['paused_for_power_tile']: moves_for_scroll = sorted([m for m in state['all_moves'] if any(letter == state['current_power_tile'] for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), state['number_checks'])], key=lambda m: m['score'], reverse=True)
                    elif state['practice_mode'] == "bingo_bango_bongo" and state['paused_for_bingo_practice']: moves_for_scroll = sorted([m for m in state['all_moves'] if m.get('is_bingo', False)], key=lambda m: m['score'], reverse=True)
                    else: moves_for_scroll = state['all_moves']
                    content_height = len(moves_for_scroll) * 30; header_height = 40; button_area_height = BUTTON_HEIGHT + 30; visible_content_height = ALL_WORDS_DIALOG_HEIGHT - header_height - button_area_height
                    if content_height > visible_content_height: max_scroll = content_height - visible_content_height; state['all_words_scroll_offset'] -= event.y * SCROLL_SPEED; state['all_words_scroll_offset'] = max(0, min(state['all_words_scroll_offset'], max_scroll))
                    else: state['all_words_scroll_offset'] = 0
            # Scroll Stats Dialog
            elif state['showing_stats']:
                stats_dialog_rect = pygame.Rect(state['stats_dialog_x'], state['stats_dialog_y'], 480, 600)
                if stats_dialog_rect.collidepoint(mouse_x, mouse_y):
                    padding = 10; button_area_height = BUTTON_HEIGHT + padding * 2; visible_content_height = 600 - padding * 2 - button_area_height
                    # Get total content height from drawn_rects
                    stats_total_content_height = drawn_rects.get('stats_total_content_height', 0)
                    if stats_total_content_height > visible_content_height:
                        max_scroll = stats_total_content_height - visible_content_height
                        state['stats_scroll_offset'] -= event.y * SCROLL_SPEED
                        state['stats_scroll_offset'] = max(0, min(state['stats_scroll_offset'], max_scroll))
                    else:
                        state['stats_scroll_offset'] = 0
            # Scroll Scoreboard
            else:
                sb_x = BOARD_SIZE + 275; sb_y = 40; sb_w = max(200, WINDOW_WIDTH - sb_x - 20); sb_h = WINDOW_HEIGHT - 80;
                if sb_x + sb_w > WINDOW_WIDTH - 10: sb_w = WINDOW_WIDTH - sb_x - 10
                if sb_w < 150: sb_x = WINDOW_WIDTH - 160; sb_w = 150
                scoreboard_rect = pygame.Rect(sb_x, sb_y, sb_w, sb_h)
                if scoreboard_rect.collidepoint(mouse_x, mouse_y):
                    history_to_draw = state['move_history'][:state['current_replay_turn']] if state['replay_mode'] else state['move_history']; history_len = len(history_to_draw); total_content_height = history_len * 20
                    is_final_turn_in_replay = state['replay_mode'] and state['current_replay_turn'] == len(state['move_history'])
                    if (state['game_over_state'] or is_final_turn_in_replay) and state['final_scores'] is not None: total_content_height += 40
                    scoreboard_height = sb_h
                    if total_content_height > scoreboard_height: max_scroll = total_content_height - scoreboard_height; state['scroll_offset'] -= event.y * SCROLL_SPEED; state['scroll_offset'] = max(0, min(state['scroll_offset'], max_scroll))
                    else: state['scroll_offset'] = 0

        # --- KEYDOWN ---
        elif event.type == pygame.KEYDOWN:
            # Handle dialog inputs first
            if state['specifying_rack'] and state['specify_rack_active_input'] is not None:
                idx = state['specify_rack_active_input']
                if event.key == pygame.K_BACKSPACE: state['specify_rack_inputs'][idx] = state['specify_rack_inputs'][idx][:-1]
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    # Use rect from drawn_rects
                    confirm_rect_sr = drawn_rects.get('confirm_rect_sr')
                    if confirm_rect_sr: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': confirm_rect_sr.center, 'button': 1}))
                elif event.key == pygame.K_TAB: state['specify_rack_active_input'] = 1 - idx
                elif len(state['specify_rack_inputs'][idx]) < 7:
                    char = event.unicode.upper()
                    if 'A' <= char <= 'Z' or char == '?' or char == ' ': state['specify_rack_inputs'][idx] += char
            elif state['showing_simulation_config'] and state['simulation_config_active_input'] is not None:
                idx = state['simulation_config_active_input']
                if event.key == pygame.K_BACKSPACE: state['simulation_config_inputs'][idx] = state['simulation_config_inputs'][idx][:-1]
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    # Use rect from drawn_rects
                    sim_simulate_rect = drawn_rects.get('sim_simulate_rect')
                    if sim_simulate_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': sim_simulate_rect.center, 'button': 1}))
                elif event.key == pygame.K_TAB: state['simulation_config_active_input'] = (idx + 1) % len(state['simulation_config_inputs'])
                elif event.unicode.isdigit(): state['simulation_config_inputs'][idx] += event.unicode

            # Handle main game typing
            elif state['selected_square'] and not (state['exchanging'] or state['hinting'] or state['showing_all_words'] or state['specifying_rack'] or state['showing_simulation_config']):
                current_player_idx = state['turn'] - 1
                is_human_turn_or_paused = (0 <= current_player_idx < len(state['is_ai'])) and (not state['is_ai'][current_player_idx] or state['paused_for_power_tile'] or state['paused_for_bingo_practice'])

                if is_human_turn_or_paused:
                    mods = pygame.key.get_mods() # Get modifier keys status
                    # --- PASTE LOGIC: Check for Modifier Key ---
                    if event.key == pygame.K_v and (mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META) and pyperclip_available:
                        try:
                            pasted_text = pyperclip.paste()
                            if pasted_text and pasted_text.isalpha():
                                pasted_text = pasted_text.upper()
                                print(f"Pasting: {pasted_text}")
                                local_current_r = current_r; local_current_c = current_c; local_typing_direction = typing_direction
                                if not state['typing']:
                                    state['typing'] = True; state['original_tiles'] = [row[:] for row in state['tiles']]; state['original_rack'] = state['racks'][state['turn']-1][:]; state['typing_start'] = state['selected_square'][:2]; state['typing_direction'] = state['selected_square'][2]; state['word_positions'] = []
                                    local_current_r, local_current_c = state['typing_start']; local_typing_direction = state['typing_direction']
                                    state['current_r'], state['current_c'] = local_current_r, local_current_c; state['typing_direction'] = local_typing_direction
                                elif local_current_r is None or local_current_c is None or local_typing_direction is None: print("  Error: Cannot paste, current cursor state (r,c,direction) is invalid."); pasted_text = ""
                                for letter in pasted_text:
                                    if local_current_r is None or local_current_c is None or not (0 <= local_current_r < GRID_SIZE and 0 <= local_current_c < GRID_SIZE): print("  Typing cursor out of bounds. Stopping paste."); break
                                    use_blank = False
                                    if letter not in state['racks'][state['turn']-1]:
                                        if ' ' in state['racks'][state['turn']-1]: use_blank = True
                                        else: print(f"  Cannot place '{letter}' (not in rack and no blanks). Stopping paste."); break
                                    state['tiles'][local_current_r][local_current_c] = letter; state['word_positions'].append((local_current_r, local_current_c, letter))
                                    if use_blank: state['racks'][state['turn']-1].remove(' '); state['blanks'].add((local_current_r, local_current_c))
                                    else: state['racks'][state['turn']-1].remove(letter)
                                    if local_typing_direction == "right":
                                        local_current_c += 1
                                        while 0 <= local_current_c < GRID_SIZE and state['original_tiles'][local_current_r][local_current_c]: local_current_c += 1
                                    elif local_typing_direction == "down":
                                        local_current_r += 1
                                        while 0 <= local_current_r < GRID_SIZE and state['original_tiles'][local_current_r][local_current_c]: local_current_r += 1
                                    else: print(f"  Error: Invalid typing direction '{local_typing_direction}' during paste. Stopping."); break
                                    state['current_r'], state['current_c'] = local_current_r, local_current_c; current_r, current_c = local_current_r, local_current_c
                                    if not (0 <= local_current_r < GRID_SIZE and 0 <= local_current_c < GRID_SIZE): print("  Typing cursor moved out of bounds after placement. Stopping paste."); break
                        except Exception as e: print(f"Error during paste: {e}")
                    # --- END PASTE LOGIC ---

                    # --- NORMAL TYPING LOGIC ---
                    elif event.unicode.isalpha() and len(event.unicode) == 1:
                        letter = event.unicode.upper()
                        current_rack_debug = state['racks'][state['turn']-1]; has_letter = letter in current_rack_debug; has_blank = ' ' in current_rack_debug
                        if has_letter or has_blank:
                            if not state['typing']:
                                state['typing'] = True; state['original_tiles'] = [row[:] for row in state['tiles']]; state['original_rack'] = state['racks'][state['turn']-1][:]; state['typing_start'] = state['selected_square'][:2]; state['typing_direction'] = state['selected_square'][2]; state['word_positions'] = []
                                current_r, current_c = state['typing_start']; typing_direction = state['typing_direction']; state['current_r'], state['current_c'] = current_r, current_c; state['typing_direction'] = typing_direction
                            elif current_r is None or current_c is None or typing_direction is None: print("ERROR: Typing mode active but cursor state invalid. Resetting typing."); state['typing'] = False; state['word_positions'] = []; state['original_tiles'] = None; state['original_rack'] = None; selected_square = None; current_r = None; current_c = None; typing_direction = None; typing_start = None; state['current_r'] = None; state['current_c'] = None; state['typing_direction'] = None; state['typing_start'] = None; continue
                            use_blank = False
                            if not has_letter and has_blank: use_blank = True
                            if current_r is not None and current_c is not None and 0 <= current_r < GRID_SIZE and 0 <= current_c < GRID_SIZE:
                                state['tiles'][current_r][current_c] = letter; state['word_positions'].append((current_r, current_c, letter))
                                if use_blank: state['racks'][state['turn']-1].remove(' '); state['blanks'].add((current_r, current_c))
                                else: state['racks'][state['turn']-1].remove(letter)
                                if typing_direction == "right":
                                    current_c += 1
                                    while 0 <= current_c < GRID_SIZE and state['original_tiles'][current_r][current_c]: current_c += 1
                                elif typing_direction == "down": # Use elif here
                                    current_r += 1
                                    while 0 <= current_r < GRID_SIZE and state['original_tiles'][current_r][current_c]: current_r += 1
                                state['current_r'] = current_r; state['current_c'] = current_c
                            else: print(f"Warning: Attempted to type '{letter}' at invalid cursor ({current_r},{current_c})")
                        # else: pass # Letter not available
                    # --- END NORMAL TYPING LOGIC ---

                    elif event.key == pygame.K_BACKSPACE and state['typing']:
                        if state['word_positions']:
                            last_r, last_c, last_letter = state['word_positions'].pop(); state['tiles'][last_r][last_c] = ''
                            tile_to_return = ' ' if (last_r, last_c) in state['blanks'] else last_letter
                            state['racks'][state['turn']-1].append(tile_to_return)
                            if not state['is_ai'][state['turn']-1]: state['racks'][state['turn']-1].sort()
                            if (last_r, last_c) in state['blanks']: state['blanks'].remove((last_r, last_c))
                            current_r, current_c = last_r, last_c; state['current_r'], state['current_c'] = current_r, current_c
                            if not state['word_positions']: state['typing'] = False; state['original_tiles'] = None; state['original_rack'] = None; current_r, current_c = None, None; typing_direction = None; typing_start = None; state['current_r'] = None; state['current_c'] = None; state['typing_direction'] = None; state['typing_start'] = None
                        else: state['typing'] = False; state['original_tiles'] = None; state['original_rack'] = None; current_r, current_c = None, None; typing_direction = None; typing_start = None; state['current_r'] = None; state['current_c'] = None; state['typing_direction'] = None; state['typing_start'] = None

                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and state['typing']:
                        # Finalize the typed play
                        if state['word_positions']:
                            newly_placed_details = [(r, c, l) for r, c, l in state['word_positions']];
                            initial_rack_size_for_play = len(state['original_rack']) if state['original_rack'] else 0

                            # --- DEBUG PRINTS (Keep for now) ---
                            print("\n--- DEBUG: Finalizing Typed Play ---")
                            print(f"  Newly Placed: {newly_placed_details}")
                            print(f"  First Play? {state['first_play']}")
                            print(f"  Initial Rack Size: {initial_rack_size_for_play}")
                            print(f"  Original Rack: {state['original_rack']}")
                            # --- ADDED: Print original_tiles and current tiles ---
                            print(f"  Original Tiles (at typing start):")
                            # for r_idx, row_data in enumerate(state['original_tiles']): print(f"    {r_idx}: {''.join(t if t else '.' for t in row_data)}")
                            print(f"  Current Tiles (with typed letters):")
                            # for r_idx, row_data in enumerate(state['tiles']): print(f"    {r_idx}: {''.join(t if t else '.' for t in row_data)}")
                            # --- END ADDED ---
                            print("  Calling is_valid_play_cython...") # Updated print message
                            # --- END DEBUG PRINTS ---

                            # --- Call Cython version of is_valid_play ---
                            is_valid, is_bingo = is_valid_play_cython(newly_placed_details, state['tiles'], state['first_play'], initial_rack_size_for_play, state['original_tiles'], state['original_rack']) # Pass original_rack here
                            # --- End Call Cython version ---

                            print(f"  is_valid_play_cython returned: is_valid={is_valid}, is_bingo={is_bingo}") # Updated print message

                            if is_valid:
                                # --- Use Cython version of calculate_score ---
                                score = calculate_score_cython(newly_placed_details, state['board'], state['tiles'], state['blanks']); proceed_with_finalization = True
                                # --- End Use Cython version ---

                                # --- Practice Mode Validation ---
                                if state['practice_mode'] == "power_tiles" and state['paused_for_power_tile']:
                                    is_power_tile_play = any(letter == state['current_power_tile'] for _, _, letter in newly_placed_details); power_moves_filtered = [m for m in state['all_moves'] if any(letter == state['current_power_tile'] for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), state['number_checks'])]; max_power_score = max(m['score'] for m in power_moves_filtered) if power_moves_filtered else 0
                                    if is_power_tile_play and score >= max_power_score: show_message_dialog(f"Correct! You found the highest scoring play ({score} pts) with {state['current_power_tile']} matching the selected lengths.", "Power Tile Success!")
                                    elif is_power_tile_play: show_message_dialog(f"You played the {state['current_power_tile']}, but there is a higher score: ({max_power_score}). Try again!", "Incorrect Score"); proceed_with_finalization = False
                                    else: show_message_dialog(f"You didn't use the required power tile: {state['current_power_tile']}. Try again!", "Incorrect Play"); proceed_with_finalization = False
                                    if not proceed_with_finalization: # Revert
                                        for r_wp, c_wp, _ in state['word_positions']: state['tiles'][r_wp][c_wp] = state['original_tiles'][r_wp][c_wp]
                                        state['racks'][state['turn']-1] = state['original_rack'][:]
                                        if not state['is_ai'][state['turn']-1]: state['racks'][state['turn']-1].sort()
                                        blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in state['word_positions'] if (r_wp, c_wp) in state['blanks']); state['blanks'].difference_update(blanks_to_remove); state['typing'] = False; state['word_positions'] = []; state['original_tiles'] = None; state['original_rack'] = None; state['selected_square'] = None; current_r = None; current_c = None; state['current_r'] = None; state['current_c'] = None
                                elif state['practice_mode'] == "bingo_bango_bongo" and state['paused_for_bingo_practice']:
                                    bingo_moves = [m for m in state['all_moves'] if m.get('is_bingo', False)]; max_bingo_score = max(m['score'] for m in bingo_moves) if bingo_moves else 0
                                    if is_bingo and score >= max_bingo_score: show_message_dialog(f"Correct! You found the highest scoring bingo ({score} pts).", "Bingo Success!"); proceed_with_finalization = True
                                    elif is_bingo: show_message_dialog(f"You played a bingo, but score {score} is not the highest ({max_bingo_score}). Try again!", "Incorrect Score"); proceed_with_finalization = False
                                    else: show_message_dialog("That wasn't a bingo. Try again!", "Incorrect Play"); proceed_with_finalization = False
                                    if not proceed_with_finalization: # Revert
                                        for r_wp, c_wp, _ in state['word_positions']: state['tiles'][r_wp][c_wp] = state['original_tiles'][r_wp][c_wp]
                                        state['racks'][state['turn']-1] = state['original_rack'][:]
                                        if not state['is_ai'][state['turn']-1]: state['racks'][state['turn']-1].sort()
                                        blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in state['word_positions'] if (r_wp, c_wp) in state['blanks']); state['blanks'].difference_update(blanks_to_remove); state['typing'] = False; state['word_positions'] = []; state['original_tiles'] = None; state['original_rack'] = None; state['selected_square'] = None; current_r = None; current_c = None; state['current_r'] = None; state['current_c'] = None
                                elif state['practice_mode'] == "eight_letter":
                                    # --- Use Cython version of find_all_words_formed ---
                                    all_words_details_8l = find_all_words_formed_cython(newly_placed_details, state['tiles']); played_word_str_8l = ""; played_primary_tiles_8l = []; newly_placed_coords_8l = set((r,c) for r,c,_ in newly_placed_details)
                                    # --- End Use Cython version ---
                                    for word_detail in all_words_details_8l:
                                        if any((t[0], t[1]) in newly_placed_coords_8l for t in word_detail): played_word_str_8l = "".join(t[2] for t in word_detail); played_primary_tiles_8l = word_detail; break
                                    max_score_8l = state['practice_target_moves'][0].get('score', 0) if state['practice_target_moves'] else 0
                                    if is_bingo and score >= max_score_8l and max_score_8l > 0:
                                        print(f"8-Letter Bingo CORRECT! Played: '{played_word_str_8l}' ({score} pts), Max Score: {max_score_8l}"); state['practice_solved'] = True; state['showing_practice_end_dialog'] = True; word_with_blanks_played = ""
                                        if played_primary_tiles_8l: move_blanks_coords_played = set((r,c) for r,c in newly_placed_coords_8l if (r,c) in state['blanks']); word_with_blanks_list = [];
                                        for wr, wc, w_letter in played_primary_tiles_8l: is_blank_in_word = (wr, wc) in newly_placed_coords_8l and (wr, wc) in move_blanks_coords_played; word_with_blanks_list.append(w_letter.lower() if is_blank_in_word else w_letter.upper()); word_with_blanks_played = "".join(word_with_blanks_list)
                                        else: word_with_blanks_played = played_word_str_8l.upper()
                                        state['practice_end_message'] = f"Correct! You found the highest scoring bingo:\n{word_with_blanks_played} ({score} pts)"; proceed_with_finalization = False
                                    else:
                                        print(f"8-Letter Bingo INCORRECT. Played: '{played_word_str_8l}' ({score} pts), Is Bingo: {is_bingo}, Target Max Score: {max_score_8l}"); incorrect_msg = f"Try again. The highest score is {max_score_8l}."; show_message_dialog(incorrect_msg, "8-Letter Bingo")
                                        for r_wp, c_wp, _ in state['word_positions']: state['tiles'][r_wp][c_wp] = state['original_tiles'][r_wp][c_wp]
                                        state['racks'][state['turn']-1] = state['original_rack'][:]
                                        if not state['is_ai'][state['turn']-1]: state['racks'][state['turn']-1].sort()
                                        blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in state['word_positions'] if (r_wp, c_wp) in state['blanks']); state['blanks'].difference_update(blanks_to_remove); state['typing'] = False; state['word_positions'] = []; state['original_tiles'] = None; state['original_rack'] = None; state['selected_square'] = None; current_r = None; current_c = None; state['current_r'] = None; state['current_c'] = None; proceed_with_finalization = False
                                elif state['practice_mode'] == "only_fives":
                                    # --- Use Cython version of find_all_words_formed ---
                                    # (Need to call it here to check the condition)
                                    temp_all_words = find_all_words_formed_cython(newly_placed_details, state['tiles'])
                                    if not any(len("".join(t[2] for t in word_detail)) == 5 for word_detail in temp_all_words):
                                    # --- End Use Cython version ---
                                        show_message_dialog("At least one 5-letter word must be formed.", "Invalid Play"); proceed_with_finalization = False
                                        for r_wp, c_wp, _ in state['word_positions']: state['tiles'][r_wp][c_wp] = state['original_tiles'][r_wp][c_wp]
                                        state['racks'][state['turn']-1] = state['original_rack'][:]
                                        if not state['is_ai'][state['turn']-1]: state['racks'][state['turn']-1].sort()
                                        blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in state['word_positions'] if (r_wp, c_wp) in state['blanks']); state['blanks'].difference_update(blanks_to_remove); state['typing'] = False; state['word_positions'] = []; state['original_tiles'] = None; state['original_rack'] = None; state['selected_square'] = None; current_r = None; current_c = None; state['current_r'] = None; state['current_c'] = None

                                # --- Finalize Valid Play ---
                                if proceed_with_finalization:
                                    # Update board tile counts
                                    for r_fin, c_fin, letter_fin in newly_placed_details:
                                        board_tile_counts[letter_fin] += 1

                                    state['scores'][state['turn']-1] += score;
                                    # --- Use Cython version of find_all_words_formed ---
                                    all_words_formed_details = find_all_words_formed_cython(newly_placed_details, state['tiles'])
                                    # --- End Use Cython version ---
                                    primary_word_tiles = []; primary_word_str = ""; start_pos = state.get('typing_start'); orientation = state.get('typing_direction')

                                    # Derive start/orientation if needed
                                    if start_pos is None or orientation is None:
                                        if newly_placed_details:
                                            rows = set(r for r,c,_ in newly_placed_details)
                                            cols = set(c for r,c,_ in newly_placed_details)
                                            if len(rows) == 1: orientation = 'right'; start_pos = min(newly_placed_details, key=lambda x: x[1])[:2]
                                            elif len(cols) == 1: orientation = 'down'; start_pos = min(newly_placed_details, key=lambda x: x[0])[:2]
                                            else: orientation = '?'; start_pos = newly_placed_details[0][:2]
                                        else: orientation = '?'; start_pos = (0,0)

                                    word_with_blanks = ""; newly_placed_coords = set((r,c) for r,c,_ in newly_placed_details); move_blanks_coords = set((r,c) for r,c in newly_placed_coords if (r,c) in state['blanks'])
                                    # Find primary word
                                    if all_words_formed_details:
                                        found_primary = False;
                                        for word_detail in all_words_formed_details:
                                            is_along_axis = False;
                                            if orientation == 'right' and len(set(r for r,c,l in word_detail)) == 1: is_along_axis = True
                                            elif orientation == 'down' and len(set(c for r,c,l in word_detail)) == 1: is_along_axis = True
                                            if is_along_axis and any((t[0], t[1]) in newly_placed_coords for t in word_detail): primary_word_tiles = word_detail; found_primary = True; break
                                        if not found_primary:
                                            longest_len = 0;
                                            for word_detail in all_words_formed_details:
                                                if any((t[0], t[1]) in newly_placed_coords for t in word_detail):
                                                    if len(word_detail) > longest_len: longest_len = len(word_detail); primary_word_tiles = word_detail;
                                                    if len(set(r for r,c,l in primary_word_tiles)) == 1: orientation = 'right'
                                                    elif len(set(c for r,c,l in primary_word_tiles)) == 1: orientation = 'down'
                                    # Format primary word details
                                    if primary_word_tiles:
                                        primary_word_str = "".join(t[2] for t in primary_word_tiles);
                                        if orientation == 'right': start_pos = min(primary_word_tiles, key=lambda x: x[1])[:2]
                                        elif orientation == 'down': start_pos = min(primary_word_tiles, key=lambda x: x[0])[:2]
                                        word_with_blanks_list = [];
                                        for wr, wc, w_letter in primary_word_tiles: is_blank_in_word = (wr, wc) in newly_placed_coords and (wr, wc) in move_blanks_coords; word_with_blanks_list.append(w_letter.lower() if is_blank_in_word else w_letter.upper()); word_with_blanks = "".join(word_with_blanks_list)
                                    else: print("  Warning: Could not determine primary word for history."); primary_word_str = "".join(l for r,c,l in newly_placed_details); word_with_blanks = primary_word_str

                                    # Draw tiles, update history, etc.
                                    num_to_draw = len(newly_placed_details); drawn_tiles = [state['bag'].pop() for _ in range(num_to_draw) if state['bag']]; state['racks'][state['turn']-1].extend(drawn_tiles);
                                    if not state['is_ai'][state['turn']-1]: state['racks'][state['turn']-1].sort()
                                    luck_factor = 0.0;
                                    if drawn_tiles: drawn_leave_value = evaluate_leave(drawn_tiles); expected_draw_value_simple = state['current_turn_pool_quality_score'] * len(drawn_tiles); luck_factor = drawn_leave_value - expected_draw_value_simple; drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles)); print(f"  Drew: {drawn_tiles_str}, Leave Value: {drawn_leave_value:.2f}, Luck: {luck_factor:+.2f}")
                                    move_data = {'player': state['turn'], 'move_type': 'place', 'rack': state['original_rack'], 'positions': [(t[0], t[1], t[2]) for t in primary_word_tiles], 'blanks': move_blanks_coords, 'score': score, 'word': primary_word_str, 'drawn': drawn_tiles, 'coord': get_coord(start_pos, orientation), 'word_with_blanks': word_with_blanks, 'is_bingo': is_bingo, 'newly_placed': newly_placed_details, 'start': start_pos, 'direction': orientation, 'turn_duration': 0.0, 'pool_quality_before_draw': state['current_turn_pool_quality_score'], 'luck_factor': luck_factor}; state['move_history'].append(move_data); state['current_replay_turn'] = len(state['move_history']); state['last_played_highlight_coords'] = newly_placed_coords
                                    state['first_play'] = False; state['consecutive_zero_point_turns'] = 0; state['pass_count'] = 0; state['exchange_count'] = 0; state['human_played'] = True; state['paused_for_power_tile'] = False; state['paused_for_bingo_practice'] = False; state['turn'] = 3 - state['turn']
                                    if state['practice_solved'] and state['practice_mode'] == "eight_letter": pass # Already handled
                            else: # Play was invalid
                                print("--- DEBUG: Invalid Play Detected by is_valid_play_cython ---") # Updated print
                                show_message_dialog("Invalid play.", "Invalid")
                                if state['original_tiles'] and state['original_rack']:
                                    for r_wp, c_wp, _ in state['word_positions']: state['tiles'][r_wp][c_wp] = state['original_tiles'][r_wp][c_wp]
                                    state['racks'][state['turn']-1] = state['original_rack'][:]
                                    if not state['is_ai'][state['turn']-1]: state['racks'][state['turn']-1].sort()
                                    blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in state['word_positions'] if (r_wp, c_wp) in state['blanks']); state['blanks'].difference_update(blanks_to_remove)
                            # Exit typing mode regardless of validity
                            state['typing'] = False; state['word_positions'] = []; state['original_tiles'] = None; state['original_rack'] = None; state['selected_square'] = None
                            current_r, current_c = None, None; typing_direction = None; typing_start = None
                            state['current_r'], state['current_c'] = None, None; state['typing_direction'] = None; state['typing_start'] = None
                        else: # Enter pressed but no letters typed
                            state['typing'] = False; state['selected_square'] = None
                            current_r, current_c = None, None; typing_direction = None; typing_start = None
                            state['current_r'], state['current_c'] = None, None; state['typing_direction'] = None; state['typing_start'] = None
                    else: # Not a typing key
                         pass # Ignore other keys

                # --- Handle other KEYDOWN events (dialogs, global shortcuts) ---
                elif state['game_over_state'] and not state['is_batch_running']:
                    save_rect = drawn_rects.get('save_rect')
                    quit_rect = drawn_rects.get('quit_rect')
                    replay_rect = drawn_rects.get('replay_rect')
                    play_again_rect = drawn_rects.get('play_again_rect')
                    if event.key == pygame.K_s:
                        if save_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': save_rect.center, 'button': 1}))
                    elif event.key == pygame.K_q:
                        if quit_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': quit_rect.center, 'button': 1}))
                    elif event.key == pygame.K_r:
                        if replay_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': replay_rect.center, 'button': 1}))
                    elif event.key == pygame.K_p:
                        if play_again_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': play_again_rect.center, 'button': 1}))
                elif state['showing_stats'] and event.key == pygame.K_RETURN:
                     stats_ok_button_rect = drawn_rects.get('stats_ok_button_rect')
                     if stats_ok_button_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': stats_ok_button_rect.center, 'button': 1}))
                elif state['hinting'] and event.key == pygame.K_RETURN:
                     play_button_rect = drawn_rects.get('play_button_rect')
                     if play_button_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': play_button_rect.center, 'button': 1}))
                elif state['showing_all_words'] and event.key == pygame.K_RETURN:
                     all_words_play_rect = drawn_rects.get('all_words_play_rect')
                     if all_words_play_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': all_words_play_rect.center, 'button': 1}))
                elif state['exchanging'] and event.key == pygame.K_RETURN:
                     exchange_button_rect = drawn_rects.get('exchange_button_rect')
                     if exchange_button_rect: pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': exchange_button_rect.center, 'button': 1}))
                elif event.key == pygame.K_ESCAPE:
                    # Close dialogs or cancel actions in order of precedence
                    if state['exchanging']: state['exchanging'] = False; state['selected_tiles'].clear()
                    elif state['hinting']: state['hinting'] = False
                    elif state['showing_all_words']: state['showing_all_words'] = False
                    elif state['specifying_rack']: state['specifying_rack'] = False; state['specify_rack_inputs'] = ["", ""]; state['specify_rack_active_input'] = None; state['specify_rack_original_racks'] = [[], []]; state['confirming_override'] = False
                    elif state['showing_simulation_config']: state['showing_simulation_config'] = False; state['simulation_config_active_input'] = None
                    elif state['typing']:
                        # Revert typing state
                        if state['original_tiles'] and state['original_rack']:
                            for r_wp, c_wp, _ in state['word_positions']: state['tiles'][r_wp][c_wp] = state['original_tiles'][r_wp][c_wp]
                            state['racks'][state['turn']-1] = state['original_rack'][:]
                            if not state['is_ai'][state['turn']-1]: state['racks'][state['turn']-1].sort()
                            blanks_to_remove = set((r_wp, c_wp) for r_wp, c_wp, _ in state['word_positions'] if (r_wp, c_wp) in state['blanks'])
                            state['blanks'].difference_update(blanks_to_remove)
                        state['typing'] = False; state['word_positions'] = []; state['original_tiles'] = None; state['original_rack'] = None; state['selected_square'] = None; current_r = None; current_c = None; typing_direction = None; typing_start = None; state['current_r'] = None; state['current_c'] = None; state['typing_direction'] = None; state['typing_start'] = None # Reset cursor state
                    elif state['selected_square']: state['selected_square'] = None; current_r = None; current_c = None; state['current_r'] = None; state['current_c'] = None # Deselect square and reset cursor
                    elif state['dropdown_open']: state['dropdown_open'] = False # Close dropdown
                    elif state['showing_stats']: state['showing_stats'] = False # Close stats
                    elif state['game_over_state']: pass # Don't close game over with Esc
                    elif state['showing_practice_end_dialog']: pass # Don't close practice end with Esc
                    else:
                        # If nothing else is active, open the options dropdown
                        if not state['replay_mode'] and not state['game_over_state']:
                            state['dropdown_open'] = True

    # Update control flags in the state dictionary before returning
    state['running_inner'] = running_inner
    state['return_to_mode_selection'] = return_to_mode_selection

    # Pack potentially modified local cursor state back into the main state dict
    state['current_r'] = current_r
    state['current_c'] = current_c
    state['typing_direction'] = typing_direction
    state['typing_start'] = typing_start
    # --- Pack updated board_tile_counts back into state ---
    state['board_tile_counts'] = board_tile_counts


    return state





# Function to Replace: draw_game_screen
# REASON: Use pre-rendered surfaces from TILE_LETTER_CACHE for tile letters on the board.

def draw_game_screen(screen, state):
    """
    Draws the entire game screen based on the current state.
    Prioritizes GADDAG loading message over batch progress message.
    Reads global GADDAG status directly for indicator display.
    Unpacks board_tile_counts and exchange hint info.
    Uses cached surfaces for board tiles.
    """
    # --- Access global directly for indicator logic ---
    global gaddag_loading_status, TILE_LETTER_CACHE # Add cache access

    # --- Unpack State Variables Needed for Drawing ---
    # (Unpack all variables previously used directly in the main loop's drawing section)
    board = state['board']
    tiles = state['tiles']
    blanks = state['blanks']
    scores = state['scores']
    racks = state['racks']
    turn = state['turn']
    player_names = state['player_names']
    dragged_tile = state['dragged_tile']
    drag_pos = state['drag_pos']
    drag_offset = state['drag_offset'] # Needed for drawing dragged tile correctly
    practice_mode = state['practice_mode']
    bag = state['bag']
    move_history = state['move_history']
    scroll_offset = state['scroll_offset']
    is_ai = state['is_ai']
    final_scores = state['final_scores']
    game_over_state = state['game_over_state']
    replay_mode = state['replay_mode']
    current_replay_turn = state['current_replay_turn']
    is_loaded_game = state['is_loaded_game']
    replay_initial_shuffled_bag = state['replay_initial_shuffled_bag']
    initial_racks = state['initial_racks']
    last_played_highlight_coords = state['last_played_highlight_coords']
    selected_square = state['selected_square']
    typing = state['typing']
    current_r = state['current_r']
    current_c = state['current_c']
    preview_score_enabled = state['preview_score_enabled']
    current_preview_score = state['current_preview_score']
    is_solving_endgame = state['is_solving_endgame']
    is_batch_running = state['is_batch_running']
    current_batch_game_num = state['current_batch_game_num']
    total_batch_games = state['total_batch_games']
    # gaddag_loading_status = state['gaddag_loading_status'] # REMOVED - Use global directly
    showing_simulation_config = state['showing_simulation_config']
    simulation_config_inputs = state['simulation_config_inputs']
    simulation_config_active_input = state['simulation_config_active_input']
    specifying_rack = state['specifying_rack']
    specify_rack_inputs = state['specify_rack_inputs']
    specify_rack_active_input = state['specify_rack_active_input']
    specify_rack_original_racks = state['specify_rack_original_racks']
    confirming_override = state['confirming_override']
    exchanging = state['exchanging']
    selected_tiles = state['selected_tiles']
    hinting = state['hinting']
    hint_moves = state['hint_moves']
    selected_hint_index = state['selected_hint_index']
    showing_all_words = state['showing_all_words']
    all_moves = state['all_moves'] # Needed for 'All Words' dialog content
    practice_target_moves = state['practice_target_moves'] # Needed for 'All Words' dialog content
    paused_for_power_tile = state['paused_for_power_tile'] # Needed for 'All Words' dialog content
    current_power_tile = state['current_power_tile'] # Needed for 'All Words' dialog content
    number_checks = state['number_checks'] # Needed for 'All Words' dialog content
    paused_for_bingo_practice = state['paused_for_bingo_practice'] # Needed for 'All Words' dialog content
    all_words_scroll_offset = state['all_words_scroll_offset']
    showing_practice_end_dialog = state['showing_practice_end_dialog']
    practice_end_message = state['practice_end_message']
    dialog_x = state['dialog_x']
    dialog_y = state['dialog_y']
    reason = state.get('reason', '') # Use .get() for safety if reason isn't always set before game over
    showing_stats = state['showing_stats']
    stats_dialog_x = state['stats_dialog_x']
    stats_dialog_y = state['stats_dialog_y']
    stats_scroll_offset = state['stats_scroll_offset']
    dropdown_open = state['dropdown_open']
    board_tile_counts = state['board_tile_counts'] # Unpack the counter
    # --- MODIFICATION: Unpack exchange hint info ---
    best_exchange_for_hint = state.get('best_exchange_for_hint') # Use .get for safety
    best_exchange_score_for_hint = state.get('best_exchange_score_for_hint', -float('inf')) # Use .get for safety
    # --- END MODIFICATION ---
    bag_count = len(bag) # Calculate here


    # --- Initialize dictionary to store drawn rects ---
    drawn_rects = {}

    # --- Start Drawing ---
    screen.fill(WHITE)

    # Determine state to display (replay vs active)
    turn_to_display = turn
    if replay_mode:
        # Replay mode needs the actual board state at that turn, not the counter
        # We still need to simulate to get the correct tiles_to_display etc.
        if is_loaded_game and replay_initial_shuffled_bag is not None:
            tiles_to_display, blanks_to_display, scores_to_display, racks_to_display = simulate_game_up_to(current_replay_turn, move_history, replay_initial_shuffled_bag)
        elif not is_loaded_game and initial_racks is not None:
            tiles_to_display, blanks_to_display, scores_to_display, racks_to_display = get_replay_state(current_replay_turn, initial_racks)
        else: # Fallback if initial state is missing
            print("Replay Warning: Missing initial state. Displaying turn 0.")
            tiles_to_display, blanks_to_display, scores_to_display, racks_to_display = get_replay_state(0, [[], []]) # Provide default empty racks

        # Determine whose turn it *would* be after the displayed move
        if current_replay_turn == 0:
            turn_to_display = 1 # Start of game
        elif 0 < current_replay_turn <= len(move_history):
            # The turn *after* move N is the opposite of the player who made move N
            turn_to_display = 3 - move_history[current_replay_turn - 1]['player']
        else: # Should not happen if current_replay_turn is clamped, but fallback
            turn_to_display = 1
    else: # Active game state
        tiles_to_display, blanks_to_display, racks_to_display = tiles, blanks, racks
        scores_to_display = final_scores if game_over_state else scores
        turn_to_display = turn

    # Draw Board and Tiles
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            # Draw square background
            square_rect = pygame.Rect(40 + c * SQUARE_SIZE, 40 + r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, board[r][c], square_rect)
            pygame.draw.rect(screen, BLACK, square_rect, 1) # Border

            # Draw tile if present
            if tiles_to_display[r][c]:
                tile_char = tiles_to_display[r][c]
                is_blank_on_board = (r, c) in blanks_to_display
                is_last_played = (r, c) in last_played_highlight_coords and not replay_mode # Highlight only in active game
                tile_bg_color = PALE_YELLOW if is_last_played else GREEN

                # Define tile content area (slightly smaller than square)
                tile_rect = pygame.Rect(square_rect.left + 2, square_rect.top + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4)

                if is_blank_on_board:
                    # Draw background square first for consistency if highlighted
                    pygame.draw.rect(screen, tile_bg_color, tile_rect)
                    # Draw black circle
                    center = tile_rect.center
                    radius = SQUARE_SIZE // 2 - 3
                    pygame.draw.circle(screen, BLACK, center, radius)
                    # --- Use Cache for Blank Letter ---
                    text_surf = TILE_LETTER_CACHE['blank'].get('?')
                    if text_surf:
                        text_rect = text_surf.get_rect(center=center)
                        screen.blit(text_surf, text_rect)
                    # --- End Use Cache ---
                else:
                    # Draw regular tile background
                    pygame.draw.rect(screen, tile_bg_color, tile_rect)
                    # --- Use Cache for Regular Letter ---
                    text_surf = TILE_LETTER_CACHE['regular'].get(tile_char)
                    if text_surf:
                        text_rect = text_surf.get_rect(center=tile_rect.center)
                        screen.blit(text_surf, text_rect)
                    # --- End Use Cache ---

    # Highlight newly placed tiles in replay mode (unchanged)
    if replay_mode and current_replay_turn > 0 and current_replay_turn <= len(move_history):
        last_move_data = move_history[current_replay_turn - 1]
        if last_move_data['move_type'] == 'place':
            newly_placed_coords = last_move_data.get('newly_placed')
            if newly_placed_coords is None:
                original_replay_tiles = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                try:
                    if is_loaded_game and replay_initial_shuffled_bag:
                        original_replay_tiles,_,_,_ = simulate_game_up_to(current_replay_turn - 1, move_history, replay_initial_shuffled_bag)
                    elif not is_loaded_game and initial_racks is not None:
                        original_replay_tiles,_,_,_ = get_replay_state(current_replay_turn - 1, initial_racks)
                except Exception as e:
                    print(f"Error getting previous state for replay highlight: {e}")
                newly_placed_coords = []
                for p_r, p_c, p_l in last_move_data.get('positions', []):
                     if 0 <= p_r < GRID_SIZE and 0 <= p_c < GRID_SIZE and not original_replay_tiles[p_r][p_c]:
                          newly_placed_coords.append((p_r, p_c, p_l))
            if newly_placed_coords:
                for r, c, _ in newly_placed_coords:
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        pygame.draw.rect(screen, YELLOW, (40 + c * SQUARE_SIZE, 40 + r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    # Draw Board Labels (unchanged)
    draw_board_labels(screen, ui_font)

    # Draw Racks (uses cached surfaces via draw_rack)
    p1_alpha_rect, p1_rand_rect, p2_alpha_rect, p2_rand_rect = draw_player_racks(screen, racks_to_display, scores_to_display, turn_to_display, player_names, dragged_tile, drag_pos, practice_mode)
    drawn_rects['p1_alpha_rect'] = p1_alpha_rect
    drawn_rects['p1_rand_rect'] = p1_rand_rect
    drawn_rects['p2_alpha_rect'] = p2_alpha_rect
    drawn_rects['p2_rand_rect'] = p2_rand_rect


    # Draw Remaining Tiles (conditionally - unchanged)
    if practice_mode != "eight_letter":
        current_player_index = turn_to_display - 1
        if 0 <= current_player_index < len(racks_to_display):
            current_board_counts = state['board_tile_counts'] if not replay_mode else Counter()
            remaining = get_remaining_tiles(racks_to_display[current_player_index], current_board_counts)
            draw_remaining_tiles(remaining, turn_to_display)
        else:
            draw_remaining_tiles({}, turn_to_display)

    # Draw Scoreboard (unchanged)
    history_to_draw = move_history[:current_replay_turn] if replay_mode else move_history
    is_final_turn_in_replay = replay_mode and current_replay_turn == len(move_history)
    sb_x = BOARD_SIZE + 275
    sb_y = 40
    sb_w = max(200, WINDOW_WIDTH - sb_x - 20)
    sb_h = WINDOW_HEIGHT - 80
    if sb_x + sb_w > WINDOW_WIDTH - 10: sb_w = WINDOW_WIDTH - sb_x - 10
    if sb_w < 150: sb_x = WINDOW_WIDTH - 160; sb_w = 150
    draw_scoreboard(screen, history_to_draw, scroll_offset, scores_to_display, is_ai, player_names, final_scores=final_scores, game_over_state=game_over_state or is_final_turn_in_replay)

    # Draw Typing Cursor / Selection Arrow (unchanged)
    if selected_square and not typing:
        draw_arrow(selected_square[0], selected_square[1], selected_square[2])
    elif typing:
        if current_r is not None and current_c is not None:
            cursor_x = 40 + current_c * SQUARE_SIZE + SQUARE_SIZE // 2
            cursor_y = 40 + current_r * SQUARE_SIZE + SQUARE_SIZE - 5
            if int(time.time() * 2) % 2 == 0:
                pygame.draw.line(screen, BLACK, (cursor_x - 5, cursor_y), (cursor_x + 5, cursor_y), 2)

    # Draw UI Buttons (Suggest, Simulate, Preview) - Conditionally (unchanged)
    suggest_rect_base = None
    simulate_button_rect = None
    preview_checkbox_rect = None
    if not replay_mode and not game_over_state and not is_batch_running:
        suggest_rect_base = draw_suggest_button()
        if suggest_rect_base:
            simulate_button_rect = pygame.Rect(suggest_rect_base.x, suggest_rect_base.bottom + BUTTON_GAP, OPTIONS_WIDTH, OPTIONS_HEIGHT)
            hover = simulate_button_rect.collidepoint(pygame.mouse.get_pos())
            color = BUTTON_HOVER if hover else BUTTON_COLOR
            pygame.draw.rect(screen, color, simulate_button_rect)
            simulate_text = button_font.render("Simulate", True, BLACK)
            simulate_text_rect = simulate_text.get_rect(center=simulate_button_rect.center)
            screen.blit(simulate_text, simulate_text_rect)
        else:
            simulate_button_rect = None

        is_human_turn_or_paused_practice = not is_ai[turn-1] or paused_for_power_tile or paused_for_bingo_practice
        if is_human_turn_or_paused_practice:
            relevant_rand_rect = p1_rand_rect if turn == 1 else p2_rand_rect
            if relevant_rand_rect:
                preview_checkbox_height = 20
                checkbox_x = relevant_rand_rect.left
                checkbox_y = relevant_rand_rect.top - preview_checkbox_height - BUTTON_GAP
                preview_checkbox_rect = pygame.Rect(checkbox_x, checkbox_y, 20, preview_checkbox_height)
                draw_checkbox(screen, checkbox_x, checkbox_y, preview_score_enabled)
                label_text = "Score Preview: "
                label_surf = ui_font.render(label_text, True, BLACK)
                label_x = checkbox_x + 25
                label_y = checkbox_y + (preview_checkbox_rect.height - label_surf.get_height()) // 2
                screen.blit(label_surf, (label_x, label_y))
                if preview_score_enabled:
                    score_text = str(current_preview_score)
                    score_surf = ui_font.render(score_text, True, BLACK)
                    score_x = label_x + label_surf.get_width() + 2
                    score_y = label_y
                    screen.blit(score_surf, (score_x, score_y))
            else:
                preview_checkbox_rect = None
        else:
             preview_checkbox_rect = None

    drawn_rects['suggest_rect_base'] = suggest_rect_base
    drawn_rects['simulate_button_rect'] = simulate_button_rect
    drawn_rects['preview_checkbox_rect'] = preview_checkbox_rect

    # Draw Indicators (Endgame Solving, Batch Mode, GADDAG Loading) (unchanged)
    if is_solving_endgame:
        draw_endgame_solving_indicator()
    if gaddag_loading_status == 'loading':
        draw_loading_indicator(sb_x, sb_y, sb_w)
    elif is_batch_running:
        batch_text = f"Running Game: {current_batch_game_num} / {total_batch_games}"
        batch_surf = ui_font.render(batch_text, True, BLUE)
        indicator_center_x = sb_x + sb_w // 2
        indicator_top_y = sb_y - batch_surf.get_height() - 5
        batch_rect = batch_surf.get_rect(centerx=indicator_center_x, top=max(5, indicator_top_y))
        screen.blit(batch_surf, batch_rect)


    # --- Draw Dialogs (if active) --- (unchanged logic, uses cached surfaces via draw_rack/draw_hint etc.)
    drawn_rects.update({
        'sim_input_rects': [], 'sim_simulate_rect': None, 'sim_cancel_rect': None,
        'p1_input_rect_sr': None, 'p2_input_rect_sr': None, 'p1_reset_rect_sr': None,
        'p2_reset_rect_sr': None, 'confirm_rect_sr': None, 'cancel_rect_sr': None,
        'go_back_rect_ov': None, 'override_rect_ov': None,
        'tile_rects': [], 'exchange_button_rect': None, 'cancel_button_rect': None,
        'hint_rects': [], 'play_button_rect': None, 'ok_button_rect': None, 'all_words_button_rect': None,
        'all_words_rects': [], 'all_words_play_rect': None, 'all_words_ok_rect': None,
        'practice_play_again_rect': None, 'practice_main_menu_rect': None, 'practice_quit_rect': None,
        'save_rect': None, 'quit_rect': None, 'replay_rect': None, 'play_again_rect': None, 'stats_rect': None,
        'stats_ok_button_rect': None, 'stats_total_content_height': 0
    })

    if showing_simulation_config:
        sim_input_rects, sim_simulate_rect, sim_cancel_rect = draw_simulation_config_dialog(simulation_config_inputs, simulation_config_active_input)
        drawn_rects['sim_input_rects'] = sim_input_rects
        drawn_rects['sim_simulate_rect'] = sim_simulate_rect
        drawn_rects['sim_cancel_rect'] = sim_cancel_rect
    elif specifying_rack:
        p1_name_disp = player_names[0] if player_names[0] else "Player 1"
        p2_name_disp = player_names[1] if player_names[1] else "Player 2"
        p1_input_rect_sr, p2_input_rect_sr, p1_reset_rect_sr, p2_reset_rect_sr, confirm_rect_sr, cancel_rect_sr = draw_specify_rack_dialog(p1_name_disp, p2_name_disp, specify_rack_inputs, specify_rack_active_input, specify_rack_original_racks)
        drawn_rects['p1_input_rect_sr'] = p1_input_rect_sr
        drawn_rects['p2_input_rect_sr'] = p2_input_rect_sr
        drawn_rects['p1_reset_rect_sr'] = p1_reset_rect_sr
        drawn_rects['p2_reset_rect_sr'] = p2_reset_rect_sr
        drawn_rects['confirm_rect_sr'] = confirm_rect_sr
        drawn_rects['cancel_rect_sr'] = cancel_rect_sr
        if confirming_override:
            go_back_rect_ov, override_rect_ov = draw_override_confirmation_dialog()
            drawn_rects['go_back_rect_ov'] = go_back_rect_ov
            drawn_rects['override_rect_ov'] = override_rect_ov
    elif exchanging:
        tile_rects, exchange_button_rect, cancel_button_rect = draw_exchange_dialog(racks[turn-1], selected_tiles)
        drawn_rects['tile_rects'] = tile_rects
        drawn_rects['exchange_button_rect'] = exchange_button_rect
        drawn_rects['cancel_button_rect'] = cancel_button_rect
    elif hinting:
        is_simulation_result = bool(hint_moves and isinstance(hint_moves[0], dict) and 'final_score' in hint_moves[0])
        hint_rects, play_button_rect, ok_button_rect, all_words_button_rect = draw_hint_dialog(
            hint_moves,
            selected_hint_index,
            is_simulation_result=is_simulation_result,
            best_exchange_tiles=best_exchange_for_hint,
            best_exchange_score=best_exchange_score_for_hint
        )
        drawn_rects['hint_rects'] = hint_rects
        drawn_rects['play_button_rect'] = play_button_rect
        drawn_rects['ok_button_rect'] = ok_button_rect
        drawn_rects['all_words_button_rect'] = all_words_button_rect
    elif showing_all_words:
        if practice_mode == "eight_letter": moves_for_all = practice_target_moves
        elif practice_mode == "power_tiles" and paused_for_power_tile: moves_for_all = sorted([m for m in all_moves if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)], key=lambda m: m['score'], reverse=True)
        elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice: moves_for_all = sorted([m for m in all_moves if m.get('is_bingo', False)], key=lambda m: m['score'], reverse=True)
        else: moves_for_all = all_moves
        all_words_rects, all_words_play_rect, all_words_ok_rect = draw_all_words_dialog(moves_for_all, selected_hint_index, all_words_scroll_offset)
        drawn_rects['all_words_rects'] = all_words_rects
        drawn_rects['all_words_play_rect'] = all_words_play_rect
        drawn_rects['all_words_ok_rect'] = all_words_ok_rect
    elif showing_practice_end_dialog:
        practice_play_again_rect, practice_main_menu_rect, practice_quit_rect = draw_practice_end_dialog(practice_end_message)
        drawn_rects['practice_play_again_rect'] = practice_play_again_rect
        drawn_rects['practice_main_menu_rect'] = practice_main_menu_rect
        drawn_rects['practice_quit_rect'] = practice_quit_rect

    # Draw Game Over / Replay Controls / Options Menu (unchanged)
    if game_over_state and not is_batch_running:
        if final_scores is not None:
            save_rect, quit_rect, replay_rect, play_again_rect, stats_rect = draw_game_over_dialog(dialog_x, dialog_y, final_scores, reason, player_names)
            drawn_rects['save_rect'] = save_rect
            drawn_rects['quit_rect'] = quit_rect
            drawn_rects['replay_rect'] = replay_rect
            drawn_rects['play_again_rect'] = play_again_rect
            drawn_rects['stats_rect'] = stats_rect
        if showing_stats and final_scores:
            stats_ok_button_rect, stats_total_content_height = draw_stats_dialog(stats_dialog_x, stats_dialog_y, player_names, final_scores, tiles, stats_scroll_offset)
            drawn_rects['stats_ok_button_rect'] = stats_ok_button_rect
            drawn_rects['stats_total_content_height'] = stats_total_content_height
        options_rect_base, dropdown_rects_base = draw_options_menu(turn_to_display, dropdown_open, bag_count)
        drawn_rects['options_rect_base'] = options_rect_base
        drawn_rects['dropdown_rects_base'] = dropdown_rects_base
    elif replay_mode:
        replay_start_rect = state['replay_start_rect']
        replay_prev_rect = state['replay_prev_rect']
        replay_next_rect = state['replay_next_rect']
        replay_end_rect = state['replay_end_rect']
        replay_controls = [(replay_start_rect, "start"), (replay_prev_rect, "prev"), (replay_next_rect, "next"), (replay_end_rect, "end")]
        for rect, icon_type in replay_controls:
            hover = rect.collidepoint(pygame.mouse.get_pos())
            color = BUTTON_HOVER if hover else BUTTON_COLOR
            pygame.draw.rect(screen, color, rect)
            draw_replay_icon(screen, rect, icon_type)
        options_rect_base, dropdown_rects_base = draw_options_menu(turn_to_display, dropdown_open, bag_count)
        drawn_rects['options_rect_base'] = options_rect_base
        drawn_rects['dropdown_rects_base'] = dropdown_rects_base
    else: # Active game, not game over
        options_rect_base, dropdown_rects_base = draw_options_menu(turn_to_display, dropdown_open, bag_count)
        drawn_rects['options_rect_base'] = options_rect_base
        drawn_rects['dropdown_rects_base'] = dropdown_rects_base

    # Draw Dragged Tile Last (uses cached surfaces via draw_rack) (unchanged logic)
    if dragged_tile and drag_pos:
        player_idx_drag = dragged_tile[0]-1
        tile_val = None
        current_racks_for_drag = racks_to_display if replay_mode else racks
        if 0 <= player_idx_drag < len(current_racks_for_drag) and 0 <= dragged_tile[1] < len(current_racks_for_drag[player_idx_drag]):
            tile_val = current_racks_for_drag[player_idx_drag][dragged_tile[1]]

        if tile_val:
            center_x = drag_pos[0] - drag_offset[0]
            center_y = drag_pos[1] - drag_offset[1]
            draw_x = center_x - TILE_WIDTH // 2
            draw_y = center_y - TILE_HEIGHT // 2

            if tile_val == ' ':
                radius = TILE_WIDTH // 2 - 2
                pygame.draw.circle(screen, BLACK, (center_x, center_y), radius)
                text_surf = TILE_LETTER_CACHE['blank'].get('?')
                if text_surf:
                    text_rect = text_surf.get_rect(center=(center_x, center_y))
                    screen.blit(text_surf, text_rect)
            else:
                pygame.draw.rect(screen, GREEN, (draw_x, draw_y, TILE_WIDTH, TILE_HEIGHT))
                text_surf = TILE_LETTER_CACHE['regular'].get(tile_val)
                if text_surf:
                    text_rect = text_surf.get_rect(center=(center_x, center_y))
                    screen.blit(text_surf, text_rect)

    return drawn_rects
   







def check_and_handle_game_over(state):
    """
    Checks for game over conditions and handles the consequences.
    Updates and returns the game state dictionary.

    Args:
        state (dict): The current game state dictionary.

    Returns:
        dict: The potentially updated game state dictionary.
    """
    # Unpack necessary variables from state
    replay_mode = state['replay_mode']
    game_over_state = state['game_over_state']
    practice_mode = state['practice_mode']
    bag = state['bag']
    racks = state['racks']
    consecutive_zero_point_turns = state['consecutive_zero_point_turns']
    scores = state['scores']
    is_batch_running = state['is_batch_running']
    initial_game_config = state['initial_game_config']
    player_names = state['player_names']
    move_history = state['move_history']
    # Use current_game_initial_racks if available, else fall back to initial_racks
    current_game_initial_racks = state.get('current_game_initial_racks', state['initial_racks'])
    current_batch_game_num = state['current_batch_game_num']
    batch_results = state['batch_results']
    # Constants can be accessed globally or retrieved from state if needed
    # WINDOW_WIDTH, WINDOW_HEIGHT, DIALOG_WIDTH, DIALOG_HEIGHT

    if not replay_mode and not game_over_state and practice_mode != "eight_letter":
        game_ended = False
        reason = ""
        # Check rack existence before accessing
        rack0_exists = len(racks) > 0 and racks[0] is not None
        rack1_exists = len(racks) > 1 and racks[1] is not None
        rack0_empty = rack0_exists and not racks[0]
        rack1_empty = rack1_exists and not racks[1]

        if not bag and (rack0_empty or rack1_empty):
            game_ended = True
            reason = "Bag empty & rack empty"
        elif consecutive_zero_point_turns >= 6:
            game_ended = True
            reason = "Six Consecutive Zero-Point Turns"

        if game_ended:
            print(f"Game over triggered: {reason}")
            final_scores = calculate_final_scores(scores, racks, bag)
            state['game_over_state'] = True
            state['final_scores'] = final_scores
            state['reason'] = reason # Store the reason

            # Reset UI states
            state['exchanging'] = False
            state['hinting'] = False
            state['showing_all_words'] = False
            state['dropdown_open'] = False
            state['dragging'] = False
            state['typing'] = False
            state['selected_square'] = None
            state['specifying_rack'] = False
            state['showing_simulation_config'] = False
            state['current_r'] = None
            state['current_c'] = None
            state['dialog_x'] = (WINDOW_WIDTH - DIALOG_WIDTH) // 2
            state['dialog_y'] = (WINDOW_HEIGHT - DIALOG_HEIGHT) // 2
            state['last_played_highlight_coords'] = set()

            # Handle batch game saving
            if is_batch_running:
                batch_prefix = initial_game_config.get('batch_filename_prefix', 'UNKNOWN-BATCH')
                individual_gcg_filename = f"{batch_prefix}-GAME-{current_batch_game_num}.gcg"
                try:
                    # Ensure current_game_initial_racks is valid before saving
                    if not isinstance(current_game_initial_racks, list) or len(current_game_initial_racks) != 2:
                         print(f"  ERROR: Invalid current_game_initial_racks for GCG save: {current_game_initial_racks}. Using empty racks.")
                         gcg_initial_racks = [[], []]
                    else:
                         gcg_initial_racks = current_game_initial_racks

                    gcg_content = save_game_to_gcg(player_names, move_history, gcg_initial_racks, final_scores)
                    with open(individual_gcg_filename, "w") as f_gcg:
                        f_gcg.write(gcg_content)
                    print(f"  Saved individual game GCG: {individual_gcg_filename}")
                except Exception as e:
                    print(f"  ERROR saving individual game GCG '{individual_gcg_filename}': {e}")
                    individual_gcg_filename = "SAVE_ERROR" # Mark as error for stats

                game_stats = collect_game_stats(current_batch_game_num, player_names, final_scores, move_history, individual_gcg_filename)
                batch_results.append(game_stats)
                state['batch_results'] = batch_results # Update state
                state['running_inner'] = False # End the inner loop for this batch game

    return state # Return the modified state dictionary





# Function to Replace: handle_turn_start_updates
# REASON: Skip redundant pool quality calculation and move generation during batch mode.

def handle_turn_start_updates(state):
    """
    Handles updates needed at the start of a new turn:
    - Calculates pool quality score (skipped in batch).
    - Generates moves for the current player (skipped in batch).
    - Resets turn-specific flags.
    - Updates previous_turn.

    Args:
        state (dict): The current game state dictionary.

    Returns:
        dict: The updated state dictionary.
    """
    # Unpack necessary variables
    turn = state['turn']
    previous_turn = state['previous_turn']
    replay_mode = state['replay_mode']
    game_over_state = state['game_over_state']
    is_solving_endgame = state['is_solving_endgame']
    racks = state['racks']
    board_tile_counts = state['board_tile_counts'] # Unpack the counter
    practice_mode = state['practice_mode']
    paused_for_power_tile = state['paused_for_power_tile']
    paused_for_bingo_practice = state['paused_for_bingo_practice']
    gaddag_loading_status = state['gaddag_loading_status']
    GADDAG_STRUCTURE = state['GADDAG_STRUCTURE'] # Assume GADDAG is passed in state
    board = state['board']
    is_ai = state['is_ai']
    player_names = state['player_names']
    is_batch_running = state.get('is_batch_running', False) # Get batch status

    # Default values for outputs (read from input state)
    current_turn_pool_quality_score = state.get('current_turn_pool_quality_score', 0.0)
    all_moves = state.get('all_moves', [])
    human_played = state.get('human_played', False)
    power_tile_message_shown = state.get('power_tile_message_shown', False)
    bingo_practice_message_shown = state.get('bingo_practice_message_shown', False)

    if turn != previous_turn and not replay_mode and not game_over_state:
        if not is_solving_endgame:

            # --- MODIFICATION START: Conditional Calculation/Generation ---
            if not is_batch_running: # Only do these if NOT in batch mode
                # Calculate pool quality score
                if racks and len(racks) > turn - 1 and racks[turn - 1] is not None:
                    temp_remaining_tiles = get_remaining_tiles(racks[turn - 1], board_tile_counts)
                    temp_pool_analysis = analyze_unseen_pool(temp_remaining_tiles)
                    current_turn_pool_quality_score = temp_pool_analysis.get('quality_score', 0.0)
                else:
                    current_turn_pool_quality_score = 0.0

                # Generate moves (if not practice mode needing deferred gen, and GADDAG ready)
                # Needed for human hints primarily
                if practice_mode != "eight_letter" and not paused_for_power_tile and not paused_for_bingo_practice:
                    if gaddag_loading_status == 'loaded' and GADDAG_STRUCTURE:
                        if racks and len(racks) > turn - 1 and racks[turn - 1] is not None:
                            # Pass blanks correctly here as generate_all_moves still needs it
                            all_moves = generate_all_moves_gaddag(racks[turn - 1], state['tiles'], board, state['blanks'], GADDAG_STRUCTURE.root)
                            if all_moves is None:
                                all_moves = []
                        else:
                            all_moves = [] # Handle case where rack might be invalid
                    elif gaddag_loading_status == 'idle' or gaddag_loading_status == 'loading':
                        all_moves = [] # GADDAG not ready
                    else: # Error state
                        all_moves = []
            else: # In batch mode
                 # Ensure pool score is reset/defaulted if needed, though AI calculates its own
                 current_turn_pool_quality_score = 0.0
                 # Don't generate moves here, AI will do it
                 all_moves = []
            # --- MODIFICATION END ---

        # Print turn start message for human players (only if not batch)
        if not is_batch_running and 0 <= turn - 1 < len(is_ai) and not is_ai[turn - 1]:
            rack_display = ''.join(sorted(racks[turn - 1])) if racks and len(racks) > turn - 1 and racks[turn - 1] is not None else "N/A"
            print(f"Player {turn} turn started. Rack: {rack_display}")
            # Only print pool quality if calculated (i.e., not batch)
            print(f"  Current Unseen Pool Quality Score: {current_turn_pool_quality_score:.2f}")

        # Update turn tracking and reset flags
        previous_turn = turn
        human_played = False
        power_tile_message_shown = False
        bingo_practice_message_shown = False

    # Update state dictionary with new values
    state['current_turn_pool_quality_score'] = current_turn_pool_quality_score
    state['all_moves'] = all_moves
    state['previous_turn'] = previous_turn
    state['human_played'] = human_played
    state['power_tile_message_shown'] = power_tile_message_shown
    state['bingo_practice_message_shown'] = bingo_practice_message_shown

    return state




# python
# MODIFIED: Pass board_tile_counts to ai_turn
def handle_ai_turn_trigger(state):
    """
    Checks if it's the AI's turn and conditions are met, then executes the AI turn.
    If GADDAG is loading or structure is None, the turn is skipped or passed.
    Updates and returns the game state dictionary.
    Reads global gaddag_loading_status and GADDAG_STRUCTURE directly.
    Passes board_tile_counts to ai_turn.
    """
    # --- Access global directly for GADDAG status check ---
    global gaddag_loading_status, GADDAG_STRUCTURE # Need GADDAG_STRUCTURE too

    # Unpack necessary variables from state
    game_over_state = state['game_over_state']
    replay_mode = state['replay_mode']
    paused_for_power_tile = state['paused_for_power_tile']
    paused_for_bingo_practice = state['paused_for_bingo_practice']
    practice_mode = state['practice_mode']
    turn = state['turn']
    is_ai = state['is_ai']
    human_played = state['human_played']
    is_solving_endgame = state['is_solving_endgame']
    # gaddag_loading_status = state['gaddag_loading_status'] # REMOVED - Use global directly
    racks = state['racks']
    tiles = state['tiles']
    board = state['board']
    blanks = state['blanks']
    scores = state['scores']
    bag = state['bag']
    first_play = state['first_play']
    pass_count = state['pass_count']
    exchange_count = state['exchange_count']
    consecutive_zero_point_turns = state['consecutive_zero_point_turns']
    player_names = state['player_names']
    dropdown_open = state['dropdown_open']
    hinting = state['hinting']
    showing_all_words = state['showing_all_words']
    letter_checks = state['letter_checks']
    move_history = state['move_history']
    current_turn_pool_quality_score = state['current_turn_pool_quality_score']
    current_replay_turn = state['current_replay_turn']
    last_played_highlight_coords = state['last_played_highlight_coords']
    all_moves = state['all_moves']
    current_power_tile = state['current_power_tile']
    board_tile_counts = state['board_tile_counts'] # Unpack counter

    # --- AI Turn Trigger Logic (Moved from main) ---
    if not game_over_state and not replay_mode and not paused_for_power_tile and not paused_for_bingo_practice and practice_mode != "eight_letter" and 0 <= turn-1 < len(is_ai) and is_ai[turn-1] and not human_played and not is_solving_endgame:
        # *** Check GLOBAL status AND GADDAG_STRUCTURE directly ***
        if gaddag_loading_status == 'loading':
            print(f"AI {turn} waiting for GADDAG to load... Skipping turn execution.")
            return state # Return unchanged state
        # *** Check GLOBAL status AND GADDAG_STRUCTURE directly ***
        elif gaddag_loading_status == 'error' or GADDAG_STRUCTURE is None:
            status_reason = "GADDAG failed to load" if gaddag_loading_status == 'error' else "GADDAG structure is None"
            print(f"AI {turn} cannot play, {status_reason}. Passing.")
            move_rack = racks[turn-1][:] if racks and len(racks) > turn-1 else [] # Safety check
            state['consecutive_zero_point_turns'] += 1
            state['pass_count'] += 1
            state['exchange_count'] = 0
            state['move_history'].append({'player': turn, 'move_type': 'pass', 'rack': move_rack, 'score': 0, 'word': '', 'coord': '', 'blanks': set(), 'positions': [], 'drawn': [], 'is_bingo': False, 'word_with_blanks': '', 'turn_duration': 0.0, 'pool_quality_before_draw': current_turn_pool_quality_score, 'luck_factor': 0.0})
            state['current_replay_turn'] = len(state['move_history'])
            state['turn'] = 3 - turn
            state['last_played_highlight_coords'] = set()
            return state # Return updated state after forced pass
        # *** Check GLOBAL status directly ***
        elif gaddag_loading_status == 'loaded': # GADDAG_STRUCTURE is implicitly not None here
            # Expect 13 return values now
            # Note: ai_turn accesses globals or needs them passed if refactored further
            # --- MODIFICATION: Pass board_tile_counts ---
            ai_result = ai_turn(turn, racks, tiles, board, blanks, scores, bag, first_play, pass_count, exchange_count, consecutive_zero_point_turns, player_names, board_tile_counts, dropdown_open, hinting, showing_all_words, letter_checks)
            # --- END MODIFICATION ---
            if len(ai_result) == 13:
                # Unpack 13, ignore last one (empty set)
                turn_after_ai, first_play_after_ai, pass_count_after_ai, exchange_count_after_ai, consecutive_zero_after_ai, returned_moves, dropdown_open_after_ai, hinting_after_ai, showing_all_words_after_ai, paused_power_after_ai, current_power_tile_after_ai, paused_bingo_after_ai, _ = ai_result

                # Update state based on AI turn results
                state['first_play'] = first_play_after_ai
                state['pass_count'] = pass_count_after_ai
                state['exchange_count'] = exchange_count_after_ai
                state['consecutive_zero_point_turns'] = consecutive_zero_after_ai
                state['dropdown_open'] = dropdown_open_after_ai
                state['hinting'] = hinting_after_ai
                state['showing_all_words'] = showing_all_words_after_ai
                state['paused_for_power_tile'] = paused_power_after_ai
                state['current_power_tile'] = current_power_tile_after_ai
                state['paused_for_bingo_practice'] = paused_bingo_after_ai

                if turn_after_ai != turn and state['move_history']: # Check if turn changed and history exists
                     last_ai_move = state['move_history'][-1]
                     if last_ai_move.get('move_type') == 'place':
                         state['last_played_highlight_coords'] = set((pos[0], pos[1]) for pos in last_ai_move.get('positions', []))
                     else:
                         state['last_played_highlight_coords'] = set()
                state['turn'] = turn_after_ai # Update turn last

                # Update all_moves if AI turn was paused for practice
                if paused_power_after_ai or paused_bingo_after_ai:
                    state['all_moves'] = returned_moves

            else:
                print(f"Error: AI turn returned unexpected number of values: {len(ai_result)}")
                # Optionally handle error, e.g., force a pass
                # state['turn'] = 3 - turn # Force pass equivalent

    return state # Return the potentially modified state


    





# Function to Replace: handle_deferred_practice_init
# REASON: Remove the verbose debug print that runs every frame.

def handle_deferred_practice_init(state):
    """
    Handles the deferred move generation specifically for 8-Letter Bingo practice
    once the GADDAG is loaded. Updates and returns the game state.
    Reads global gaddag_loading_status and GADDAG_STRUCTURE directly.
    Removed reference to batch_stop_requested.
    Removed verbose debug print.
    """
    # --- Access global directly for GADDAG status check ---
    global gaddag_loading_status, GADDAG_STRUCTURE # Need GADDAG_STRUCTURE too

    # Unpack necessary variables
    practice_mode = state['practice_mode']
    practice_target_moves = state['practice_target_moves']
    # gaddag_loading_status = state['gaddag_loading_status'] # REMOVED - Use global directly
    # GADDAG_STRUCTURE = state['GADDAG_STRUCTURE'] # REMOVED - Use global directly
    racks = state['racks']
    tiles = state['tiles']
    board = state['board']
    blanks = state['blanks']
    # Variables that might be modified
    practice_best_move = state['practice_best_move']
    all_moves = state['all_moves']
    running_inner = state['running_inner']
    # batch_stop_requested = state['batch_stop_requested'] # REMOVED - No longer needed here

    # --- Deferred 8-Letter Practice Move Generation Logic (Moved from main) ---
    # *** Check GLOBAL status AND GADDAG_STRUCTURE directly ***
    # --- REMOVED VERBOSE DEBUG PRINT 1 ---
    # print(f"DEBUG handle_deferred_practice_init: practice_mode='{practice_mode}', practice_target_moves length: {len(practice_target_moves)}, gaddag_status='{gaddag_loading_status}'")

    if practice_mode == "eight_letter" and not practice_target_moves and gaddag_loading_status == 'loaded':
        # --- DEBUG PRINT 2: Conditions met, check GADDAG structure ---
        print(f"--- Main Loop: Conditions met for 8-letter practice move gen. GADDAG_STRUCTURE is None? {GADDAG_STRUCTURE is None} ---") # Keep this one

        # *** Check GADDAG_STRUCTURE is not None ***
        if GADDAG_STRUCTURE is not None and racks and racks[0] is not None:
             # --- DEBUG PRINT 3: About to generate moves ---
             print(f"  DEBUG: Calling generate_all_moves_gaddag for rack: {''.join(sorted(racks[0]))}") # Keep this one
             # --- DEBUG PRINT 3.5: Confirming IF block entered ---
             print(f"  DEBUG: *** ENTERED GENERATION BLOCK ***") # Keep this one
             # Generate moves into a temporary variable first
             generated_moves = generate_all_moves_gaddag(racks[0], tiles, board, blanks, GADDAG_STRUCTURE.root)

             # --- DEBUG PRINT 4: Result of generation ---
             if generated_moves is None:
                 print("  DEBUG: generate_all_moves_gaddag returned None!") # Keep this one
                 generated_moves = [] # Treat None as empty list
             else:
                 print(f"  DEBUG: generate_all_moves_gaddag returned {len(generated_moves)} moves.") # Keep this one

             if generated_moves:
                 # Update state variables
                 state['practice_target_moves'] = generated_moves
                 state['practice_best_move'] = generated_moves[0] # Assumes sorted by score desc
                 state['all_moves'] = generated_moves # Also store in all_moves for consistency
                 print(f"  DEBUG: Successfully set practice_target_moves. Length is now: {len(state['practice_target_moves'])}. Best move: {state['practice_best_move']['word']} ({state['practice_best_move']['score']} pts)") # Keep this one
             else:
                 print("Error: No valid moves found for 8-letter practice setup (deferred gen)!")
                 show_message_dialog("Error: No possible moves found for this setup.", "Practice Error")
                 state['running_inner'] = False # Update state dict
                 # state['batch_stop_requested'] = True # REMOVED - Let main handle outer stop
        # *** Handle case where GADDAG_STRUCTURE is None even if status is 'loaded' ***
        elif GADDAG_STRUCTURE is None:
            print("Error: Cannot generate practice moves, GADDAG structure is missing (status was 'loaded').")
            show_message_dialog("Error: AI data (GADDAG) structure missing.", "Error")
            state['running_inner'] = False # Update state dict
            # state['batch_stop_requested'] = True # REMOVED - Let main handle outer stop
        else: # Handle invalid rack state
            print("Error: Invalid rack state for Player 1 in 8-letter practice (deferred gen).")
            state['running_inner'] = False # Update state dict
            # state['batch_stop_requested'] = True # REMOVED - Let main handle outer stop
    # *** Check GLOBAL status directly ***
    elif practice_mode == "eight_letter" and not practice_target_moves and gaddag_loading_status == 'loading':
         print("--- Main Loop: Waiting for GADDAG to load for 8-letter practice... ---") # Keep this one


    return state # Return the modified state dictionary






def handle_practice_restart(state):
    """
    Handles the logic for restarting the 8-Letter Bingo practice mode.
    Updates and returns the game state dictionary.

    Args:
        state (dict): The current game state dictionary.

    Returns:
        dict: The potentially updated state dictionary.
    """
    # Unpack necessary flags and variables
    restart_practice_mode = state['restart_practice_mode']
    running_inner = state['running_inner']
    return_to_mode_selection = state['return_to_mode_selection']

    if restart_practice_mode:
        print("--- Restarting 8-Letter Bingo Practice ---")
        # Call the setup function for the practice mode
        proceed, p_board, p_tiles, p_racks, p_blanks, p_bag = eight_letter_practice()

        if proceed:
            # Reset game state variables for the new puzzle using data from setup
            state['board'] = p_board
            state['tiles'] = p_tiles
            state['racks'] = p_racks
            state['blanks'] = p_blanks
            state['bag'] = p_bag
            state['scores'] = [0, 0]
            state['turn'] = 1
            state['first_play'] = False # 8-letter practice never starts on first play

            # Reset practice-specific state
            state['practice_target_moves'] = []
            state['practice_best_move'] = None
            state['all_moves'] = [] # Clear general moves as well
            state['practice_solved'] = False
            state['showing_practice_end_dialog'] = False

            # Reset other relevant UI/game flow state variables
            state['word_positions'] = []
            state['exchanging'] = False
            state['hinting'] = False
            state['showing_all_words'] = False
            state['selected_tiles'] = set()
            state['typing'] = False
            state['selected_square'] = None
            state['original_tiles'] = None
            state['original_rack'] = None
            state['dragged_tile'] = None
            state['drag_pos'] = None
            state['last_played_highlight_coords'] = set()
            state['current_r'] = None # Reset cursor state
            state['current_c'] = None
            state['typing_direction'] = None
            state['typing_start'] = None
            state['pass_count'] = 0
            state['exchange_count'] = 0
            state['consecutive_zero_point_turns'] = 0
            state['previous_turn'] = 0 # Reset previous turn tracking

            print("--- 8-Letter Bingo Practice Restarted Successfully ---")
        else:
            # Handle failure to restart (e.g., user cancelled setup)
            print("--- Error restarting 8-Letter Bingo Practice. Returning to menu. ---")
            state['running_inner'] = False
            state['return_to_mode_selection'] = True

        # Reset the flag regardless of success/failure
        state['restart_practice_mode'] = False

    return state # Return the modified state





def handle_practice_messages(state):
    """
    Displays messages specific to practice modes when paused.
    Updates and returns the game state dictionary.

    Args:
        state (dict): The current game state dictionary.

    Returns:
        dict: The potentially updated state dictionary.
    """
    # Unpack necessary variables
    paused_for_power_tile = state['paused_for_power_tile']
    power_tile_message_shown = state['power_tile_message_shown']
    paused_for_bingo_practice = state['paused_for_bingo_practice']
    bingo_practice_message_shown = state['bingo_practice_message_shown']
    player_names = state['player_names']
    turn = state['turn']
    current_power_tile = state['current_power_tile']

    if paused_for_power_tile and not power_tile_message_shown:
        player_name = player_names[turn-1] if player_names[turn-1] else f"Player {turn}"
        show_message_dialog(f"A {current_power_tile} is on {player_name}'s rack. Find the highest scoring play using {current_power_tile} (matching selected lengths).", "Power Tile Practice")
        state['power_tile_message_shown'] = True # Update the state directly
    elif paused_for_bingo_practice and not bingo_practice_message_shown:
        player_name = player_names[turn-1] if player_names[turn-1] else f"Player {turn}"
        show_message_dialog(f"A bingo is playable on {player_name}'s rack. Find the highest scoring bingo.", "Bingo, Bango, Bongo!")
        state['bingo_practice_message_shown'] = True # Update the state directly

    return state # Return the modified state






def update_preview_score(state):
    """
    Calculates the preview score based on the current typing state.
    """
    # ... (unpacking) ...
    typing = state.get('typing', False)
    preview_score_enabled = state.get('preview_score_enabled', False)
    word_positions = state.get('word_positions', [])
    board = state.get('board')
    tiles = state.get('tiles')
    blanks = state.get('blanks')

    preview_score = 0
    if typing and preview_score_enabled and word_positions and board and tiles:
        if blanks is None:
            print("Warning: Blanks set is None during preview score calculation.")
            blanks_to_pass = set()
        else:
            blanks_to_pass = blanks
        try:
            # --- Use the imported Cython version ---
            preview_score = calculate_score_cython(word_positions, board, tiles, blanks_to_pass)
            # --- End Use Cython version ---
        except Exception as e:
            print(f"Error calculating preview score: {e}")
            preview_score = 0

    return preview_score





# python
def reset_per_game_variables():
    """
    Resets common UI state, turn counters, and temporary variables
    to their default values before the start of each game iteration.
    Removes batch_stop_requested key.

    Returns:
        dict: A dictionary containing the reset variables and their initial values.
    """
    # Access constants needed for initialization
    # (Ensure these are accessible, e.g., defined globally)
    # WINDOW_WIDTH, WINDOW_HEIGHT, DIALOG_WIDTH, DIALOG_HEIGHT,
    # DEFAULT_AI_CANDIDATES, DEFAULT_OPPONENT_SIMULATIONS, DEFAULT_POST_SIM_CANDIDATES

    reset_values = {
        'word_positions': [],
        'running_inner': True,
        'dropdown_open': False,
        'return_to_mode_selection': False, # Keep this for explicit user actions
        'exchanging': False,
        'hinting': False,
        'showing_all_words': False,
        'selected_tiles': set(),
        'typing': False,
        'typing_start': None,
        'typing_direction': None,
        'current_r': None,
        'current_c': None,
        'last_left_click_time': 0,
        'last_left_click_pos': None,
        'hint_moves': [],
        'selected_hint_index': None,
        'scroll_offset': 0,
        'last_clicked_pos': None, # Consider removing if truly redundant later
        'last_word': "",
        'last_score': 0,
        'last_start': None,
        'last_direction': None,
        'human_played': False,
        'dragged_tile': None,
        'drag_pos': None,
        'drag_offset': (0, 0),
        'selected_square': None,
        'original_tiles': None,
        'original_rack': None,
        'previous_turn': 0,
        'game_over_state': False,
        'showing_stats': False,
        'dialog_x': (WINDOW_WIDTH - DIALOG_WIDTH) // 2,
        'dialog_y': (WINDOW_HEIGHT - DIALOG_HEIGHT) // 2,
        'dragging': False,
        'reason': "",
        'action': None, # Consider removing if truly redundant later
        'scoreboard_height': WINDOW_HEIGHT - 80, # Recalculate based on constant
        'paused_for_power_tile': False,
        'current_power_tile': None,
        'power_tile_message_shown': False,
        'preview_score_enabled': False,
        'current_preview_score': 0,
        'stats_scroll_offset': 0,
        'stats_dialog_x': (WINDOW_WIDTH - 480) // 2, # Use stats dialog width constant
        'stats_dialog_y': (WINDOW_HEIGHT - 600) // 2, # Use stats dialog height constant
        'stats_dialog_dragging': False,
        'stats_dialog_drag_offset': (0, 0),
        'all_words_scroll_offset': 0,
        'paused_for_bingo_practice': False,
        'bingo_practice_message_shown': False,
        'current_turn_pool_quality_score': 0.0,
        'specifying_rack': False,
        'confirming_override': False,
        'specify_rack_inputs': ["", ""],
        'specify_rack_active_input': None,
        'specify_rack_original_racks': [[], []],
        'specify_rack_proposed_racks': [[], []],
        'showing_simulation_config': False,
        'simulation_config_inputs': [str(DEFAULT_AI_CANDIDATES), str(DEFAULT_OPPONENT_SIMULATIONS), str(DEFAULT_POST_SIM_CANDIDATES)],
        'simulation_config_active_input': None,
        'practice_solved': False,
        'showing_practice_end_dialog': False,
        'practice_end_message': "",
        'restart_practice_mode': False,
        'drawn_rects': {}
        # REMOVED 'batch_stop_requested'
    }
    return reset_values






# python
# MODIFIED: Initialize board_tile_counts
def reset_for_play_again(is_ai, practice_mode):
    """
    Resets the core game state variables for starting a new single game ("Play Again").
    Also returns the current global GADDAG loading status.
    Initializes board_tile_counts.

    Args:
        is_ai (list[bool]): List indicating if players are AI (for rack sorting).
        practice_mode (str or None): The current practice mode, if any.

    Returns:
        tuple or None: A tuple containing the reset state variables:
                       (board, tiles, scores, blanks, bag, racks, initial_racks,
                        current_game_initial_racks, first_play, turn, replay_mode,
                        move_history, pass_count, exchange_count,
                        consecutive_zero_point_turns, last_played_highlight_coords,
                        is_solving_endgame, practice_target_moves,
                        practice_best_move, all_moves, gaddag_loading_status,
                        board_tile_counts) # Added counter
                       Returns None if there's an error (e.g., not enough tiles).
    """
    # Access the global status to return the *current* value
    global gaddag_loading_status

    print("--- Resetting state for Play Again ---")
    # Reset game variables for a new non-batch game
    board, _, tiles = create_board()
    scores = [0, 0]
    blanks = set()
    bag = create_standard_bag()
    random.shuffle(bag)
    racks = [[], []]
    board_tile_counts = Counter() # Initialize counter

    try:
        racks[0] = [bag.pop() for _ in range(7)]
        racks[1] = [bag.pop() for _ in range(7)]
    except IndexError:
        print("Error: Not enough tiles in bag for restart.")
        return None # Indicate failure

    # Sort human racks
    for i, rack in enumerate(racks):
        # Check index validity for is_ai before accessing
        if 0 <= i < len(is_ai) and not is_ai[i]:
            rack.sort()

    initial_racks = [r[:] for r in racks]
    current_game_initial_racks = [r[:] for r in racks] # Also reset this copy
    first_play = True
    turn = 1
    replay_mode = False
    move_history = []
    pass_count = 0
    exchange_count = 0
    consecutive_zero_point_turns = 0
    last_played_highlight_coords = set()
    is_solving_endgame = False

    # Reset practice state variables if restarting a non-practice game
    # or if the specific practice mode requires it
    practice_target_moves = []
    practice_best_move = None
    all_moves = []
    if practice_mode: # Reset these regardless if coming back from a practice mode
        practice_target_moves = []
        practice_best_move = None
        all_moves = []

    # --- MODIFICATION: Include current global status and counter in return ---
    return (board, tiles, scores, blanks, bag, racks, initial_racks,
            current_game_initial_racks, first_play, turn, replay_mode,
            move_history, pass_count, exchange_count,
            consecutive_zero_point_turns, last_played_highlight_coords,
            is_solving_endgame, practice_target_moves,
            practice_best_move, all_moves, gaddag_loading_status,
            board_tile_counts) # Added counter






# ADDED: Function to contain the main game loop for profiling
def run_game_loop():
    """Runs the main game loop, handling restarts."""
    global main_called # Need to modify the global flag

    print("--- Script execution started (run_game_loop) ---")
    main_called = False # Initialize ONCE globally before the loop
    running = True
    while running: # Loop for "Play Again" / Return to Main Menu
        # Pass the current initialization status to main
        should_restart = main(main_called)

        if should_restart:
            # If main returns True, it means "Play Again" or "Main Menu" was selected.
            # Reset main_called to False ONLY if returning to menu/restarting.
            # This forces the mode selection screen to show again.
            main_called = False
            print("--- Restarting main loop for new game/mode selection ---")
        else:
            # If main returns False, it means Quit was selected.
            running = False # Exit the loop

    print("--- Script exiting (run_game_loop) ---")
    pygame.quit()
    sys.exit()







# Function to Replace: main (Complete)
# REASON: Skip event processing calls when not visualizing batch mode for performance.

def main(is_initialized): # Accept initialization status as argument
    # Declare key modified globals ---
    # (Keep all existing global declarations)
    global turn, previous_turn, game_over_state, final_scores, running_inner # Removed batch_stop_requested from globals
    global return_to_mode_selection, human_played, pass_count, exchange_count, consecutive_zero_point_turns
    global dropdown_open, exchanging, hinting, showing_all_words, selected_tiles, typing, selected_square
    global word_positions, original_tiles, original_rack, dragged_tile, drag_pos, dragging, scroll_offset
    global last_played_highlight_coords, current_replay_turn, showing_stats, dialog_x, dialog_y
    global paused_for_power_tile, current_power_tile, power_tile_message_shown
    global practice_solved # Keep this
    global showing_practice_end_dialog, practice_end_message
    global bag, last_word, last_score, last_start, last_direction, move_history, replay_mode, game_mode, is_ai, practice_mode, board, tiles, racks, blanks, scores, GADDAG_STRUCTURE
    global is_loaded_game, replay_initial_shuffled_bag, initial_racks
    global number_checks
    global is_solving_endgame
    global USE_ENDGAME_SOLVER
    global USE_AI_SIMULATION
    global is_batch_running, total_batch_games, current_batch_game_num, batch_results, initial_game_config
    global stats_scroll_offset, stats_dialog_x, stats_dialog_y, stats_dialog_dragging, stats_dialog_drag_offset
    global drag_offset
    global all_words_scroll_offset
    global paused_for_bingo_practice, bingo_practice_message_shown
    global current_turn_pool_quality_score
    global specifying_rack, confirming_override, specify_rack_inputs
    global specify_rack_active_input, specify_rack_original_racks, specify_rack_proposed_racks
    # global preview_checkbox_rect # REMOVED global declaration if it exists
    global preview_score_enabled, current_preview_score # Keep these state vars
    global all_moves
    global showing_simulation_config, simulation_config_inputs, simulation_config_active_input
    global letter_checks
    global practice_target_moves, practice_best_move
    global player_names
    global hint_moves, selected_hint_index
    global gaddag_loading_status # Keep global declaration
    global pyperclip_available, pyperclip # Add pyperclip globals
    # Add replay button rect globals needed in event handler
    global replay_start_rect, replay_prev_rect, replay_next_rect, replay_end_rect
    # Add cursor state globals
    global current_r, current_c, typing_direction, typing_start
    # Add game over reason global
    global reason
    # Add restart practice flag global
    global restart_practice_mode
    # Add last click time/pos globals
    global last_left_click_time, last_left_click_pos
    # --- ADD board_tile_counts global ---
    global board_tile_counts
    # --- ADD clock global ---
    global clock
    # --- ADD Mode Constants ---
    global MODE_HVH, MODE_HVA, MODE_AVA

    # --- Line Count Before This Change: 220 ---

    print("--- main() function entered ---")

    # --- Initialization Block (Runs Once per script execution) ---
    if not is_initialized:
        # --- ADD clock initialization ---
        clock = pygame.time.Clock() # Initialize clock here
        # --- End clock initialization ---
        print("--- main(): 'if not is_initialized' block entered ---")
        print("--- main(): Calling mode_selection_screen()... ---")
        selected_mode_result, return_data = mode_selection_screen()
        print(f"--- main(): mode_selection_screen() returned: mode={selected_mode_result} ---")

        # --- Call the new initialization function ---
        init_result = initialize_game(selected_mode_result, return_data, False)
        if init_result is None:
            print("--- main(): Initialization failed. Exiting. ---")
            return False # Signal exit

        # --- Unpack the returned state variables ---
        # --- MODIFICATION: Added visualize_batch to unpacking ---
        (game_mode, is_loaded_game, player_names, move_history, final_scores,
         replay_initial_shuffled_bag, board, tiles, scores, blanks, racks, bag,
         replay_mode, current_replay_turn, practice_mode, is_ai, human_player,
         first_play, initial_racks, number_checks, USE_ENDGAME_SOLVER,
         USE_AI_SIMULATION, is_batch_running, total_batch_games,
         current_batch_game_num, batch_results, initial_game_config,
         GADDAG_STRUCTURE, practice_target_moves, practice_best_move, all_moves,
         letter_checks, turn, pass_count, exchange_count, consecutive_zero_point_turns,
         last_played_highlight_coords, is_solving_endgame, gaddag_loading_status,
         board_tile_counts, visualize_batch) = init_result # Added visualize_batch here
    # --- End of Initialization Block ---


    # --- Outer Batch Loop ---
    batch_stop_requested = False # Initialize here for the outer loop scope
    num_loops = total_batch_games if is_batch_running else 1
    # Ensure current_game_initial_racks exists if needed later
    # Use initial_racks from the most recent initialization/reset
    current_game_initial_racks = [r[:] for r in initial_racks] if 'initial_racks' in locals() and initial_racks is not None else [[], []]
    current_state = {}
    return_to_mode_selection = False
    was_batch_running = is_batch_running

    for game_num in range(1, num_loops + 1):
        # --- Batch Game Reset Logic ---
        if is_batch_running:
            current_batch_game_num = game_num
            print(f"\n--- Starting Batch Game {current_batch_game_num} of {total_batch_games} ---")
            if game_num > 1: # Reset state for games after the first
                reset_result = reset_game_state(initial_game_config)
                if reset_result is None: print(f"FATAL: Failed to reset state for game {game_num}. Stopping batch."); batch_stop_requested = True; break # Set flag and break
                # --- MODIFICATION: Unpack board_tile_counts ---
                (board, tiles, racks, blanks, scores, turn, first_play, bag, move_history, pass_count, exchange_count, consecutive_zero_point_turns, last_played_highlight_coords, is_solving_endgame, board_tile_counts) = reset_result # Unpack counter
                is_ai = initial_game_config.get('is_ai', [False, False]); player_names = initial_game_config.get('player_names', ["P1", "P2"])
                current_game_initial_racks = [r[:] for r in racks] # Capture new initial racks for this game
                all_moves = [] # Reset all_moves for new batch game
                current_replay_turn = 0 # Reset replay turn for new batch game
                # --- Retrieve visualize_batch from config ---
                visualize_batch = initial_game_config.get('visualize_batch', False)
        # --- Non-Batch Game Reset Logic (for "Play Again") ---
        elif is_initialized and not is_batch_running: # This handles "Play Again"
            reset_result = reset_for_play_again(is_ai, practice_mode)
            if reset_result is None:
                print("Error resetting game for Play Again. Exiting.")
                return False # Signal exit
            # --- MODIFICATION: Unpack board_tile_counts ---
            (board, tiles, scores, blanks, bag, racks, initial_racks,
             current_game_initial_racks, first_play, turn, replay_mode,
             move_history, pass_count, exchange_count,
             consecutive_zero_point_turns, last_played_highlight_coords,
             is_solving_endgame, practice_target_moves,
             practice_best_move, all_moves, gaddag_loading_status,
             board_tile_counts) = reset_result
            current_replay_turn = 0
            visualize_batch = False # Not applicable in non-batch


        # --- Reset Common Per-Game Variables & Initialize State Dictionary ---
        reset_vars = reset_per_game_variables()
        # Create the state dictionary for this game iteration
        current_state = {
            # Core Game State (from initialization or reset)
            'turn': turn, 'game_over_state': False, 'final_scores': None,
            'pass_count': pass_count, 'exchange_count': exchange_count,
            'consecutive_zero_point_turns': consecutive_zero_point_turns,
            'bag': bag, 'move_history': move_history, 'replay_mode': replay_mode,
            'game_mode': game_mode, 'is_ai': is_ai, 'practice_mode': practice_mode,
            'board': board, 'tiles': tiles, 'racks': racks, 'blanks': blanks, 'scores': scores,
            'GADDAG_STRUCTURE': GADDAG_STRUCTURE, 'is_loaded_game': is_loaded_game,
            'replay_initial_shuffled_bag': replay_initial_shuffled_bag,
            'initial_racks': initial_racks, 'number_checks': number_checks,
            'is_solving_endgame': is_solving_endgame, 'USE_ENDGAME_SOLVER': USE_ENDGAME_SOLVER,
            'USE_AI_SIMULATION': USE_AI_SIMULATION, 'is_batch_running': is_batch_running,
            'total_batch_games': total_batch_games, 'current_batch_game_num': current_batch_game_num,
            'batch_results': batch_results, 'initial_game_config': initial_game_config,
            'letter_checks': letter_checks, 'practice_target_moves': practice_target_moves,
            'practice_best_move': practice_best_move, 'player_names': player_names,
            'gaddag_loading_status': gaddag_loading_status, 'first_play': first_play,
            'last_played_highlight_coords': last_played_highlight_coords,
            'current_game_initial_racks': current_game_initial_racks,
            'all_moves': all_moves, # Explicitly add all_moves
            'current_replay_turn': current_replay_turn, # *** Explicitly add current_replay_turn ***
            'board_tile_counts': board_tile_counts, # Add the counter
            'visualize_batch': visualize_batch, # Add the setting
            # UI / Temporary State (from reset_vars)
            **reset_vars, # Unpack the reset dictionary
            # Add other necessary state not covered by reset_vars
            'pyperclip_available': pyperclip_available, 'pyperclip': pyperclip,
            'replay_start_rect': replay_start_rect, 'replay_prev_rect': replay_prev_rect,
            'replay_next_rect': replay_next_rect, 'replay_end_rect': replay_end_rect,
            'bag_count': len(bag) # Calculate initial bag count
            # REMOVED 'batch_stop_requested' from here
        }
        print("--- main(): Initialized current_state dictionary. ---")
        # --- End Common Variable Resets & State Dictionary Init ---


        # --- Inner Game Loop ---
        # Use the running_inner flag from the state dictionary
        while current_state['running_inner']:
            # --- MODIFICATION: Check outer flag directly ---
            if batch_stop_requested: # Check the outer flag
                print(f"--- DEBUG: Inner loop start check: batch_stop_requested is TRUE. Breaking inner loop. ---") # DEBUG
                current_state['running_inner'] = False # Stop inner loop if outer flag is set
                break
            # else: # Optional debug
            #    print(f"--- DEBUG: Inner loop start check: batch_stop_requested is FALSE. Continuing inner loop. ---") # DEBUG


            # --- Deferred 8-Letter Practice Move Generation ---
            current_state = handle_deferred_practice_init(current_state)
            if not current_state['running_inner']:
                print("--- DEBUG: Setting batch_stop_requested = True due to handle_deferred_practice_init failure. ---") # DEBUG
                batch_stop_requested = True # Ensure outer loop stops if init fails
                break # Exit if init failed

            # --- Game State Update / Turn Logic ---
            # --- MODIFICATION: Only call update_preview_score when needed ---
            if current_state['typing'] and current_state['preview_score_enabled']:
                current_state['current_preview_score'] = update_preview_score(current_state)
            else:
                # Ensure score is 0 if not typing or preview disabled
                current_state['current_preview_score'] = 0
            # --- END MODIFICATION ---

            current_state = handle_turn_start_updates(current_state)
            current_state = handle_ai_turn_trigger(current_state)

            # --- Add Delay if AI turn skipped due to loading ---
            if current_state['gaddag_loading_status'] == 'loading' and 0 <= current_state['turn'] - 1 < len(current_state['is_ai']) and current_state['is_ai'][current_state['turn'] - 1]:
                pygame.time.wait(100) # Wait 100ms to yield CPU

            # --- Practice Mode Message Display ---
            current_state = handle_practice_messages(current_state)

            # --- Determine if drawing/event processing should happen ---
            should_process_events_and_draw = True # Default to True
            if current_state['is_batch_running'] and not current_state['visualize_batch']:
                should_process_events_and_draw = False # Skip if non-visual batch

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%%%% EVENT PROCESSING %%%%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # --- MODIFICATION START: Conditional Event Processing ---
            if should_process_events_and_draw:
                event_result_state = process_game_events(current_state, current_state['drawn_rects'])
                # Update state dictionary based on event processing results
                current_state.update(event_result_state)
                # Update local control flags based on the returned state
                return_to_mode_selection = current_state['return_to_mode_selection']
                # --- MODIFICATION: Update outer flag based on return_to_mode_selection ---
                if return_to_mode_selection:
                    # --- ADD DEBUG PRINT ---
                    print(f"--- DEBUG: Setting batch_stop_requested = True because return_to_mode_selection is TRUE (event: {pygame.event.event_name(event.type) if 'event' in locals() else 'N/A'}). ---")
                    batch_stop_requested = True # Set outer flag if user explicitly wants to stop/go back
            # --- MODIFICATION END ---
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%% END EVENT PROCESSING %%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


            # --- Game Over Check ---
            current_state = check_and_handle_game_over(current_state)
            # Update local control flag if game ended in batch mode
            if current_state['game_over_state'] and current_state['is_batch_running']:
                current_state['running_inner'] = False


            # --- Practice Mode Restart Logic ---
            # --- MODIFICATION: Explicitly prevent restart check in batch ---
            if not current_state['is_batch_running']:
                current_state = handle_practice_restart(current_state)
                # --- REMOVED check and setting of batch_stop_requested ---
            # --- END MODIFICATION ---


            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%%%%%% DRAWING CALL %%%%%%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # --- MODIFICATION START: Conditional Drawing ---
            if should_process_events_and_draw: # Use the same flag
                current_state['drawn_rects'] = draw_game_screen(screen, current_state)
                pygame.display.flip()
            # --- MODIFICATION END ---
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%%%% END DRAWING CALL %%%%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # --- Clock Tick (Conditional) ---
            if not current_state['is_batch_running'] and current_state['game_mode'] != MODE_AVA:
                clock.tick(60) # Limit frame rate only if interactive and not AI vs AI
            # --- End Clock Tick ---


        # --- End of Inner Game Loop ---

        # --- ADD DEBUG PRINT ---
        print(f"--- DEBUG: End of outer loop iteration for game {game_num}. batch_stop_requested = {batch_stop_requested} ---")

        # *** MOVED BREAK CHECK HERE ***
        if batch_stop_requested:
            print("--- Batch run stopping due to user request or error ---")
            break # Exit the outer 'for game_num...' loop

    # --- End of Outer Batch Loop ---

    # --- Save batch stats after loop finishes ---
    # *** Use the initial was_batch_running flag and check current_state exists ***
    if was_batch_running and current_state and current_state.get('batch_results'):
        batch_summary_filename = f"{current_state.get('initial_game_config', {}).get('batch_filename_prefix', 'UNKNOWN-BATCH')}.txt"
        save_batch_statistics(current_state['batch_results'], current_state.get('player_names', ["P1","P2"]), batch_summary_filename)
        # *** Set return_to_mode_selection to True after batch save ***
        return_to_mode_selection = True # Ensure we go back to menu

    print("--- main(): Exited main game loop(s). ---")

    # --- MODIFIED FINAL RETURN LOGIC ---
    # If we started in batch mode, always return True to go back to menu
    if was_batch_running:
        print("--- main(): Batch run finished. Returning to mode selection... ---")
        return True # Signal restart

    # Otherwise (not batch mode), use the flag set by events
    elif return_to_mode_selection:
        print("--- main(): Returning to mode selection... ---")
        return True # Signal restart
    else:
        print("--- main(): Quitting pygame and exiting. ---")
        return False # Signal exit

# --- Line Count After This Change: 228 ---







# MODIFIED: Program Entry Point for Profiling
if __name__ == "__main__":
    import cProfile
    import pstats
    import io # Needed for pstats output redirection if desired

    profile_filename = 'scrabble_profile.prof'
    print(f"--- Starting Profiling: Output will be saved to {profile_filename} ---")

    # Profile the main game loop function
    cProfile.run('run_game_loop()', filename=profile_filename)

    print(f"--- Profiling Complete: Data saved to {profile_filename} ---")
    print("\nTo visualize the results, install snakeviz (`pip install snakeviz`)")
    print(f"Then run the following command in your terminal:")
    print(f"  snakeviz {profile_filename}")

    # Optional: Print basic stats to console
    print("\n--- Basic Profile Stats ---")
    p = pstats.Stats(profile_filename)
    p.strip_dirs().sort_stats('cumulative').print_stats(20) # Print top 20 cumulative time functions
