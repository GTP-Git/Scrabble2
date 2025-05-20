


#python
#Scrabble 06MAY25 Cython V5


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
from itertools import permutations, product, combinations
from collections import Counter
import copy
import threading
import array
import numpy as np
import cProfile
import pstats
import io



# --- scrabble_helpers import ---
from scrabble_helpers import (
    get_coord, # REMOVED evaluate_leave,
    get_anchor_points,
    Gaddag, GaddagNode, DAWG as DAWG_cls,
    GRID_SIZE, CENTER_SQUARE, TILE_DISTRIBUTION,
    RED, PINK, BLUE, LIGHT_BLUE, LETTERS,
    LEAVE_LOOKUP_TABLE # <<< Ensure LEAVE_LOOKUP_TABLE is imported/available globally
)


print("Loading DAWG dictionary...")
try:
    with open("dawg.pkl", "rb") as f_dawg:
        # Load the DAWG object into the global variable
        DAWG = pickle.load(f_dawg)
    if not isinstance(DAWG, DAWG_cls): # Optional: Check type
         print("Error: Loaded object is not a DAWG instance!")
         DAWG = None # Or handle error appropriately
    elif DAWG is None:
         print("Error: Loaded DAWG object is None!")
    else:
         print("DAWG dictionary loaded successfully.")
except FileNotFoundError:
    print("Error: dawg.pkl not found. Please ensure the dictionary file exists.")
    DAWG = None # Set to None if loading fails
    # sys.exit("DAWG file not found.") # Optional: Exit if critical
except Exception as e:
    print(f"Error loading DAWG dictionary: {e}")
    DAWG = None # Set to None if loading fails
    # sys.exit("Failed to load DAWG.") # Optional: Exit if critical

# Ensure DAWG is not None before proceeding if it's critical
if DAWG is None:
     print("CRITICAL ERROR: DAWG object could not be loaded. AI features requiring DAWG will fail.")
     # Consider exiting if DAWG is essential for basic operation
     # sys.exit("Exiting due to DAWG load failure.")



# --- Consolidated Cython Import Block ---
try:
    from gaddag_cython import _gaddag_traverse as _gaddag_traverse_cython
    from gaddag_cython import calculate_score as calculate_score_cython
    from gaddag_cython import is_valid_play as is_valid_play_cython
    from gaddag_cython import find_all_words_formed as find_all_words_formed_cython
    from gaddag_cython import compute_cross_checks_cython
    from gaddag_cython import evaluate_leave_cython
    from gaddag_cython import generate_all_moves_gaddag_cython
    from gaddag_cython import evaluate_single_move_cython
    from gaddag_cython import standard_evaluation_cython
    from gaddag_cython import ai_turn_logic_cython
    from gaddag_cython import calculate_luck_factor_cython
    from gaddag_cython import get_expected_draw_value_cython

    print("--- SUCCESS: Imported ALL Cython functions. ---")

    USE_CYTHON_GADDAG = True
    USE_CYTHON_CROSS_CHECKS = True
    USE_CYTHON_EVALUATE_LEAVE = True
    USE_CYTHON_MOVE_GENERATION = True # Use a more descriptive name
    USE_CYTHON_EVALUATE_SINGLE_MOVE = True
    USE_CYTHON_STANDARD_EVALUATION = True
    USE_CYTHON_AI_TURN_LOGIC = True
    USE_CYTHON_LUCK_FACTOR = True
    USE_CYTHON_EXPECTED_DRAW = True
    

    # --- Add explicit check for all imported functions ---
    print(f"    _gaddag_traverse_cython: {repr(_gaddag_traverse_cython)}")
    print(f"    calculate_score_cython: {repr(calculate_score_cython)}")
    print(f"    is_valid_play_cython: {repr(is_valid_play_cython)}")
    print(f"    find_all_words_formed_cython: {repr(find_all_words_formed_cython)}")
    print(f"    compute_cross_checks_cython: {repr(compute_cross_checks_cython)}")
    print(f"    evaluate_leave_cython: {repr(evaluate_leave_cython)}")
    print(f"    generate_all_moves_gaddag_cython: {repr(generate_all_moves_gaddag_cython)}")
    print(f"    evaluate_single_move_cython: {repr(evaluate_single_move_cython)}")
    print(f"    standard_evaluation_cython: {repr(standard_evaluation_cython)}")
    print(f"    ai_turn_logic_cython: {repr(ai_turn_logic_cython)}")
    print(f"    calculate_luck_factor_cython: {repr(calculate_luck_factor_cython)}")
    print(f"    get_expected_draw_value_cython: {repr(get_expected_draw_value_cython)}")
    

except ImportError as e:
    print(f"--- FAILURE: Cython import failed: {e} ---")
    print("!!! CRITICAL ERROR: Falling back to dummy Python functions. Performance will be impacted. !!!")

    # Define dummy functions for ALL Cython imports
    def _gaddag_traverse_dummy(*args, **kwargs): print("DUMMY _gaddag_traverse called"); return None
    def calculate_score_dummy(*args, **kwargs): print("DUMMY calculate_score called"); return 0
    def is_valid_play_dummy(*args, **kwargs): print("DUMMY is_valid_play called"); return False, False
    def find_all_words_formed_dummy(*args, **kwargs): print("DUMMY find_all_words_formed called"); return []
    def compute_cross_checks_dummy(tiles, dawg_obj):
         print("!!! Using dummy Python compute_cross_checks !!!")
         # ... (fallback logic) ...
         return {}
    def evaluate_leave_dummy(rack, verbose=False): # Corrected dummy signature
        print("!!! Using dummy evaluate_leave - returning 0.0 !!!")
        return 0.0
    def generate_all_moves_gaddag_dummy(*args, **kwargs):
        print("!!! Using dummy generate_all_moves_gaddag - returning empty list !!!")
        return []
    def evaluate_single_move_dummy(move_dict):
        print("!!! Using dummy evaluate_single_move - returning raw score !!!")
        # Fallback logic: just return the raw score if leave eval fails
        return float(move_dict.get('score', 0.0))
    def standard_evaluation_dummy(all_moves):
        print("!!! Using dummy standard_evaluation - returning raw scores !!!")
        if all_moves is None: return []
        temp_evaluated_plays = []
        for move in all_moves:
             if isinstance(move, dict):
                 temp_evaluated_plays.append({'move': move, 'final_score': float(move.get('score', 0.0))})
        temp_evaluated_plays.sort(key=lambda x: x['final_score'], reverse=True)
        return temp_evaluated_plays
    def ai_turn_logic_dummy(all_moves, current_rack, board_tile_counts_obj,
                            blanks_played_count, bag_count,
                            get_remaining_tiles_func, find_best_exchange_option_func,
                            EXCHANGE_PREFERENCE_THRESHOLD, MIN_SCORE_TO_AVOID_EXCHANGE):
        print("!!! Using dummy ai_turn_logic - defaulting to pass !!!")
        # Minimal fallback: just pass if logic fails
        return ('pass', None)
    def calculate_luck_factor_dummy(drawn_tiles, move_rack_before,
                                    board_tile_counts_obj, blanks_played_count,
                                    get_remaining_tiles_func):
        print("!!! Using dummy calculate_luck_factor - returning 0.0 !!!")
        return 0.0
    def get_expected_draw_value_dummy(current_rack, board_tile_counts_obj,
                                      blanks_played_count, get_remaining_tiles_func):
        print("!!! Using dummy get_expected_draw_value - returning 0.0 !!!")
        return 0.0
    



    # Assign ALL dummies to the original names
    _gaddag_traverse_cython = _gaddag_traverse_dummy
    calculate_score_cython = calculate_score_dummy
    is_valid_play_cython = is_valid_play_dummy
    find_all_words_formed_cython = find_all_words_formed_dummy
    compute_cross_checks_cython = compute_cross_checks_dummy
    evaluate_leave_cython = evaluate_leave_dummy
    # --- MODIFICATION: Assign new dummy, remove helper assignment ---
    generate_all_moves_gaddag_cython = generate_all_moves_gaddag_dummy
    evaluate_single_move_cython = evaluate_single_move_dummy
    standard_evaluation_cython = standard_evaluation_dummy
    ai_turn_logic_cython = ai_turn_logic_dummy
    calculate_luck_factor_cython = calculate_luck_factor_dummy
    get_expected_draw_value_cython = get_expected_draw_value_dummy
    



    # Set ALL flags to False
    USE_CYTHON_GADDAG = False
    USE_CYTHON_CROSS_CHECKS = False
    USE_CYTHON_EVALUATE_LEAVE = False
    USE_CYTHON_MOVE_GENERATION = False
    USE_CYTHON_EVALUATE_SINGLE_MOVE = False
    USE_CYTHON_STANDARD_EVALUATION = False
    USE_CYTHON_AI_TURN_LOGIC = False
    USE_CYTHON_LUCK_FACTOR = False
    USE_CYTHON_EXPECTED_DRAW = False
    

    # Optionally, exit here if fallbacks are unacceptable
    # sys.exit("Essential Cython modules failed to load.")
# --- End Consolidated Cython Import Block ---




print(f"DEBUG: is_valid_play_cython is assigned to: {is_valid_play_cython}")




# --- pyperclip import block starts below ---




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
DEVELOPER_PROFILE_ENABLED = False # Global flag for cProfile toggle


STATS_LABEL_X_OFFSET = 10
STATS_P1_VAL_X_OFFSET = 160 # Increased from 120
STATS_P2_VAL_X_OFFSET = 270 # Increased from 230



LEAVE_LOOKUP_TABLE = {}

EXCHANGE_PREFERENCE_THRESHOLD = 1.0
MIN_SCORE_TO_AVOID_EXCHANGE = 15
#POOL_QUALITY_FACTOR = 1.5
#POOL_TILE_VALUES = {' ': 5, 'S': 3, 'Z': 1, 'X': 1, 'Q': -3, 'J': -2, 'V': -2, 'W': -2, 'U': -2, 'I': -1}

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
    'regular': {},        # For standard tiles (black text)
    'blank': {},          # For blank tiles on rack (white '?')
    'blank_assigned': {} # <<< ADDED: For assigned blanks on board (white letter)
}

print("Pre-rendering tile letters...")
# Pre-render regular tile letters (A-Z) - Black Text
for i in range(26):
    letter = chr(ord('A') + i)
    text_surf = font.render(letter, True, BLACK) # Black text
    TILE_LETTER_CACHE['regular'][letter] = text_surf

# Pre-render blank tile symbol (?) for rack - White Text
blank_surf = font.render('?', True, WHITE)
TILE_LETTER_CACHE['blank']['?'] = blank_surf

# <<< ADDED: Pre-render assigned blank letters (A-Z) - White Text >>>
for i in range(26):
    letter = chr(ord('A') + i)
    text_surf = font.render(letter, True, WHITE) # White text
    TILE_LETTER_CACHE['blank_assigned'][letter] = text_surf
# <<< END ADDED >>>

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
final_scores = None
is_loaded_game = False
game_over_state = False
restart_practice_mode = False
return_to_mode_selection = False
replay_initial_shuffled_bag = None
turn = 1
first_play = True
game_mode = None
is_ai = None
practice_mode = None # Added to track practice modes like "eight_letter", "power_tiles"
move_history = []
replay_mode = False
current_replay_turn = 0
human_player = 1
last_word = ""
last_score = 0
last_start = None
last_direction = None
last_played_highlight_coords = set()
gaddag_loading_status = 'idle' # Tracks status: 'idle', 'loading', 'loaded', 'error'
gaddag_load_thread = None # Holds the thread object
gaddag_loaded_event = None

is_solving_endgame = False # Flag to indicate AI is in endgame calculation
endgame_start_time = 0 # To track duration if needed

profiler = None # For cProfile
clock = None # For Pygame clock

is_batch_running = False
total_batch_games = 0
current_batch_game_num = 0
batch_results = []
initial_game_config = {} # Store settings for batch games
practice_probability_max_index = None


# In Scrabble Game.py

def _load_gaddag_background(loaded_event): # MODIFICATION: Accept event argument
    """Loads the GADDAG structure in a background thread."""
    global GADDAG_STRUCTURE, gaddag_loading_status, DAWG 
    
    if DAWG is None: # DAWG should be loaded by initialize_game before this thread starts
        print("Background Thread: CRITICAL - DAWG is None. GADDAG loading aborted.")
        gaddag_loading_status = 'error'
        GADDAG_STRUCTURE = None
        # MODIFICATION START: Signal completion (with error)
        if loaded_event:
            loaded_event.set()
        # MODIFICATION END
        return

    gaddag_file = "gaddag.pkl"
    print(f"Background Thread: Attempting to load GADDAG structure from {gaddag_file}...")
    try:
        with open(gaddag_file, "rb") as f_load:
            load_start_time = time.time() # Renamed from load_start to avoid conflict if it's global
            loaded_gaddag = pickle.load(f_load) 
        GADDAG_STRUCTURE = loaded_gaddag 
        gaddag_loading_status = 'loaded' # Set status AFTER successful assignment
        load_duration = time.time() - load_start_time
        print(f"Background Thread: GADDAG loaded successfully in {load_duration:.2f} seconds. Status: {gaddag_loading_status}")
    except FileNotFoundError:
        print(f"Background Thread: ERROR - GADDAG file '{gaddag_file}' not found.")
        GADDAG_STRUCTURE = None
        gaddag_loading_status = 'error'
    except Exception as e:
        print(f"Background Thread: ERROR loading GADDAG - {e}")
        GADDAG_STRUCTURE = None
        gaddag_loading_status = 'error'
    finally:
        # MODIFICATION START: Signal that loading (or attempt) is complete
        if loaded_event:
            loaded_event.set()
        # MODIFICATION END



# In Scrabble Game.py

def draw_loading_indicator(scoreboard_x, scoreboard_y, scoreboard_width):
    """Draws a 'Loading AI Data...' message."""
    # MODIFICATION START: Explicitly use global gaddag_loaded_event and gaddag_loading_status
    global gaddag_loading_status, screen, ui_font, RED, gaddag_loaded_event
    # MODIFICATION END

    show_loading_message = False
    if gaddag_loaded_event is not None:
        if not gaddag_loaded_event.is_set(): # If event exists but is not set, means loading is in progress
            show_loading_message = True
    elif gaddag_loading_status == 'loading': # Fallback: if event isn't there yet, but status says loading
        show_loading_message = True
    # If event is set, or status is 'loaded' or 'error', don't show "Loading..."
    # (Errors might be handled by other dialogs or messages if needed)

    if show_loading_message:
        loading_text = "Loading AI Data..."
        loading_surf = ui_font.render(loading_text, True, RED) 
        target_center_x = scoreboard_x + scoreboard_width // 2
        target_bottom_y = scoreboard_y - 10 
        target_top_y = max(5, target_bottom_y - loading_surf.get_height())
        loading_rect = loading_surf.get_rect(centerx=target_center_x, top=target_top_y)
        
        bg_rect = loading_rect.inflate(20, 10) 
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA) 
        bg_surf.fill((200, 200, 200, 180)) 
        screen.blit(bg_surf, bg_rect.topleft)
        screen.blit(loading_surf, loading_rect)






def draw_checkbox(screen, x, y, checked):
    pygame.draw.rect(screen, BLACK, (x, y, 20, 20), 1)
    if checked:
        pygame.draw.line(screen, BLACK, (x+2, y+2), (x+18, y+18), 2)
        pygame.draw.line(screen, BLACK, (x+18, y+2), (x+2, y+18), 2)





def save_game_to_sgs(filename, game_data_dict):
    """
    Saves comprehensive game data to a file using pickle.

    Args:
        filename (str): The name of the file to save to (should end in .sgs).
        game_data_dict (dict): A dictionary containing all game state to be saved.
                               Expected keys include:
                               - 'player_names', 'game_mode', 'is_ai', 'practice_mode',
                               - 'use_endgame_solver_setting', 'use_ai_simulation_setting',
                               - 'initial_shuffled_bag_order_sgs', 'initial_racks_sgs',
                               - 'move_history', 'final_scores',
                               - 'sgs_version', 'timestamp'
    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        with open(filename, "wb") as f_sgs:
            pickle.dump(game_data_dict, f_sgs)
        print(f"--- Game successfully saved to {filename} (SGS format) ---")
        return True
    except Exception as e:
        print(f"--- Error saving game to SGS file '{filename}': {e} ---")
        traceback.print_exc()
        return False





# (Place this function alongside other save/load utility functions like load_game_from_gcg)

def load_game_from_sgs(filename):
    """
    Loads comprehensive game data from an SGS file using pickle.

    Args:
        filename (str): The name of the .sgs file to load.

    Returns:
        dict or None: A dictionary containing the loaded game state,
                      or None if loading fails.
    """
    try:
        with open(filename, "rb") as f_sgs:
            game_data_dict = pickle.load(f_sgs)
        print(f"--- Game successfully loaded from {filename} (SGS format) ---")
        # Basic validation (can be expanded)
        if not isinstance(game_data_dict, dict):
            print(f"--- Error: Loaded SGS data from '{filename}' is not a dictionary. ---")
            return None
        if 'sgs_version' not in game_data_dict or 'move_history' not in game_data_dict:
            print(f"--- Error: Loaded SGS data from '{filename}' is missing essential keys. ---")
            return None
        return game_data_dict
    except FileNotFoundError:
        print(f"--- Error: SGS file not found '{filename}' ---")
        return None
    except pickle.UnpicklingError:
        print(f"--- Error: Could not unpickle SGS file '{filename}'. File may be corrupted or not a valid SGS file. ---")
        return None
    except Exception as e:
        print(f"--- Error loading game from SGS file '{filename}': {e} ---")
        traceback.print_exc()
        return None















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




# Function to Replace: draw_dev_tools_dialog
# REASON: Add cProfile checkbox.

def draw_dev_tools_dialog(visualize_checked, cprofile_checked): # Added cprofile_checked parameter
    """
    Draws the Developer Tools dialog box.

    Args:
        visualize_checked (bool): The current state of the "Visualize Batch Games" checkbox.
        cprofile_checked (bool): The current state of the "cProfile" checkbox.

    Returns:
        tuple: (visualize_checkbox_rect, cprofile_checkbox_rect, close_button_rect) # Added cprofile rect
    """
    dialog_width, dialog_height = 350, 220 # Increased height for new checkbox
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
    checkbox_y_visualize = dialog_y + 60
    visualize_checkbox_rect = pygame.Rect(checkbox_x, checkbox_y_visualize, 20, 20)
    draw_checkbox(screen, checkbox_x, checkbox_y_visualize, visualize_checked)
    label_text_visualize = ui_font.render("Visualize Batch Games", True, BLACK)
    screen.blit(label_text_visualize, (checkbox_x + 30, checkbox_y_visualize + 2))

    # --- ADDED: Checkbox for cProfile ---
    checkbox_y_cprofile = checkbox_y_visualize + 35 # Position below previous checkbox
    cprofile_checkbox_rect = pygame.Rect(checkbox_x, checkbox_y_cprofile, 20, 20)
    draw_checkbox(screen, checkbox_x, checkbox_y_cprofile, cprofile_checked)
    label_text_cprofile = ui_font.render("Enable cProfile on Exit", True, BLACK)
    screen.blit(label_text_cprofile, (checkbox_x + 30, checkbox_y_cprofile + 2))
    # --- END ADDED ---

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

    return visualize_checkbox_rect, cprofile_checkbox_rect, close_button_rect # Return new rect






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






# Function to Replace: eight_letter_practice
# REASON: Auto-focus probability input field.

def eight_letter_practice():
    """Handles the setup dialog and initialization for 8-Letter Bingo practice."""
    try:
        with open("7-letter-list.txt", "r") as seven_file, open("8-letter-list.txt", "r") as eight_file:
            seven_letter_words = [line.strip().upper() for line in seven_file.readlines()]; eight_letter_words = [line.strip().upper() for line in eight_file.readlines()]
    except FileNotFoundError: print("Error: Could not find 7-letter-list.txt or 8-letter-list.txt"); return False, None, None, None, None, None, None # Added None for max_index
    if not seven_letter_words or not eight_letter_words: print("Error: Word list files are empty."); return False, None, None, None, None, None, None # Added None for max_index
    dialog_width, dialog_height = 300, 150; dialog_x = (WINDOW_WIDTH - dialog_width) // 2; dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
    proceed = False;
    # --- MODIFICATION: Initialize text_box_active to True ---
    text_box_active = True
    # --- END MODIFICATION ---
    probability_input = ""
    max_index = len(eight_letter_words) # Default to all words

    while True: # Dialog loop
        screen.fill(WHITE); pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height)); pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2)
        title_text = dialog_font.render("8-Letter Bingo Options", True, BLACK); screen.blit(title_text, (dialog_x + 10, dialog_y + 10))
        prob_text = ui_font.render("Probability", True, BLACK); screen.blit(prob_text, (dialog_x + 20, dialog_y + 50))
        text_box_rect = pygame.Rect(dialog_x + 120, dialog_y + 45, 150, 30); pygame.draw.rect(screen, WHITE, text_box_rect); pygame.draw.rect(screen, BLACK, text_box_rect, 1 if not text_box_active else 2) # Highlight if active
        input_text = ui_font.render(probability_input, True, BLACK); screen.blit(input_text, (text_box_rect.x + 5, text_box_rect.y + 5))
        go_rect = pygame.Rect(dialog_x + 50, dialog_y + 100, 100, 30); cancel_rect = pygame.Rect(dialog_x + 160, dialog_y + 100, 100, 30)
        pygame.draw.rect(screen, BUTTON_COLOR, go_rect); pygame.draw.rect(screen, BUTTON_COLOR, cancel_rect)
        go_text = button_font.render("Go", True, BLACK); cancel_text = button_font.render("Cancel", True, BLACK)
        screen.blit(go_text, go_text.get_rect(center=go_rect.center)); screen.blit(cancel_text, cancel_text.get_rect(center=cancel_rect.center))

        # Draw blinking cursor if active
        if text_box_active and int(time.time() * 2) % 2 == 0:
            cursor_x = text_box_rect.x + 5 + input_text.get_width()
            cursor_y1 = text_box_rect.y + 5
            cursor_y2 = text_box_rect.bottom - 5
            pygame.draw.line(screen, BLACK, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos;
                # Update active state based on click
                text_box_active = text_box_rect.collidepoint(x, y)
                if go_rect.collidepoint(x, y):
                    # --- MODIFICATION: Calculate max_index here ---
                    max_index = len(eight_letter_words); # Default
                    if probability_input.isdigit():
                        prob_val = int(probability_input);
                        max_index = min(max(1, prob_val), len(eight_letter_words))
                    # --- END MODIFICATION ---
                    selected_eight = random.choice(eight_letter_words[:max_index]); print("Selected 8-letter word:", selected_eight)
                    remove_idx = random.randint(0, 7); removed_letter = selected_eight[remove_idx]; removed_eight = selected_eight[:remove_idx] + selected_eight[remove_idx + 1:]
                    print("Player 1 rack (7 letters):", removed_eight); print("Removed letter:", removed_letter)
                    selected_seven = select_seven_letter_word(removed_letter, seven_letter_words)
                    if selected_seven is None: print("Error: Could not find a suitable 7-letter word."); return False, None, None, None, None, None, None # Added None for max_index
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
                    if not placement_successful: print("Error: Could not place 7-letter word centered H or V."); return False, None, None, None, None, None, None # Added None for max_index
                    local_bag = [];
                    # --- MODIFICATION: Return max_index ---
                    return True, board, tiles, local_racks, local_blanks, local_bag, max_index
                    # --- END MODIFICATION ---
                elif cancel_rect.collidepoint(x, y):
                    return False, None, None, None, None, None, None # Added None for max_index
            elif event.type == pygame.KEYDOWN and text_box_active:
                if event.key == pygame.K_BACKSPACE: probability_input = probability_input[:-1]
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                     # Simulate clicking Go button on Enter
                     pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': go_rect.center, 'button': 1}))
                elif event.unicode.isdigit(): probability_input += event.unicode
        pygame.display.flip()
    # return False, None, None, None, None, None, None # Should not be reached # Added None for max_index








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
# REASON: Store max_index from eight_letter_practice. Ensure return tuple has correct length. Add debug print.
def mode_selection_screen():
    """Displays the mode selection screen and handles user input."""
    global DEVELOPER_PROFILE_ENABLED # Allow modification if cProfile is checked

    # --- Initialization ---
    try: # Load background image
        image = pygame.image.load("Scrabble_S.png").convert_alpha()
        # Scale image to fit width while maintaining aspect ratio (approx)
        # Or scale to fit available content area
        content_width = WINDOW_WIDTH - 200 # Example: leave 100px padding on each side
        image = pygame.transform.scale(image, (content_width, WINDOW_HEIGHT)) # Adjust height as needed or calc based on aspect ratio
        image.set_alpha(128) # Make it semi-transparent
        content_left = (WINDOW_WIDTH - content_width) // 2
    except pygame.error as e:
        print(f"--- mode_selection_screen(): Error loading background image: {e} ---")
        image = None
        content_width = WINDOW_WIDTH # Use full width if no image
        content_left = 0

    print("--- mode_selection_screen() entered ---")
    if image: print("--- mode_selection_screen(): Background image loaded and processed. ---")

    modes = [MODE_HVH, MODE_HVA, MODE_AVA]
    selected_mode = None
    player_names = ["Player 1", "Player 2"]
    human_player = 1 # Default for HVA
    input_active = [False, False] # Which name input is active
    current_input = 0 # Index of the currently selected mode button

    # Practice Mode State
    practice_mode = None
    dropdown_open = False
    showing_power_tiles_dialog = False
    letter_checks = [True, True, True, True] # J, Q, X, Z (Power Tiles defaults)
    number_checks = [True, True, True, True, False, False] # 2, 3, 4, 5, 6, 7+ (Power Tiles defaults)
    practice_state = None # To store pre-configured state for some practice modes
    practice_probability_max_index = None # <<< MODIFICATION: Initialize here

    # Loaded Game State
    loaded_game_data = None # Will store data loaded from SGS

    # AI Settings
    use_endgame_solver_checked = False
    use_ai_simulation_checked = False

    # State for Load Game Text Input
    showing_load_input = False
    load_filename_input = ""
    load_input_active = False
    load_confirm_button_rect = None
    load_input_rect = None
    load_cancel_button_rect = None

    # Developer Tools State
    showing_dev_tools_dialog = False
    visualize_batch_checked = False # Default to False (unchecked)
    cprofile_checked = False # Default False
    dev_tools_visualize_rect = None # Rect for checkbox click detection
    dev_tools_cprofile_rect = None # Rect for cProfile checkbox
    dev_tools_close_rect = None # Rect for close button click detection

    print("--- mode_selection_screen(): Entering main loop (while selected_mode is None:)... ---")
    loop_count = 0 # Debug counter
    while selected_mode is None:
        loop_count += 1
        # --- Define positions INSIDE the loop ---
        # This allows recalculation if window size changes (though not currently supported)
        option_rects = [] # For practice dropdown
        name_rect_x = content_left + (content_width - 200) // 2 # Center name inputs within content area

        # Button Positions (Bottom Row)
        play_later_rect = pygame.Rect(WINDOW_WIDTH - BUTTON_WIDTH - 10, WINDOW_HEIGHT - BUTTON_HEIGHT - 10, BUTTON_WIDTH, BUTTON_HEIGHT)
        load_game_button_rect = pygame.Rect(play_later_rect.left - BUTTON_GAP - BUTTON_WIDTH, play_later_rect.top, BUTTON_WIDTH, BUTTON_HEIGHT)
        batch_game_button_rect = pygame.Rect(load_game_button_rect.left - BUTTON_GAP - BUTTON_WIDTH, play_later_rect.top, BUTTON_WIDTH, BUTTON_HEIGHT)
        start_game_button_rect = pygame.Rect(batch_game_button_rect.left - BUTTON_GAP - BUTTON_WIDTH, play_later_rect.top, BUTTON_WIDTH, BUTTON_HEIGHT)

        # Developer Tools Button (Bottom Left)
        dev_tools_button_width = 120 # Slightly wider for text
        dev_tools_button_rect = pygame.Rect(
            10, # Left edge padding
            play_later_rect.top, # Align vertically with other bottom buttons
            dev_tools_button_width, BUTTON_HEIGHT
        )

        # Calculate Load Input Field/Button Positions unconditionally
        load_input_width = 300
        load_input_x = load_game_button_rect.left
        load_input_y = load_game_button_rect.top - BUTTON_GAP - BUTTON_HEIGHT
        load_input_rect = pygame.Rect(load_input_x, load_input_y, load_input_width, BUTTON_HEIGHT)
        load_confirm_button_rect = pygame.Rect(load_input_x + load_input_width + BUTTON_GAP, load_input_y, BUTTON_WIDTH, BUTTON_HEIGHT)
        load_cancel_button_rect = pygame.Rect(load_confirm_button_rect.right + BUTTON_GAP, load_input_y, BUTTON_WIDTH, BUTTON_HEIGHT)

        # Calculate Checkbox Positions
        # Position checkboxes below the mode buttons
        last_mode_button_y = 100 + (len(modes) - 1) * 60 + BUTTON_HEIGHT # Y-coord of bottom of last mode button
        checkbox_x_base = content_left + (content_width - 250) // 2 # Approx center alignment
        checkbox_gap = 25 # Vertical gap between checkboxes

        # Endgame Solver Checkbox Position
        endgame_checkbox_x = checkbox_x_base
        endgame_checkbox_y = last_mode_button_y + 20 # Space below mode buttons
        endgame_checkbox_rect = pygame.Rect(endgame_checkbox_x, endgame_checkbox_y, 20, 20)
        endgame_label_x = endgame_checkbox_x + 25
        endgame_label_y = endgame_checkbox_y + 2 # Align text vertically with checkbox

        # AI Simulation Checkbox Position
        simulation_checkbox_x = checkbox_x_base
        simulation_checkbox_y = endgame_checkbox_y + checkbox_gap # Below endgame checkbox
        simulation_checkbox_rect = pygame.Rect(simulation_checkbox_x, simulation_checkbox_y, 20, 20)
        simulation_label_x = simulation_checkbox_x + 25
        simulation_label_y = simulation_checkbox_y + 2

        # Calculate Player Name Input Positions Dynamically
        name_input_gap = 30 # Gap below the last checkbox
        p1_y_pos = simulation_checkbox_y + simulation_checkbox_rect.height + name_input_gap
        player_name_gap = 40 # Gap between P1 and P2 inputs
        p2_y_pos = p1_y_pos + BUTTON_HEIGHT + player_name_gap # P2 position relative to P1

        # Calculate HVA button rects unconditionally (relative to p2_y_pos)
        hva_button_row_y = p2_y_pos + BUTTON_HEIGHT + 10 # Below P2 input area
        hva_buttons_total_width = (BUTTON_WIDTH * 2 + 20) # Width of two buttons + gap
        hva_buttons_start_x = content_left + (content_width - hva_buttons_total_width) // 2 # Centered
        p1_rect_hva = pygame.Rect(hva_buttons_start_x, hva_button_row_y, BUTTON_WIDTH, BUTTON_HEIGHT)
        p2_rect_hva = pygame.Rect(hva_buttons_start_x + BUTTON_WIDTH + 20, hva_button_row_y, BUTTON_WIDTH, BUTTON_HEIGHT)

        # Dropdown positioning (relative to p2_y_pos if HVH, or equivalent space if not)
        # Position dropdown below name inputs / HVA buttons depending on mode
        dropdown_base_y = p2_y_pos + BUTTON_HEIGHT + 10 # Default base below P2 input
        if modes[current_input] == MODE_HVA:
            dropdown_base_y = hva_button_row_y + BUTTON_HEIGHT + 10 # Below HVA buttons if HVA mode

        dropdown_x = name_rect_x # Align horizontally with name inputs
        dropdown_button_y = dropdown_base_y # Y position for the "Practice" button itself
        dropdown_y = dropdown_button_y + 30 # Y position for the start of the dropdown options

        options = ["Power Tiles", "8-Letter Bingos", "Bingo, Bango, Bongo", "Only Fives", "End Game"]
        for i, option in enumerate(options):
            option_rect = pygame.Rect(dropdown_x, dropdown_y + 30 * i, 200, 30)
            option_rects.append(option_rect)

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                selected_mode = None # Indicate quit
                loaded_game_data = None
                print("--- mode_selection_screen(): Quit event detected. ---")
                return None, None # Explicitly return None, None on quit

            # Handle Dev Tools Dialog FIRST if open
            if showing_dev_tools_dialog:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos
                    # Checkbox click - Visualize
                    if dev_tools_visualize_rect and dev_tools_visualize_rect.collidepoint(x, y):
                        visualize_batch_checked = not visualize_batch_checked
                    # Checkbox click - cProfile
                    elif dev_tools_cprofile_rect and dev_tools_cprofile_rect.collidepoint(x, y):
                        cprofile_checked = not cprofile_checked
                        DEVELOPER_PROFILE_ENABLED = cprofile_checked # Update global immediately
                    # Close button click
                    elif dev_tools_close_rect and dev_tools_close_rect.collidepoint(x, y):
                        showing_dev_tools_dialog = False
                elif event.type == pygame.KEYDOWN:
                     if event.key == pygame.K_ESCAPE: # Allow Esc to close
                         showing_dev_tools_dialog = False
                continue # Skip other event handling when dialog is open

            # Handle Load Game Input if active
            if showing_load_input:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos
                    if load_input_rect.collidepoint(x, y):
                        load_input_active = True
                    else:
                        load_input_active = False

                    if load_confirm_button_rect.collidepoint(x, y): # Load Confirm ...
                        filepath = load_filename_input.strip()
                        if filepath:
                            # Ensure the filepath has .sgs extension, or add it.
                            if not filepath.lower().endswith(".sgs"):
                                filepath += ".sgs"
                                load_filename_input = filepath # Update the input field display

                            try:
                                loaded_sgs_data = load_game_from_sgs(filepath)
                                if loaded_sgs_data:
                                    selected_mode = "LOADED_SGS_GAME" # More specific mode type
                                    loaded_game_data = loaded_sgs_data # Store the entire dict
                                    showing_load_input = False
                                    load_input_active = False
                                    # print(f"--- mode_selection_screen(): SGS Game Loaded: {filepath} ---")
                                    break # Exit mode selection loop
                                else:
                                    # load_game_from_sgs prints its own errors, but we can show a dialog too
                                    show_message_dialog(f"Failed to load SGS game:\n{filepath}\n(Check console for details)", "Load Error")
                                    load_input_active = True # Keep dialog open
                            except Exception as e: # Catch any other unexpected errors
                                print(f"--- mode_selection_screen(): Unexpected error during SGS load attempt for '{filepath}': {e} ---")
                                traceback.print_exc()
                                show_message_dialog(f"Unexpected error loading file:\n{e}", "Load Error")
                                load_input_active = True
                        else: # Filepath was empty
                            show_message_dialog("Please enter a filename (.sgs).", "Load Error")
                            load_input_active = True
                    elif load_cancel_button_rect.collidepoint(x,y): # Load Cancel ...
                        showing_load_input = False
                        load_input_active = False
                        load_filename_input = ""

                    # If clicked outside load input area, deactivate name inputs too
                    if not load_input_rect.collidepoint(x,y):
                        input_active = [False, False]

                elif event.type == pygame.KEYDOWN and load_input_active: # Load Input Typing ...
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        # Simulate clicking Load Confirm button
                        # (Need mouse pos, maybe use button center?)
                        confirm_center = load_confirm_button_rect.center
                        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=confirm_center))
                    elif event.key == pygame.K_BACKSPACE:
                        load_filename_input = load_filename_input[:-1]
                    elif event.key == pygame.K_v and (event.mod & pygame.KMOD_META): # CMD+V for paste
                         if pyperclip_available:
                             try:
                                 pasted_text = pyperclip.paste()
                                 if pasted_text: load_filename_input += pasted_text
                             except Exception as e:
                                 print(f"Error during paste: {e}")
                         else: print("Paste unavailable (pyperclip not loaded).")
                    elif event.unicode.isprintable(): # Allow typing printable characters
                        load_filename_input += event.unicode
                # If load input is active, skip other event processing for this cycle
                continue # Skip rest of event handling

            # Handle other events if load input wasn't active

            # Handle Power Tiles Dialog Events
            if showing_power_tiles_dialog:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos
                    # Define rects locally for collision detection
                    dialog_width, dialog_height = 300, 250
                    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
                    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
                    letter_rects = [pygame.Rect(dialog_x + 20, dialog_y + 40 + i*30, 20, 20) for i in range(4)]
                    number_rects = [pygame.Rect(dialog_x + 150, dialog_y + 40 + i*30, 20, 20) for i in range(6)]
                    go_rect = pygame.Rect(dialog_x + 50, dialog_y + 220, 100, 30)
                    cancel_rect = pygame.Rect(dialog_x + 160, dialog_y + 220, 100, 30)

                    for i, rect in enumerate(letter_rects):
                        if rect.collidepoint(x, y): letter_checks[i] = not letter_checks[i]
                    for i, rect in enumerate(number_rects):
                        if rect.collidepoint(x, y): number_checks[i] = not number_checks[i]

                    if go_rect.collidepoint(x, y):
                        practice_mode = "power_tiles"
                        selected_mode = MODE_AVA # Default Power Tiles to AI vs AI
                        showing_power_tiles_dialog = False
                        print(f"--- mode_selection_screen(): Mode selected via Power Tiles Go: {selected_mode} ---")
                        break # Exit event loop, then outer while loop
                    elif cancel_rect.collidepoint(x, y):
                        showing_power_tiles_dialog = False
                continue # Skip other events if dialog was open

            # Handle MOUSEBUTTONDOWN for main screen elements
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                # Mode Selection Buttons
                mode_rects = [] # Recalculate rects for collision check
                for i, mode in enumerate(modes):
                    y_pos_mode = 100 + i * 60
                    rect = pygame.Rect(content_left + (content_width - (BUTTON_WIDTH * 2 + 20)) // 2, y_pos_mode, BUTTON_WIDTH * 2 + 20, BUTTON_HEIGHT)
                    mode_rects.append(rect)
                    if rect.collidepoint(x, y):
                        current_input = i
                        dropdown_open = False # Close dropdown if mode changes
                        # Update player names and input active state based on new mode
                        if i == 0: # HVH
                            player_names = ["Player 1", "Player 2"]
                            input_active = [False, False]
                        elif i == 1: # HVA
                            # Keep existing name if switching between P1/P2 HVA
                            p1_name = player_names[0] if player_names[0] != "AI" else "Player 1"
                            p2_name = player_names[1] if player_names[1] != "AI" else "Player 2"
                            player_names = [p1_name, "AI"] if human_player == 1 else ["AI", p2_name]
                            input_active = [True, False] if human_player == 1 else [False, True]
                        elif i == 2: # AVA
                            player_names = ["AI 1", "AI 2"]
                            input_active = [False, False]
                        break # Mode button handled

                # Checkbox Clicks (Endgame Solver / AI Simulation)
                if endgame_checkbox_rect.collidepoint(x, y):
                    use_endgame_solver_checked = not use_endgame_solver_checked
                elif simulation_checkbox_rect.collidepoint(x, y):
                    use_ai_simulation_checked = not use_ai_simulation_checked

                # Developer Tools Button Click
                elif dev_tools_button_rect.collidepoint(x, y):
                    showing_dev_tools_dialog = True

                # Bottom Row Button Clicks
                elif start_game_button_rect.collidepoint(x, y):
                    current_selected_game_mode = modes[current_input]
                    # Validate player names (e.g., ensure not empty if human)
                    valid_start = True
                    if current_selected_game_mode == MODE_HVH:
                        if not player_names[0].strip() or not player_names[1].strip():
                            show_message_dialog("Please enter names for both players.", "Input Error")
                            valid_start = False
                    elif current_selected_game_mode == MODE_HVA:
                        human_idx = 0 if human_player == 1 else 1
                        if not player_names[human_idx].strip():
                            show_message_dialog(f"Please enter a name for Player {human_player}.", "Input Error")
                            valid_start = False
                    
                    if valid_start:
                        selected_mode = modes[current_input]
                        print(f"--- mode_selection_screen(): Start Game clicked. Mode: {selected_mode} ---")
                        # Break handled by while loop condition (selected_mode is not None)
                elif batch_game_button_rect.collidepoint(x, y):
                     current_selected_game_mode = modes[current_input]
                     if current_selected_game_mode == MODE_HVH:
                         show_message_dialog("Batch mode is only available for modes involving AI (HVA or AVA).", "Batch Mode Error")
                     else:
                         num_games = get_batch_game_dialog()
                         if num_games is not None and num_games > 0:
                             selected_mode = "BATCH_MODE"
                             # Package all necessary config for batch mode start
                             loaded_game_data = (current_selected_game_mode, player_names, human_player, use_endgame_solver_checked, use_ai_simulation_checked, num_games, visualize_batch_checked, cprofile_checked)
                             print(f"--- mode_selection_screen(): Batch Game selected. Mode: {current_selected_game_mode}, Games: {num_games} ---")
                             # Break handled by while loop condition
                         else:
                             print("--- mode_selection_screen(): Batch game cancelled or invalid number entered. ---")

                elif load_game_button_rect.collidepoint(x, y):
                    showing_load_input = True
                    load_input_active = True # Activate input field immediately
                    load_filename_input = "" # Clear previous input
                elif play_later_rect.collidepoint(x, y):
                    selected_mode = None # Indicate quit / play later
                    loaded_game_data = None
                    print("--- mode_selection_screen(): Play Later selected. Exiting. ---")
                    return None, None # Explicitly return None, None

                # Player Name Input Activation
                clicked_name_input = False
                p1_name_rect = pygame.Rect(name_rect_x, p1_y_pos, 200, BUTTON_HEIGHT)
                p2_name_rect = pygame.Rect(name_rect_x, p2_y_pos, 200, BUTTON_HEIGHT) if modes[current_input] == MODE_HVH else None

                if modes[current_input] == MODE_HVH:
                    if p1_name_rect.collidepoint(x, y):
                        input_active = [True, False]
                        clicked_name_input = True
                    elif p2_name_rect and p2_name_rect.collidepoint(x, y):
                        input_active = [False, True]
                        clicked_name_input = True
                elif modes[current_input] == MODE_HVA:
                    # Determine which input box corresponds to the human player
                    human_input_rect = p1_name_rect if human_player == 1 else pygame.Rect(name_rect_x, p2_y_pos, 200, BUTTON_HEIGHT) # Need p2 rect for HVA P2
                    if human_input_rect.collidepoint(x, y):
                         input_active = [True, False] if human_player == 1 else [False, True]
                         clicked_name_input = True
                    # HVA Player Selection Buttons
                    if p1_rect_hva.collidepoint(x, y):
                        human_player = 1
                        player_names = [player_names[0] if player_names[0] != "AI" else "Player 1", "AI"]
                        input_active = [True, False] # Activate P1 input
                    elif p2_rect_hva.collidepoint(x, y):
                        human_player = 2
                        player_names = ["AI", player_names[1] if player_names[1] != "AI" else "Player 2"]
                        input_active = [False, True] # Activate P2 input

                # Practice Dropdown Handling (Only if HVH mode is selected)
                if modes[current_input] == MODE_HVH:
                    dropdown_button_rect = pygame.Rect(name_rect_x, dropdown_button_y, 200, 30)
                    if dropdown_button_rect.collidepoint(x, y):
                        dropdown_open = not dropdown_open
                    elif dropdown_open:
                        clicked_option = False
                        current_options = ["Power Tiles", "8-Letter Bingos", "Bingo, Bango, Bongo", "Only Fives", "End Game"]
                        for i, option_rect in enumerate(option_rects):
                            if option_rect.collidepoint(x, y):
                                clicked_option = True
                                dropdown_open = False
                                selected_practice_option = current_options[i]
                                print(f"Practice option selected: {selected_practice_option}")
                                if selected_practice_option == "Power Tiles":
                                    showing_power_tiles_dialog = True
                                elif selected_practice_option == "8-Letter Bingos":
                                    # Call the setup function which now returns state or None
                                    setup_ok, p_board, p_tiles, p_racks, p_blanks, p_bag, p_max_idx = eight_letter_practice()
                                    if setup_ok:
                                        practice_mode = "eight_letter"
                                        selected_mode = MODE_HVH # 8-letter is HVH practice
                                        player_names = ["Player 1", ""] # P2 name not used
                                        human_player = 1
                                        practice_state = {"board": p_board, "tiles": p_tiles, "racks": p_racks, "blanks": p_blanks, "bag": p_bag, "first_play": False, "scores": [0, 0], "turn": 1}
                                        practice_probability_max_index = p_max_idx # Store the index
                                        print(f"--- mode_selection_screen(): 8-Letter Practice selected. Max Index: {practice_probability_max_index} ---")
                                        # Break handled by while loop condition
                                    else:
                                        print("--- mode_selection_screen(): 8-Letter Practice setup cancelled or failed. ---")

                                elif selected_practice_option == "Bingo, Bango, Bongo":
                                    practice_mode = "bingo_bango_bongo"
                                    selected_mode = MODE_AVA # Default to AI vs AI
                                    player_names = ["AI 1", "AI 2"]
                                    practice_state = None # No specific pre-state needed usually
                                    print(f"--- mode_selection_screen(): Bingo Bango Bongo selected. Mode: {selected_mode} ---")
                                    # Break handled by while loop condition
                                elif selected_practice_option == "Only Fives":
                                    practice_mode = "only_fives"
                                    selected_mode = MODE_HVA # Default to HVA
                                    human_player = 1
                                    player_names = ["Player 1", "AI"]
                                    practice_state = None
                                    print(f"--- mode_selection_screen(): Only Fives selected. Mode: {selected_mode} ---")
                                    # Break handled by while loop condition
                                elif selected_practice_option == "End Game":
                                    # Placeholder for End Game practice setup
                                    print("End Game practice mode selected (Not Implemented Yet)")
                                    show_message_dialog("End Game practice mode is not yet implemented.", "Coming Soon")
                                break # Option selected, exit inner loop

                        # Close dropdown if clicked outside of it or its options
                        if not clicked_option and not dropdown_button_rect.collidepoint(x,y):
                             dropdown_open = False
                    # If clicked outside dropdown button when it's closed
                    elif not dropdown_open and not dropdown_button_rect.collidepoint(x,y):
                         pass # Do nothing, dropdown remains closed

                # Deactivate name input if clicked elsewhere (and not on another input)
                # Check dropdown button too if HVH mode
                dropdown_button_y_check = p2_y_pos + BUTTON_HEIGHT + 10 # Recalculate base Y
                if modes[current_input] == MODE_HVA: dropdown_button_y_check = hva_button_row_y + BUTTON_HEIGHT + 10
                dropdown_rect_check = pygame.Rect(name_rect_x, dropdown_button_y_check, 200, 30) if modes[current_input] == MODE_HVH else None

                is_click_on_input_area = clicked_name_input
                if dropdown_rect_check and dropdown_rect_check.collidepoint(x,y): is_click_on_input_area = True
                if dropdown_open: # Also check dropdown options if open
                    for r in option_rects:
                        if r.collidepoint(x,y): is_click_on_input_area = True; break

                if not is_click_on_input_area:
                    input_active = [False, False]


            # Handle KEYDOWN for text input
            elif event.type == pygame.KEYDOWN:
                active_idx = -1
                if input_active[0]: active_idx = 0
                elif input_active[1] and modes[current_input] == MODE_HVH: active_idx = 1
                elif input_active[1] and modes[current_input] == MODE_HVA and human_player == 2: active_idx = 1

                if active_idx != -1: # If a name input is active
                    if event.key == pygame.K_BACKSPACE:
                        player_names[active_idx] = player_names[active_idx][:-1]
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        input_active[active_idx] = False # Deactivate on enter
                    elif event.unicode.isprintable(): # Allow typing printable characters
                        player_names[active_idx] += event.unicode

        # --- Drawing Logic ---
        screen.fill(WHITE)
        if image: screen.blit(image, (content_left, 0)) # Draw background image if loaded

        # Title
        title_text = dialog_font.render("Select Game Mode", True, BLACK)
        title_x = content_left + (content_width - title_text.get_width()) // 2
        screen.blit(title_text, (title_x, 50))

        # Draw Mode Buttons
        mode_rects = [] # Redefine for drawing scope
        for i, mode in enumerate(modes):
            y_pos_mode = 100 + i * 60 # Use different variable name
            rect = pygame.Rect(content_left + (content_width - (BUTTON_WIDTH * 2 + 20)) // 2, y_pos_mode, BUTTON_WIDTH * 2 + 20, BUTTON_HEIGHT)
            hover = rect.collidepoint(pygame.mouse.get_pos())
            color = BUTTON_HOVER if i == current_input or hover else BUTTON_COLOR
            pygame.draw.rect(screen, color, rect)
            text = button_font.render(mode, True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
            mode_rects.append(rect) # Store rect for potential later use

        # Draw Endgame Solver Checkbox
        pygame.draw.rect(screen, WHITE, endgame_checkbox_rect)
        pygame.draw.rect(screen, BLACK, endgame_checkbox_rect, 1)
        if use_endgame_solver_checked:
            pygame.draw.line(screen, BLACK, endgame_checkbox_rect.topleft, endgame_checkbox_rect.bottomright, 2)
            pygame.draw.line(screen, BLACK, endgame_checkbox_rect.topright, endgame_checkbox_rect.bottomleft, 2)
        endgame_label_surf = ui_font.render("Use AI Endgame Solver", True, BLACK)
        screen.blit(endgame_label_surf, (endgame_label_x, endgame_label_y))

        # Draw AI Simulation Checkbox
        pygame.draw.rect(screen, WHITE, simulation_checkbox_rect)
        pygame.draw.rect(screen, BLACK, simulation_checkbox_rect, 1)
        if use_ai_simulation_checked:
            pygame.draw.line(screen, BLACK, simulation_checkbox_rect.topleft, simulation_checkbox_rect.bottomright, 2)
            pygame.draw.line(screen, BLACK, simulation_checkbox_rect.topright, simulation_checkbox_rect.bottomleft, 2)
        simulation_label_surf = ui_font.render("Use AI 2-ply Simulation", True, BLACK)
        screen.blit(simulation_label_surf, (simulation_label_x, simulation_label_y))

        # Draw Bottom Buttons
        # Play Later Button
        hover = play_later_rect.collidepoint(pygame.mouse.get_pos())
        color = BUTTON_HOVER if hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, play_later_rect)
        play_later_text = button_font.render("Play Later", True, BLACK)
        play_later_text_rect = play_later_text.get_rect(center=play_later_rect.center)
        screen.blit(play_later_text, play_later_text_rect)
        # Load Game Button
        hover = load_game_button_rect.collidepoint(pygame.mouse.get_pos())
        color = BUTTON_HOVER if hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, load_game_button_rect)
        load_game_text = button_font.render("Load Game", True, BLACK)
        load_game_text_rect = load_game_text.get_rect(center=load_game_button_rect.center)
        screen.blit(load_game_text, load_game_text_rect)
        # Batch Game Button
        hover = batch_game_button_rect.collidepoint(pygame.mouse.get_pos())
        color = BUTTON_HOVER if hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, batch_game_button_rect)
        batch_game_text = button_font.render("Batch Games", True, BLACK)
        batch_game_text_rect = batch_game_text.get_rect(center=batch_game_button_rect.center)
        screen.blit(batch_game_text, batch_game_text_rect)
        # Start Game Button
        hover = start_game_button_rect.collidepoint(pygame.mouse.get_pos())
        color = BUTTON_HOVER if hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, start_game_button_rect)
        start_game_text = button_font.render("Start Game", True, BLACK)
        start_game_text_rect = start_game_text.get_rect(center=start_game_button_rect.center)
        screen.blit(start_game_text, start_game_text_rect)
        # Developer Tools Button (Bottom Left)
        hover = dev_tools_button_rect.collidepoint(pygame.mouse.get_pos())
        color = BUTTON_HOVER if hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, dev_tools_button_rect)
        dev_tools_text = button_font.render("Developer Tools", True, BLACK)
        dev_tools_text_rect = dev_tools_text.get_rect(center=dev_tools_button_rect.center)
        screen.blit(dev_tools_text, dev_tools_text_rect)


        # Draw Load Game Input Field (if active)
        if showing_load_input:
            pygame.draw.rect(screen, WHITE, load_input_rect) # Background
            pygame.draw.rect(screen, BLACK, load_input_rect, 1 if not load_input_active else 2) # Border (thicker if active)
            input_surf = ui_font.render(load_filename_input, True, BLACK)
            screen.blit(input_surf, (load_input_rect.x + 5, load_input_rect.y + 5))
            # Blinking cursor
            if load_input_active and int(time.time() * 2) % 2 == 0:
                cursor_x = load_input_rect.x + 5 + input_surf.get_width()
                cursor_y1 = load_input_rect.y + 5
                cursor_y2 = load_input_rect.y + load_input_rect.height - 5
                pygame.draw.line(screen, BLACK, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 1)
            # Draw Load Confirm/Cancel buttons
            hover = load_confirm_button_rect.collidepoint(pygame.mouse.get_pos())
            color = BUTTON_HOVER if hover else BUTTON_COLOR # Confirm Button
            pygame.draw.rect(screen, color, load_confirm_button_rect)
            text = button_font.render("Load File", True, BLACK)
            text_rect = text.get_rect(center=load_confirm_button_rect.center)
            screen.blit(text, text_rect)
            hover = load_cancel_button_rect.collidepoint(pygame.mouse.get_pos())
            color = BUTTON_HOVER if hover else BUTTON_COLOR # Cancel Button
            pygame.draw.rect(screen, color, load_cancel_button_rect)
            text = button_font.render("Cancel", True, BLACK)
            text_rect = text.get_rect(center=load_cancel_button_rect.center)
            screen.blit(text, text_rect)


        # Name Input / Practice Dropdown / HVA Selection Drawing
        name_rect_width = 200 # Width of name input boxes

        # Player 1 Input Box (Always potentially visible)
        p1_label_text = "Player 1 Name:"
        p1_label = ui_font.render(p1_label_text, True, BLACK)
        p1_name_rect = pygame.Rect(name_rect_x, p1_y_pos, name_rect_width, BUTTON_HEIGHT) # Define for drawing
        p1_label_x = name_rect_x - p1_label.get_width() - 10 # Position label left of box
        screen.blit(p1_label, (p1_label_x, p1_y_pos + 5))
        # Background color depends on mode and active state
        p1_bg_color = LIGHT_BLUE if input_active[0] else (GRAY if modes[current_input] == MODE_AVA else WHITE)
        pygame.draw.rect(screen, p1_bg_color, p1_name_rect)
        pygame.draw.rect(screen, BLACK, p1_name_rect, 1) # Border
        p1_name_text = ui_font.render(player_names[0], True, BLACK)
        screen.blit(p1_name_text, (p1_name_rect.x + 5, p1_name_rect.y + 5))
        # Blinking cursor for P1 input
        if input_active[0] and int(time.time() * 2) % 2 == 0:
            cursor_x = p1_name_rect.x + 5 + p1_name_text.get_width()
            cursor_y1 = p1_name_rect.y + 5
            cursor_y2 = p1_name_rect.bottom - 5
            pygame.draw.line(screen, BLACK, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 1)


        # Player 2 Input Box / AI Label / Practice Dropdown
        if modes[current_input] == MODE_HVH:
            # Draw P2 Input Box
            p2_label_text = "Player 2 Name:"
            p2_label = ui_font.render(p2_label_text, True, BLACK)
            p2_name_rect = pygame.Rect(name_rect_x, p2_y_pos, name_rect_width, BUTTON_HEIGHT) # Define for drawing
            p2_label_x = name_rect_x - p2_label.get_width() - 10
            screen.blit(p2_label, (p2_label_x, p2_y_pos + 5))
            pygame.draw.rect(screen, WHITE if not input_active[1] else LIGHT_BLUE, p2_name_rect)
            pygame.draw.rect(screen, BLACK, p2_name_rect, 1)
            p2_name_text = ui_font.render(player_names[1], True, BLACK)
            screen.blit(p2_name_text, (p2_name_rect.x + 5, p2_name_rect.y + 5))
            # Blinking cursor for P2 input
            if input_active[1] and int(time.time() * 2) % 2 == 0:
                cursor_x = p2_name_rect.x + 5 + p2_name_text.get_width()
                cursor_y1 = p2_name_rect.y + 5
                cursor_y2 = p2_name_rect.bottom - 5
                pygame.draw.line(screen, BLACK, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 1)

            # Draw Practice Dropdown Button (Only in HVH)
            dropdown_rect = pygame.Rect(name_rect_x, dropdown_button_y, 200, 30)
            hover = dropdown_rect.collidepoint(pygame.mouse.get_pos())
            color = BUTTON_HOVER if hover else BUTTON_COLOR
            pygame.draw.rect(screen, color, dropdown_rect)
            text = button_font.render("Practice", True, BLACK)
            text_rect = text.get_rect(center=dropdown_rect.center)
            screen.blit(text, text_rect)
            # Draw Dropdown Options if open
            if dropdown_open:
                current_options = ["Power Tiles", "8-Letter Bingos", "Bingo, Bango, Bongo", "Only Fives", "End Game"]
                for i, option_rect in enumerate(option_rects):
                    hover = option_rect.collidepoint(pygame.mouse.get_pos())
                    color = BUTTON_HOVER if hover else DROPDOWN_COLOR
                    pygame.draw.rect(screen, color, option_rect)
                    text = button_font.render(current_options[i], True, BLACK)
                    text_rect = text.get_rect(center=option_rect.center)
                    screen.blit(text, text_rect)

        elif modes[current_input] == MODE_HVA:
            # Draw P2 as AI (Grayed out)
            p2_label_text = "AI Name:"
            p2_label = ui_font.render(p2_label_text, True, BLACK)
            p2_name_rect_hva = pygame.Rect(name_rect_x, p2_y_pos, name_rect_width, BUTTON_HEIGHT) # Define for drawing
            p2_label_x = name_rect_x - p2_label.get_width() - 10
            screen.blit(p2_label, (p2_label_x, p2_y_pos + 5))
            pygame.draw.rect(screen, GRAY, p2_name_rect_hva) # Gray background for AI
            pygame.draw.rect(screen, BLACK, p2_name_rect_hva, 1)
            p2_name_text = ui_font.render(player_names[1] if human_player == 1 else player_names[0], True, BLACK) # Show AI name
            screen.blit(p2_name_text, (p2_name_rect_hva.x + 5, p2_name_rect_hva.y + 5))

            # Draw HVA Player Selection Buttons
            p1_hover = p1_rect_hva.collidepoint(pygame.mouse.get_pos())
            p2_hover = p2_rect_hva.collidepoint(pygame.mouse.get_pos())
            # Highlight selected player button
            pygame.draw.rect(screen, BUTTON_HOVER if p1_hover or human_player == 1 else BUTTON_COLOR, p1_rect_hva)
            pygame.draw.rect(screen, BUTTON_HOVER if p2_hover or human_player == 2 else BUTTON_COLOR, p2_rect_hva)
            p1_text = button_font.render("Play as P1", True, BLACK)
            p2_text = button_font.render("Play as P2", True, BLACK)
            p1_text_rect = p1_text.get_rect(center=p1_rect_hva.center)
            p2_text_rect = p2_text.get_rect(center=p2_rect_hva.center)
            screen.blit(p1_text, p1_text_rect)
            screen.blit(p2_text, p2_text_rect)

        elif modes[current_input] == MODE_AVA:
            # Draw P2 as AI (Grayed out)
            p2_label_text = "AI 2 Name:"
            p2_label = ui_font.render(p2_label_text, True, BLACK)
            p2_name_rect_ava = pygame.Rect(name_rect_x, p2_y_pos, name_rect_width, BUTTON_HEIGHT) # Define for drawing
            p2_label_x = name_rect_x - p2_label.get_width() - 10
            screen.blit(p2_label, (p2_label_x, p2_y_pos + 5))
            pygame.draw.rect(screen, GRAY, p2_name_rect_ava) # Gray background
            pygame.draw.rect(screen, BLACK, p2_name_rect_ava, 1)
            p2_name_text = ui_font.render(player_names[1], True, BLACK)
            screen.blit(p2_name_text, (p2_name_rect_ava.x + 5, p2_name_rect_ava.y + 5))
            # Also gray out P1 input for AVA
            pygame.draw.rect(screen, GRAY, p1_name_rect)
            pygame.draw.rect(screen, BLACK, p1_name_rect, 1)
            p1_name_text = ui_font.render(player_names[0], True, BLACK)
            screen.blit(p1_name_text, (p1_name_rect.x + 5, p1_name_rect.y + 5))


        # Draw Power Tiles Dialog if active
        if showing_power_tiles_dialog:
            # Draw dialog background
            dialog_width, dialog_height = 300, 250
            dialog_x = (WINDOW_WIDTH - dialog_width) // 2
            dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
            pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))
            pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2) # Border
            # Title
            title_text = dialog_font.render("Power Tiles Options", True, BLACK)
            screen.blit(title_text, (dialog_x + 10, dialog_y + 10))
            # Checkboxes
            letters = ['J', 'Q', 'X', 'Z']
            for i, letter in enumerate(letters):
                draw_checkbox(screen, dialog_x + 20, dialog_y + 40 + i*30, letter_checks[i])
                text = ui_font.render(letter, True, BLACK)
                screen.blit(text, (dialog_x + 50, dialog_y + 40 + i*30))
            numbers = ['2', '3', '4', '5', '6', '7+']
            for i, num in enumerate(numbers):
                draw_checkbox(screen, dialog_x + 150, dialog_y + 40 + i*30, number_checks[i])
                text = ui_font.render(num, True, BLACK)
                screen.blit(text, (dialog_x + 180, dialog_y + 40 + i*30))
            # Buttons
            go_rect = pygame.Rect(dialog_x + 50, dialog_y + 220, 100, 30)
            cancel_rect = pygame.Rect(dialog_x + 160, dialog_y + 220, 100, 30)
            pygame.draw.rect(screen, BUTTON_COLOR, go_rect)
            pygame.draw.rect(screen, BUTTON_COLOR, cancel_rect)
            go_text = button_font.render("Go", True, BLACK)
            cancel_text = button_font.render("Cancel", True, BLACK)
            screen.blit(go_text, go_text.get_rect(center=go_rect.center))
            screen.blit(cancel_text, cancel_text.get_rect(center=cancel_rect.center))

        # Draw Developer Tools Dialog
        if showing_dev_tools_dialog:
            # Pass cprofile_checked state to the drawing function
            dev_tools_visualize_rect, dev_tools_cprofile_rect, dev_tools_close_rect = draw_dev_tools_dialog(visualize_batch_checked, cprofile_checked)


        # --- Display Update ---
        pygame.display.flip()

        # Break loop if a mode was selected (either by button or practice option)
        if selected_mode is not None:
            break

    # --- Exit Condition Check ---
    if selected_mode is None: # If loop exited without selecting a mode (e.g., closed window)
        print("--- mode_selection_screen(): Loop exited without mode selection. ---")
        return None, None

    # Return loaded game data or new game setup data
    if selected_mode == "LOADED_SGS_GAME":
        # loaded_game_data is the dictionary loaded from SGS file
        return selected_mode, loaded_game_data
    elif selected_mode == "BATCH_MODE":
        # loaded_game_data is the tuple configured for batch mode
        return selected_mode, loaded_game_data
    else: # New Game or Practice Mode
        # Construct the return tuple - ENSURE 11 ITEMS
        return_data_tuple = (
            player_names, human_player, practice_mode,
            letter_checks, number_checks, use_endgame_solver_checked,
            use_ai_simulation_checked, practice_state,
            visualize_batch_checked, # Pass visualize setting
            cprofile_checked,        # Pass cProfile setting
            practice_probability_max_index # Pass max index (will be None if not 8-letter)
        )
        # <<< ADD DEBUG PRINT >>>
        print(f"DEBUG: mode_selection_screen - About to return tuple with {len(return_data_tuple)} items for mode {selected_mode}")
        # <<< END DEBUG PRINT >>>
        return selected_mode, return_data_tuple







def draw_options_menu(turn, dropdown_open, bag_count, replay_mode):
    """Draw the options menu with dropdown functionality, adding Stop Batch and Specify Rack."""
    global practice_mode, is_batch_running # Access global flags
    options_x = 10; options_rect = pygame.Rect(options_x, OPTIONS_Y, OPTIONS_WIDTH, OPTIONS_HEIGHT)
    hover = options_rect.collidepoint(pygame.mouse.get_pos()); color = BUTTON_HOVER if hover else BUTTON_COLOR; pygame.draw.rect(screen, color, options_rect)
    options_text = button_font.render("Options", True, BLACK); options_text_rect = options_text.get_rect(center=(options_x + OPTIONS_WIDTH // 2, OPTIONS_Y + OPTIONS_HEIGHT // 2)); screen.blit(options_text, options_text_rect)
    dropdown_rects = []
    if dropdown_open:
        dropdown_y = OPTIONS_Y + OPTIONS_HEIGHT
        if replay_mode: # Check replay_mode first
            options = ["Main", "Quit"]
        elif is_batch_running:
            options = ["Stop Batch", "Quit"]
        elif practice_mode == "eight_letter":
            options = ["Give Up", "Main", "Quit"]
        else: # Standard game or other practice modes
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
            leave_score = evaluate_leave_cython(move['rack'])
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





# In scrabble_helpers.py (or wherever get_remaining_tiles is defined)

def get_remaining_tiles(rack, tiles, blanks, blanks_played_count): # Added tiles, blanks; removed board_tile_counts
    """
    Calculate the remaining tiles (bag + opponent rack) using the actual
    board state and rack.
    """
    # 1. Start with the absolute initial distribution
    initial_distribution_counts = Counter({letter: count 
                                           for letter, (count, _) in TILE_DISTRIBUTION.items()})

    # 2. Account for the tiles on the current player's rack
    rack_counts = Counter(rack)

    # 3. Account for the physical tiles actually on the board squares
    physical_tiles_on_board = Counter()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if tiles[r][c]: # If there's a tile on the square
                if (r, c) in blanks: # Check if this coordinate is a known blank
                    physical_tiles_on_board[' '] += 1 # Count it as a physical blank tile
                else:
                    physical_tiles_on_board[tiles[r][c]] += 1 # Count it as a letter tile

    # 4. Calculate total tiles accounted for (Player Rack + Physical Tiles on Board)
    #    Note: blanks_played_count is implicitly handled by counting ' ' in physical_tiles_on_board
    #    We need to ensure physical_tiles_on_board[' '] == blanks_played_count for consistency.
    #    Let's trust physical_tiles_on_board derived from the board state.
    if physical_tiles_on_board[' '] != blanks_played_count:
         print(f"WARNING (get_remaining_tiles): Discrepancy! Blanks on board ({physical_tiles_on_board[' ']}) != blanks_played_count ({blanks_played_count}). Using board count.")
         # This warning indicates an issue in how blanks_played_count is tracked vs the actual blanks set.

    total_tiles_accounted_for = rack_counts + physical_tiles_on_board
    
    # 5. Remaining = Initial - TotalAccountedFor
    remaining_counts = initial_distribution_counts - total_tiles_accounted_for
    
    # 6. Ensure non-negative counts and filter out zeros
    final_remaining = {tile: max(0, count) 
                       for tile, count in remaining_counts.items() if count > 0} 

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














'''
# Function to Replace: generate_all_moves_gaddag
# REASON: Remove post-processing, return result from Cython helper directly.

def generate_all_moves_gaddag(rack, tiles, board, blanks, gaddag_root):
    """
    Generates ALL valid Scrabble moves using GADDAG traversal.
    Handles setup in Python, calls Cython helper for core logic and post-processing.
    """
    # --- Access Cython functions and globals ---
    # Ensure _process_anchors_cython is imported correctly
    global _gaddag_traverse_cython, _process_anchors_cython
    global gaddag_loading_status
    global DAWG # Access the globally loaded DAWG
    from scrabble_helpers import get_anchor_points # Keep this import here

    # --- Setup Phase (Stays in Python) ---
    if gaddag_loading_status != 'loaded' or gaddag_root is None:
        print("Error: GADDAG not loaded or has no root. Cannot generate moves.")
        return []
    if DAWG is None:
        print("Error: DAWG object is None, cannot compute cross_check_sets or validate words.")
        return []
    # Check for the core traversal function existence (optional, but good practice)
    # Note: _gaddag_traverse is now a Python function defined in Cython file,
    # so checking for _gaddag_traverse_cython might be misleading if we only call _gaddag_traverse
    # Let's rely on the _process_anchors_cython check for now.
    # if '_gaddag_traverse_cython' not in globals() and '_gaddag_traverse_cython' not in locals():
    #      print("CRITICAL ERROR: _gaddag_traverse_cython not found!")
    #      return [] # Cannot proceed without traversal function
    if '_process_anchors_cython' not in globals() and '_process_anchors_cython' not in locals():
         print("CRITICAL ERROR: _process_anchors_cython not found!")
         return [] # Cannot proceed without the helper

    # Variable initializations for setup
    is_first_play = sum(1 for row in tiles for t in row if t) == 0
    anchors = get_anchor_points(tiles, is_first_play) # Call Python version
    original_tiles_state = [row[:] for row in tiles]
    full_rack_size = len(rack)

    # Rack conversion (NumPy array and Python Counter)
    rack_counts_py = Counter(rack)
    rack_counts_c_arr = np.zeros(27, dtype=np.intc)
    for i in range(26):
        letter = chr(ord('A') + i)
        rack_counts_c_arr[i] = rack_counts_py.get(letter, 0)
    rack_counts_c_arr[26] = rack_counts_py.get(' ', 0)

    # Cross-check computation call
    start_cross_check_time = time.time()
    cross_check_sets = compute_cross_checks_cython(tiles, DAWG) # Call Cython version
    # print(f"DEBUG: Cross-check computation took {time.time() - start_cross_check_time:.4f} seconds.")

    # --- Core Logic & Post-processing Phase (Call Cython Helper) ---
    final_unique_moves = [] # Initialize default return
    try:
        # Call the Cython helper function, which now returns the final list
        final_unique_moves = _process_anchors_cython( # Returns list now
            anchors,
            rack_counts_c_arr, # Pass NumPy array as object
            rack_counts_py,
            tiles,
            board,
            blanks,
            cross_check_sets,
            gaddag_root,
            original_tiles_state,
            is_first_play,
            full_rack_size,
            DAWG
        )
    except Exception as e:
        print(f"ERROR calling _process_anchors_cython: {e}")
        import traceback
        traceback.print_exc()
        return [] # Return empty list on error

    # --- REMOVED Post-processing Phase (Moved into Cython) ---

    return final_unique_moves # Return the result from Cython directly
'''






def draw_hint_dialog(moves, selected_index, is_simulation_result=False, best_exchange_tiles=None, best_exchange_score=None):
    # Dialog dimensions and position
    dialog_width = 400 
    # Calculate required height dynamically
    padding = 15
    header_height = 40 # For title
    line_height = 30   # For each move/option
    button_height_area = BUTTON_HEIGHT + padding * 2 # Space for buttons + padding

    num_play_moves_to_show = 0
    if isinstance(moves, list):
        num_play_moves_to_show = min(len(moves), 5)

    add_exchange_option = bool(best_exchange_tiles)
    
    total_items_in_list = num_play_moves_to_show
    if add_exchange_option:
        total_items_in_list += 1
    
    # Ensure at least one line height if no items, for "No options" message
    content_list_height = max(total_items_in_list * line_height, line_height if total_items_in_list == 0 else 0)

    dialog_height = header_height + content_list_height + button_height_area + padding # Top padding for content too

    dialog_x = (WINDOW_WIDTH - dialog_width) // 2
    dialog_y = (WINDOW_HEIGHT - dialog_height) // 2

    # Draw dialog background and border
    pygame.draw.rect(screen, BLACK, (dialog_x - 2, dialog_y - 2, dialog_width + 4, dialog_height + 4), 2)
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, dialog_width, dialog_height))

    # Title
    title_str = "Simulation Results" if is_simulation_result else \
                ("Top Moves / Exchange" if (moves and num_play_moves_to_show > 0) or add_exchange_option else "No Options Available")
    title_text = dialog_font.render(title_str, True, BLACK)
    screen.blit(title_text, (dialog_x + (dialog_width - title_text.get_width()) // 2, dialog_y + 10))

    hint_rects = [] # Stores (rect, original_index_in_moves_or_exchange_flag)
    y_pos = dialog_y + header_height + padding 

    # --- Draw Play Moves ---
    play_moves_drawn_count = 0
    if isinstance(moves, list):
        for i, move_data_item in enumerate(moves):
            if play_moves_drawn_count >= 5: # Max 5 play moves
                break

            move = {}
            raw_score = 0
            final_score_sim = 0.0 # For simulation results

            if is_simulation_result and isinstance(move_data_item, dict):
                move = move_data_item.get('move', {})
                final_score_sim = move_data_item.get('final_score', 0.0)
                raw_score = move.get('score', 0) # Get raw score from inner move
            elif not is_simulation_result and isinstance(move_data_item, dict): # Direct move dict
                move = move_data_item
                raw_score = move.get('score', 0)
            else:
                continue # Skip if format is unexpected

            if not move: # If inner move is empty
                continue
            
            is_selected = (play_moves_drawn_count == selected_index) 
            color = HINT_SELECTED_COLOR if is_selected else HINT_NORMAL_COLOR
            rect = pygame.Rect(dialog_x + padding, y_pos, dialog_width - 2 * padding, line_height)
            pygame.draw.rect(screen, color, rect)
            #pygame.draw.rect(screen, BLACK, rect, 1)

            word = move.get('word', 'N/A')
            start_pos_move = move.get('start', (0,0)) # Renamed to avoid conflict
            direction_move = move.get('direction', 'right') # Renamed
            leave = move.get('leave', [])
            word_display = move.get('word_with_blanks', word.upper()) 
            coord = get_coord(start_pos_move, direction_move)
            leave_str = ''.join(sorted(l if l != ' ' else '?' for l in leave))
            
            leave_val = 0.0
            try:
                leave_val = evaluate_leave_cython(leave) 
            except Exception as e_leave:
                print(f"Error evaluating leave in hint dialog: {e_leave}")
                leave_val = 0.0

            text_str = ""
            if is_simulation_result:
                avg_opp_score = move.get('avg_opp_score', 0.0) 
                text_str = f"{play_moves_drawn_count+1}. {word_display} ({raw_score}{leave_val:+0.1f}-{avg_opp_score:.1f}={final_score_sim:.1f}) L:{leave_str}"
            else: # Standard hint
                text_str = f"{play_moves_drawn_count+1}. {word_display} ({raw_score} pts) at {coord} (LVal:{leave_val:+.1f} L:{leave_str})"
            
            text_surf = ui_font.render(text_str, True, BLACK)
            max_text_width = rect.width - 10 
            if text_surf.get_width() > max_text_width:
                avg_char_width = text_surf.get_width() / len(text_str) if len(text_str) > 0 else 10
                if avg_char_width > 0:
                    max_chars = int(max_text_width / avg_char_width) - 3 
                    if max_chars < 5: max_chars = 5
                    text_str = text_str[:max_chars] + "..."
                    text_surf = ui_font.render(text_str, True, BLACK)
            
            screen.blit(text_surf, (rect.x + 5, rect.y + (line_height - text_surf.get_height()) // 2))
            hint_rects.append((rect, play_moves_drawn_count)) # Store rect and its effective index
            y_pos += line_height
            play_moves_drawn_count += 1

    # --- Draw Exchange Option (Appended to List) ---
    exchange_hint_rect = None # Initialize
    if add_exchange_option:
        exchange_index_in_dialog = play_moves_drawn_count # Index after play moves
        is_selected = (exchange_index_in_dialog == selected_index)
        color = HINT_SELECTED_COLOR if is_selected else GRAY 
        rect = pygame.Rect(dialog_x + padding, y_pos, dialog_width - 2 * padding, line_height)
        pygame.draw.rect(screen, color, rect)
        #pygame.draw.rect(screen, BLACK, rect, 1)
        exchange_hint_rect = rect # Store the rect for this option

        exchange_str_display = "".join(sorted(t if t != ' ' else '?' for t in best_exchange_tiles))
        exchange_text = f"{exchange_index_in_dialog + 1}. EXCHANGE: {exchange_str_display} (Eval: {best_exchange_score:.1f})"
        exchange_surf = ui_font.render(exchange_text, True, BLACK)
        
        max_text_width = rect.width - 10
        if exchange_surf.get_width() > max_text_width:
            avg_char_width = exchange_surf.get_width() / len(exchange_text) if len(exchange_text) > 0 else 10
            if avg_char_width > 0:
                max_chars = int(max_text_width / avg_char_width) - 3
                if max_chars < 5: max_chars = 5
                exchange_text = exchange_text[:max_chars] + "..."
                exchange_surf = ui_font.render(exchange_text, True, BLACK)

        screen.blit(exchange_surf, (rect.x + 5, rect.y + (line_height - exchange_surf.get_height()) // 2))
        hint_rects.append((rect, exchange_index_in_dialog)) # Store rect and its effective index
        y_pos += line_height

    if not moves and not add_exchange_option: # No plays and no exchange
        no_options_text = ui_font.render("No valid moves or exchanges found.", True, BLACK)
        screen.blit(no_options_text, (dialog_x + (dialog_width - no_options_text.get_width()) // 2, y_pos + 5))


    # Buttons
    button_y_pos = dialog_y + dialog_height - BUTTON_HEIGHT - padding # Position relative to new height
    play_exch_button_width = 120 
    other_button_width = BUTTON_WIDTH 
    
    total_buttons_width = play_exch_button_width + other_button_width * 2 + BUTTON_GAP * 2
    button_start_x = dialog_x + (dialog_width - total_buttons_width) // 2

    play_button_rect = pygame.Rect(button_start_x, button_y_pos, play_exch_button_width, BUTTON_HEIGHT)
    all_words_button_rect = pygame.Rect(play_button_rect.right + BUTTON_GAP, button_y_pos, other_button_width, BUTTON_HEIGHT)
    ok_button_rect = pygame.Rect(all_words_button_rect.right + BUTTON_GAP, button_y_pos, other_button_width, BUTTON_HEIGHT)

    # Draw Play/Exchange Button
    hover_play = play_button_rect.collidepoint(pygame.mouse.get_pos())
    color_play = BUTTON_HOVER if hover_play else BUTTON_COLOR
    pygame.draw.rect(screen, color_play, play_button_rect)
    play_text = button_font.render("Play/Exchange", True, BLACK)
    screen.blit(play_text, play_text.get_rect(center=play_button_rect.center))

    # Draw All Words Button
    hover_all_words = all_words_button_rect.collidepoint(pygame.mouse.get_pos())
    color_all_words = BUTTON_HOVER if hover_all_words else BUTTON_COLOR
    pygame.draw.rect(screen, color_all_words, all_words_button_rect)
    all_words_text = button_font.render("All Words", True, BLACK)
    screen.blit(all_words_text, all_words_text.get_rect(center=all_words_button_rect.center))

    # Draw OK Button
    hover_ok = ok_button_rect.collidepoint(pygame.mouse.get_pos())
    color_ok = BUTTON_HOVER if hover_ok else BUTTON_COLOR
    pygame.draw.rect(screen, color_ok, ok_button_rect)
    ok_text = button_font.render("OK", True, BLACK)
    screen.blit(ok_text, ok_text.get_rect(center=ok_button_rect.center))

    return {
        'play_button_rect': play_button_rect,       # For "Play/Exchange"
        'all_words_button_rect': all_words_button_rect,
        'ok_button_rect': ok_button_rect,
        'hint_rects': hint_rects  # This is a list of (rect, index) for individual hint items
    }










def draw_all_words_dialog(moves, selected_index, current_scroll_offset):
    #screen.fill(DIALOG_COLOR) # Fill background for the dialog
    dialog_x = (WINDOW_WIDTH - ALL_WORDS_DIALOG_WIDTH) // 2
    dialog_y = (WINDOW_HEIGHT - ALL_WORDS_DIALOG_HEIGHT) // 2

    # Dialog Border
    pygame.draw.rect(screen, BLACK, (dialog_x - 2, dialog_y - 2, ALL_WORDS_DIALOG_WIDTH + 4, ALL_WORDS_DIALOG_HEIGHT + 4), 2)
    pygame.draw.rect(screen, DIALOG_COLOR, (dialog_x, dialog_y, ALL_WORDS_DIALOG_WIDTH, ALL_WORDS_DIALOG_HEIGHT))

    header_height = 40
    unique_words_count = len(set(move.get('word', '') for move in moves if move.get('word')))
    title_text_str = f"All Valid Moves ({unique_words_count} unique words, {len(moves)} plays)"
    title_text = dialog_font.render(title_text_str, True, BLACK)
    screen.blit(title_text, (dialog_x + 10, dialog_y + 10))

    content_area_y = dialog_y + header_height
    button_area_height = BUTTON_HEIGHT + 30 # Padding above and below buttons
    content_area_height = ALL_WORDS_DIALOG_HEIGHT - header_height - button_area_height
    
    # Create a surface for the scrollable content
    # Ensure width is at least 1 to avoid Surface constructor error
    content_surface_width = max(1, ALL_WORDS_DIALOG_WIDTH - 20) # 10px padding on each side
    content_height = len(moves) * 30 # Each item is 30px high
    content_surface = pygame.Surface((content_surface_width, content_height))
    content_surface.fill(DIALOG_COLOR) # Fill with dialog background

    all_words_rects = [] # This will store (screen_rect, original_index) for clickable items
    item_height = 30

    for i, move_data in enumerate(moves):
        y_pos_on_surface = i * item_height
        
        # Basic culling: Don't draw if entirely outside the visible scroll area on the surface
        if y_pos_on_surface + item_height < current_scroll_offset or \
           y_pos_on_surface > current_scroll_offset + content_area_height:
            all_words_rects.append((None, i)) # Add placeholder for indexing if needed
            continue

        move = move_data # If moves contains dicts directly
        
        color = HINT_SELECTED_COLOR if i == selected_index else HINT_NORMAL_COLOR
        rect_on_surface = pygame.Rect(10, y_pos_on_surface, content_surface_width - 20, item_height)
        pygame.draw.rect(content_surface, color, rect_on_surface)
        #pygame.draw.rect(content_surface, BLACK, rect_on_surface, 1) # Border for each item

        word = move.get('word', 'N/A')
        score = move.get('score', 0)
        start_pos = move.get('start', (0,0))
        direction = move.get('direction', 'right')
        leave = move.get('leave', [])
        word_display = move.get('word_with_blanks', word.upper()) # Use formatted word

        coord = get_coord(start_pos, direction)
        leave_str = ''.join(sorted(l if l != ' ' else '?' for l in leave))
        text_str = f"{i+1}. {word_display} ({score} pts) at {coord} ({leave_str})"
        text = ui_font.render(text_str, True, BLACK)

        # Truncate text if too wide
        max_text_width = rect_on_surface.width - 10 
        if text.get_width() > max_text_width:
            avg_char_width = text.get_width() / len(text_str) if len(text_str) > 0 else 10
            if avg_char_width > 0:
                 max_chars = int(max_text_width / avg_char_width) - 3 # -3 for "..."
                 if max_chars < 5: max_chars = 5 
                 text_str = text_str[:max_chars] + "..."
                 text = ui_font.render(text_str, True, BLACK) 

        content_surface.blit(text, (rect_on_surface.x + 5, rect_on_surface.y + (item_height - text.get_height()) // 2))
        
        # Calculate screen rect for collision detection (relative to main screen)
        # This needs to be done carefully if we want to store screen-relative rects
        # For now, all_words_rects will store surface-relative rects for simplicity if scrolling is handled by blitting a sub-part
        # However, for click detection, we need screen-relative.
        
        # Calculate the position of this item as it would appear on the main screen
        screen_y = content_area_y + y_pos_on_surface - current_scroll_offset
        screen_rect = pygame.Rect(dialog_x + 10, screen_y, content_surface_width - 20, item_height)

        # Clip the screen_rect to the visible content area of the dialog
        visible_top = content_area_y
        visible_bottom = content_area_y + content_area_height
        
        clipped_top = max(visible_top, screen_rect.top)
        clipped_bottom = min(visible_bottom, screen_rect.bottom)

        if clipped_bottom > clipped_top: # If any part of it is visible
            clipped_rect = pygame.Rect(screen_rect.left, clipped_top, screen_rect.width, clipped_bottom - clipped_top)
            all_words_rects.append((clipped_rect, i)) # Store the visible part's rect and original index
        else:
            all_words_rects.append((None, i)) # Placeholder if not visible

    # Blit the visible portion of the content surface
    source_area_on_surface = pygame.Rect(0, current_scroll_offset, content_surface_width, content_area_height)
    screen.blit(content_surface, (dialog_x + 10, content_area_y), source_area_on_surface)
    
    # Optional: Draw a border around the content area itself
    pygame.draw.rect(screen, BLACK, (dialog_x + 10, content_area_y, content_surface_width, content_area_height), 1)


    # Buttons
    total_button_width = 2 * BUTTON_WIDTH + BUTTON_GAP
    buttons_x = dialog_x + (ALL_WORDS_DIALOG_WIDTH - total_button_width) // 2
    button_y = dialog_y + ALL_WORDS_DIALOG_HEIGHT - BUTTON_HEIGHT - 20 # Positioned near bottom

    play_button_rect = pygame.Rect(buttons_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    ok_button_rect = pygame.Rect(buttons_x + BUTTON_WIDTH + BUTTON_GAP, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)

    # Draw Play Button
    hover_play = play_button_rect.collidepoint(pygame.mouse.get_pos())
    color_play = BUTTON_HOVER if hover_play else BUTTON_COLOR
    pygame.draw.rect(screen, color_play, play_button_rect)
    play_text = button_font.render("Play", True, BLACK)
    screen.blit(play_text, play_text.get_rect(center=play_button_rect.center))

    # Draw OK Button
    hover_ok = ok_button_rect.collidepoint(pygame.mouse.get_pos())
    color_ok = BUTTON_HOVER if hover_ok else BUTTON_COLOR
    pygame.draw.rect(screen, color_ok, ok_button_rect)
    ok_text = button_font.render("OK", True, BLACK)
    screen.blit(ok_text, ok_text.get_rect(center=ok_button_rect.center))

    return {
        'all_words_play_rect': play_button_rect, 
        'all_words_ok_rect': ok_button_rect,
        'all_words_clickable_items': all_words_rects, 
        'all_words_content_height': content_height
    }



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



def calculate_final_scores(current_scores, racks, bag):
    """
    Calculates the final scores based on game end conditions and remaining tiles.
    Suppresses print statements during batch mode.

    Args:
        current_scores (list): List containing the scores of [Player1, Player2] before adjustments.
        racks (list): List containing the tile lists for [Player1's rack, Player2's rack].
        bag (list): The list representing the tile bag.

    Returns:
        list: A new list containing the final adjusted scores for [Player1, Player2].
    """
    # --- Access global batch status ---
    global is_batch_running

    final_scores = list(current_scores) # Start with a copy of current scores
    rack_values = [0, 0]

    # Calculate the value of tiles left on each rack (blanks count as 0)
    for i in range(2):
        if i < len(racks) and racks[i]: # Check if rack exists and is not None
             rack_values[i] = sum(TILE_DISTRIBUTION[tile][1] for tile in racks[i] if tile != ' ')
        # else: rack_values[i] remains 0

    # Determine if a player went out (must have empty rack AND bag must be empty)
    p1_out = (len(racks) > 0 and racks[0] is not None and not racks[0]) and (not bag)
    p2_out = (len(racks) > 1 and racks[1] is not None and not racks[1]) and (not bag)


    if p1_out:
        if not is_batch_running: print("Final Score Adjust: P1 went out.")
        adjustment = 2 * rack_values[1]
        final_scores[0] += adjustment
        if not is_batch_running: print(f"  P2 tiles value: {rack_values[1]}, P1 adjustment: +{adjustment}")
    elif p2_out:
        if not is_batch_running: print("Final Score Adjust: P2 went out.")
        adjustment = 2 * rack_values[0]
        final_scores[1] += adjustment
        if not is_batch_running: print(f"  P1 tiles value: {rack_values[0]}, P2 adjustment: +{adjustment}")
    else:
        if not is_batch_running: print("Final Score Adjust: Neither player went out.")
        final_scores[0] -= rack_values[0]
        final_scores[1] -= rack_values[1]
        if not is_batch_running: print(f"  P1 adjustment: -{rack_values[0]}, P2 adjustment: -{rack_values[1]}")

    if not is_batch_running: print(f"  Scores before: {current_scores}, Scores after: {final_scores}")
    return final_scores







# In Scrabble Game.py
def play_hint_move(move, current_scores, current_racks, player_idx, current_tiles, current_blanks, 
                   game_board_layout, current_bag, 
                   is_first_play, current_turn, # This is the 9th argument
                   board_tile_counts, # This Counter object will be modified IN-PLACE
                   blanks_played_count): # This is the integer game total BEFORE this move
    """
    Plays a given move (typically from AI or hint) and updates the game state.
    Returns a tuple of all modified game state elements.
    Modifies board_tile_counts IN-PLACE.
    Returns updated blanks_played_count.
    """
    # Make copies of mutable objects that should not be modified if the original is needed later by caller
    scores_copy = list(current_scores)
    racks_copy = [list(r) if r is not None else [] for r in current_racks] # Deep copy for racks
    tiles_copy = [list(row) if row is not None else [] for row in current_tiles] # Deep copy for tiles
    blanks_copy = set(current_blanks) # Copy for blanks
    bag_copy = list(current_bag) # Copy for bag
    
    # These are passed by value or are immutable, direct use is fine for reading
    opponent_idx = 1 - player_idx
    
    # MODIFICATION START: Debug current_rack before Counter creation
    current_rack = racks_copy[player_idx]
    print(f"DEBUG play_hint_move: current_rack before Counter: {current_rack}")
    for i_phm, tile_phm in enumerate(current_rack):
        print(f"DEBUG play_hint_move: current_rack[{i_phm}] = {tile_phm} (type: {type(tile_phm)})")
    # MODIFICATION END

    # Log rack before modification for SGS
    temp_rack_for_sgs_logging = Counter(current_rack) # This is where the TypeError occurs
    
    newly_placed_details = move.get('newly_placed', []) # List of (r, c, letter)
    move_blanks_coords = move.get('blanks', set()) # Set of (r,c) for blanks in this move
    score = move.get('score', 0)
    is_bingo = move.get('is_bingo', False)
    word_with_blanks_phm = move.get('word_with_blanks', '')
    start_pos = move.get('start', (0,0))
    direction = move.get('direction', 'right')
    coord_phm = get_coord(start_pos, direction)
    positions_phm = move.get('positions', []) # Full word positions

    blanks_just_played_this_move = 0
    tiles_placed_from_rack_sgs = [] 
    blanks_played_info_sgs = []     

    for r_placed, c_placed, letter_placed in newly_placed_details:
        if 0 <= r_placed < GRID_SIZE and 0 <= c_placed < GRID_SIZE:
            tiles_copy[r_placed][c_placed] = letter_placed
            board_tile_counts[letter_placed] += 1 
            is_this_placement_a_blank = (r_placed, c_placed) in move_blanks_coords
            if is_this_placement_a_blank:
                blanks_copy.add((r_placed, c_placed))
                blanks_just_played_this_move += 1
                blanks_played_info_sgs.append({'coord': (r_placed, c_placed), 'assigned_letter': letter_placed})

            tile_to_log_removed = ' ' if is_this_placement_a_blank else letter_placed
            if temp_rack_for_sgs_logging[tile_to_log_removed] > 0:
                tiles_placed_from_rack_sgs.append(tile_to_log_removed)
                temp_rack_for_sgs_logging[tile_to_log_removed] -=1
            elif temp_rack_for_sgs_logging[' '] > 0 and not is_this_placement_a_blank:
                 # This case might indicate a blank from rack was used for a non-blank,
                 # but newly_placed had the final letter and move_blanks_coords didn't mark it.
                 # Should be rare if move data is consistent.
                 # For logging, assume ' ' was removed from rack if tile_to_log_removed wasn't found but ' ' was.
                tiles_placed_from_rack_sgs.append(' ')
                temp_rack_for_sgs_logging[' '] -=1


    scores_copy[player_idx] += score
    if is_bingo:
        scores_copy[player_idx] += 50

    rack_counter_current_play = Counter(current_rack) # Operate on the original current_rack for removal
    for r_placed, c_placed, letter_placed in newly_placed_details: # Iterate over tiles placed from rack
        tile_removed_from_rack = ' ' if (r_placed, c_placed) in move_blanks_coords else letter_placed
        if rack_counter_current_play[tile_removed_from_rack] > 0:
            rack_counter_current_play[tile_removed_from_rack] -= 1
        # else: This indicates an error - trying to play a tile not on the rack. Handled by GADDAG.
    
    racks_copy[player_idx] = list(rack_counter_current_play.elements()) # Update the specific player's rack in the copy

    num_to_draw = len(newly_placed_details)
    drawn_tiles = []
    if bag_copy: 
        drawn_tiles = [bag_copy.pop(0) for _ in range(num_to_draw) if bag_copy]
    racks_copy[player_idx].extend(drawn_tiles)

    updated_blanks_played_count = blanks_played_count + blanks_just_played_this_move
    
    # For SGS logging, these are specific to this move
    sgs_data_for_move = {
        'player': current_turn, 'move_type': 'place', 'score': score + (50 if is_bingo else 0),
        'word': move.get('word', 'N/A'), 'positions': positions_phm,
        'blanks_coords_on_board_this_play': list(move_blanks_coords), # Blanks specific to this play
        'coord': coord_phm,
        'leave': racks_copy[player_idx][:], # Rack AFTER play and draw
        'is_bingo': is_bingo, 
        'turn_duration': 0.0, # Placeholder, can be calculated by caller
        'word_with_blanks': word_with_blanks_phm,
        'newly_placed_details': newly_placed_details,
        'tiles_placed_from_rack': tiles_placed_from_rack_sgs,
        'blanks_played_info': blanks_played_info_sgs,
        'rack_before_move': current_rack[:], # Copy of rack before modification
        'tiles_drawn_after_move': drawn_tiles, 'exchanged_tiles':[],
        # Luck factor will be calculated by the caller using this returned state
        'next_turn_after_move': 3 - current_turn,
        'pass_count_after_move': 0, 
        'exchange_count_after_move': 0,
        'consecutive_zero_after_move': 0,
        'is_first_play_after_move': False, # After a play, it's no longer the first play
         # board_tile_counts is modified in-place, blanks_played_count returned
    }
    # Caller will add luck_factor, initial_bag/racks, board_tile_counts_after_move, blanks_played_count_after_move

    last_played_coords_for_return = set((pos[0], pos[1]) for pos in positions_phm if pos)

    return (tuple(scores_copy), racks_copy[player_idx], racks_copy[opponent_idx], 
            tiles_copy, blanks_copy, bag_copy, 
            False, 3 - current_turn, # New is_first_play, new_turn
            board_tile_counts, # Modified in place
            updated_blanks_played_count, # New total
            sgs_data_for_move, # Move specific data for logging
            last_played_coords_for_return)





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


# Function to Replace: reset_game_state
# REASON: Remove print statement for silent batch mode.

def reset_game_state(initial_config):
    """
    Resets game variables for a new game in a batch and returns the new state.
    Initializes board_tile_counts.
    Removes current_turn_pool_quality_score.
    Ensures blanks_played_count is initialized and returned.
    Removed print statement for silent batch mode.
    """
    # --- REMOVED PRINT STATEMENT ---
    # print("--- Resetting Game State for New Batch Game ---")
    # --- END REMOVED PRINT STATEMENT ---

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
    local_blanks_played_count = 0 # Initialize counter for reset

    # Create and shuffle a new bag
    local_bag = create_standard_bag()
    random.shuffle(local_bag)

    # Deal new racks
    local_racks = [[], []]
    try:
        local_racks[0] = [local_bag.pop() for _ in range(7)]
        local_racks[1] = [local_bag.pop() for _ in range(7)]
    except IndexError:
        print("Error: Not enough tiles in bag for initial deal.") # Keep error print
        return None # Indicate failure by returning None

    # Sort racks based on initial config (human vs AI)
    is_ai_config = initial_config.get('is_ai', [False, False])
    for i, rack in enumerate(local_racks):
        if 0 <= i < len(is_ai_config) and not is_ai_config[i]:
            rack.sort()
        # Optionally sort AI racks too: rack.sort()

    # Return 16 values
    return (new_board, new_tiles, local_racks, local_blanks, local_scores,
            local_turn, local_first_play, local_bag, local_move_history,
            local_pass_count, local_exchange_count, local_consecutive_zero_point_turns,
            local_last_played_highlight_coords, local_is_solving_endgame,
            local_board_tile_counts, local_blanks_played_count)







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




# Function to Replace: save_batch_statistics
# REASON: Add tracking and printing of highest/lowest probability bingos.

def save_batch_statistics(batch_results, player_names, batch_summary_filename):
    """
    Calculates aggregate stats and saves batch results to a specified summary file,
    including detailed bingo info, quadrant counts, GCG filename, game duration,
    luck factor, vertical bingo quartile analysis, and details on the
    highest/lowest probability bingos played. Also includes total batch duration and luck.
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

    # --- Initialize Quartile Counters ---
    bingo_quartiles_7 = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    bingo_quartiles_8 = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    total_7_bingos_ranked = 0
    total_8_bingos_ranked = 0
    # --- End Initialize Quartile Counters ---

    # --- Initialize Min/Max Bingo Trackers ---
    min_max_bingo_info = {
        7: {'min_idx': float('inf'), 'min_word': 'N/A', 'min_game': -1, 'min_file': 'N/A',
            'max_idx': -1,          'max_word': 'N/A', 'max_game': -1, 'max_file': 'N/A'},
        8: {'min_idx': float('inf'), 'min_word': 'N/A', 'min_game': -1, 'min_file': 'N/A',
            'max_idx': -1,          'max_word': 'N/A', 'max_game': -1, 'max_file': 'N/A'}
    }
    # --- End Initialize Min/Max Bingo Trackers ---


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

    # Iterate through games to calculate combined index, power tile scores, QUARTILES, and MIN/MAX bingos
    for game in batch_results:
        game_move_history = game.get('move_history', [])
        game_number = game.get('game_number', -1)
        gcg_filename = game.get('gcg_filename', 'N/A')
        first_power_used_in_game = set() # Track first use *within this game*

        for move in game_move_history:
            if move.get('move_type') == 'place':
                word = move.get('word', '').upper()
                score = move.get('score', 0)
                word_len = len(word)

                # Calculate Bingo Index, Quartiles, and Min/Max
                if move.get('is_bingo', False):
                    index = None
                    word_list = None
                    quartile_dict = None
                    min_max_tracker = None # Tracker for this length

                    if word_len == 7 and seven_letter_words:
                        word_list = seven_letter_words
                        quartile_dict = bingo_quartiles_7
                        min_max_tracker = min_max_bingo_info[7]
                    elif word_len == 8 and eight_letter_words:
                        word_list = eight_letter_words
                        quartile_dict = bingo_quartiles_8
                        min_max_tracker = min_max_bingo_info[8]

                    if word_list and min_max_tracker:
                        index = get_word_index(word, word_list) # Returns 1-based index or None
                        if index is not None:
                            total_bingo_index_sum += index
                            bingos_with_index_count += 1
                            # Increment total ranked counter for the specific length
                            if word_len == 7: total_7_bingos_ranked += 1
                            elif word_len == 8: total_8_bingos_ranked += 1

                            # Calculate Quartile
                            list_len = len(word_list)
                            q1_limit = list_len / 4.0
                            q2_limit = list_len / 2.0
                            q3_limit = 3 * list_len / 4.0

                            if index <= q1_limit: quartile_dict['Q1'] += 1
                            elif index <= q2_limit: quartile_dict['Q2'] += 1
                            elif index <= q3_limit: quartile_dict['Q3'] += 1
                            else: quartile_dict['Q4'] += 1

                            # Check for Min Index
                            if index < min_max_tracker['min_idx']:
                                min_max_tracker['min_idx'] = index
                                min_max_tracker['min_word'] = word # Store the word
                                min_max_tracker['min_game'] = game_number
                                min_max_tracker['min_file'] = gcg_filename

                            # Check for Max Index
                            if index > min_max_tracker['max_idx']:
                                min_max_tracker['max_idx'] = index
                                min_max_tracker['max_word'] = word # Store the word
                                min_max_tracker['max_game'] = game_number
                                min_max_tracker['max_file'] = gcg_filename
                        # else: word not found in list, cannot rank or track min/max

                # --- Calculate Power Tile First Play Scores (Existing Logic) ---
                power_in_word = {char for char in word if char in power_tiles}
                for pt in power_in_word:
                    if pt not in first_power_used_in_game:
                        power_tile_scores[pt] += score
                        power_tile_counts[pt] += 1
                        first_power_used_in_game.add(pt) # Mark as used for this game
                # --- END Power Tile Logic ---

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

    # --- Generate Vertical Histogram Strings ---
    hist_lines_7 = create_vertical_histogram(bingo_quartiles_7, total_7_bingos_ranked, "7-Letter Bingo Quartiles")
    hist_lines_8 = create_vertical_histogram(bingo_quartiles_8, total_8_bingos_ranked, "8-Letter Bingo Quartiles")
    # --- End Generate Vertical Histogram Strings ---

    # Write to file
    try:
        with open(batch_summary_filename, "w") as f: # Use the parameter here
            f.write(f"--- Batch Game Results ---\n")
            f.write(f"Total Games: {num_games}\n")
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
            f.write(f"Total Luck Factor: {agg_stats['total_p1_luck']:>+12.2f} {agg_stats['total_p2_luck']:>+12.2f}\n")
            f.write(f"Avg Luck / Game:   {p1_avg_luck_per_game:>+12.2f} {p2_avg_luck_per_game:>+12.2f}\n")
            f.write("-" * 25 + "\n")
            f.write("Aggregate Bingo Index (7/8 Letter Words):\n")
            f.write(f"  Avg Index (Combined): {aggregate_avg_bingo_index:>6.1f}  (Based on {bingos_with_index_count} bingos)\n")

            # --- ADD MIN/MAX BINGO INFO ---
            f.write("\n") # Blank line before min/max info
            # 7-Letter
            f.write("  7-Letter Bingos:\n")
            if min_max_bingo_info[7]['min_idx'] != float('inf'):
                info = min_max_bingo_info[7]
                f.write(f"    Highest Probability: {info['min_word']} ({info['min_idx']}) - Game {info['min_game']} ({info['min_file']})\n")
            else:
                f.write("    Highest Probability: N/A\n")
            if min_max_bingo_info[7]['max_idx'] != -1:
                info = min_max_bingo_info[7]
                f.write(f"    Lowest Probability:  {info['max_word']} ({info['max_idx']}) - Game {info['max_game']} ({info['max_file']})\n")
            else:
                f.write("    Lowest Probability:  N/A\n")
            # 8-Letter
            f.write("  8-Letter Bingos:\n")
            if min_max_bingo_info[8]['min_idx'] != float('inf'):
                info = min_max_bingo_info[8]
                f.write(f"    Highest Probability: {info['min_word']} ({info['min_idx']}) - Game {info['min_game']} ({info['min_file']})\n")
            else:
                f.write("    Highest Probability: N/A\n")
            if min_max_bingo_info[8]['max_idx'] != -1:
                info = min_max_bingo_info[8]
                f.write(f"    Lowest Probability:  {info['max_word']} ({info['max_idx']}) - Game {info['max_game']} ({info['max_file']})\n")
            else:
                f.write("    Lowest Probability:  N/A\n")
            # --- END MIN/MAX BINGO INFO ---

            f.write("-" * 25 + "\n")
            f.write("Bingo Quartile Ranks (Based on Word List Position):\n\n") # Add extra newline
            # 7-Letter Bingos
            for line in hist_lines_7: # Write histogram lines (includes title)
                f.write(line + "\n")
            f.write("\n") # Add blank line between histograms
            # 8-Letter Bingos
            for line in hist_lines_8: # Write histogram lines (includes title)
                f.write(line + "\n")

            f.write("-" * 25 + "\n")
            f.write("Power Tile First Play Scores (Aggregate Avg):\n")
            for pt in sorted(power_tiles): # Sort J, Q, X, Z for consistent order
                 count = power_tile_counts[pt]
                 avg_score = avg_power_scores[pt]
                 f.write(f"  {pt}: {avg_score:>10.2f}  (Based on {count} plays)\n")
            f.write("-" * 25 + "\n")
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
                game_duration_str = format_duration(game.get('game_duration_seconds', 0.0))
                f.write(f"  Duration: {game_duration_str}\n")
                p1_luck = game.get('player1_total_luck', 0.0)
                p2_luck = game.get('player2_total_luck', 0.0)
                f.write(f"  Luck Factor: P1={p1_luck:+.2f}, P2={p2_luck:+.2f}\n")
                quad_counts = game.get('quadrant_counts', {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0})
                f.write(f"  Quadrants: Q2={quad_counts['Q2']}, Q1={quad_counts['Q1']}, Q3={quad_counts['Q3']}, Q4={quad_counts['Q4']}\n")
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








# Function to Replace: create_vertical_histogram
# REASON: Add title and y-axis labels to the vertical histogram output.

def create_vertical_histogram(quartile_counts, total_ranked, title, max_height=10):
    """
    Creates a multi-line string representation of a vertical histogram
    for quartiles, including a title and y-axis labels.

    Args:
        quartile_counts (dict): Dictionary with counts for 'Q1', 'Q2', 'Q3', 'Q4'.
        total_ranked (int): The total number of bingos included in the counts.
        title (str): The title to print above the histogram.
        max_height (int): The maximum desired height of the histogram bars in characters.

    Returns:
        list: A list of strings, each representing a line of the histogram.
              Returns list with title and message if total_ranked is 0.
    """
    lines = []
    # Add title centered (approximately)
    title_padding = "  " # Initial indent matches histogram body
    lines.append(title_padding + title)
    lines.append("") # Add a blank line after title

    if total_ranked == 0:
        lines.append("  (No ranked bingos)")
        return lines

    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']

    # Find the maximum count in any quartile
    max_count = 0
    for q in quartiles:
        max_count = max(max_count, quartile_counts.get(q, 0))

    if max_count == 0:
        lines.append("  (No counts in quartiles)")
        return lines

    # Determine the scale factor
    scale = math.ceil(max_count / max_height) if max_count > max_height else 1
    scale = max(1, scale) # Ensure scale is at least 1

    # Calculate scaled heights for each bar
    scaled_heights = {}
    max_scaled_height = 0
    for q in quartiles:
        # Use ceil to ensure even a small count gets at least one '*' if scale is large
        height = math.ceil(quartile_counts.get(q, 0) / scale)
        scaled_heights[q] = height
        max_scaled_height = max(max_scaled_height, height)

    # --- Y-Axis Label Calculation ---
    max_value_on_axis = max_scaled_height * scale
    # Determine number of labels (aim for ~5, but fewer if height is small)
    num_labels = min(max_scaled_height + 1, 6) # +1 for 0 label
    label_interval_h = max(1, max_scaled_height // (num_labels - 1)) if num_labels > 1 else 1

    # Find max label width for alignment
    label_width = len(str(max_value_on_axis))
    y_axis_prefix_width = label_width + 2 # width + space + separator (| or +)

    # --- Build Histogram Lines with Y-Axis ---
    for h in range(max_scaled_height, 0, -1):
        # Determine if a label should be printed for this row height
        label_str = ""
        # Print label at the top and at intervals
        if h == max_scaled_height or h % label_interval_h == 0:
            label_val = h * scale
            label_str = f"{label_val:>{label_width}} |" # Right align value, add separator
        else:
            label_str = f"{' ' * label_width} |" # Padding + separator

        # Build the bar part of the line
        bar_line = ""
        for q in quartiles:
            if scaled_heights.get(q, 0) >= h:
                bar_line += "*  " # Add asterisk and spacing
            else:
                bar_line += "   " # Add spacing

        lines.append(label_str + bar_line.rstrip()) # Combine label and bars

    # --- Add X-Axis Base ---
    # Add the 0 label line
    lines.append(f"{0:>{label_width}} +{'-' * (len(quartiles) * 3)}") # Base line with 0
    # Add the Q labels, aligned with bars
    lines.append(f"{' ' * y_axis_prefix_width}{'  '.join(quartiles)}")

    # Add the scale note if scale > 1
    if scale > 1:
        lines.append(f"{' ' * y_axis_prefix_width}(Each * represents approx. {scale} bingos)")

    return lines











def estimate_draw_value(num_to_draw, pool_analysis):
        """
        Provides a simple heuristic estimate of the value gained by drawing tiles,
        based on the expected value of a single draw.

        Args:
            num_to_draw (int): The number of tiles to be drawn.
            pool_analysis (dict): The result from analyze_unseen_pool
                                  (expected to contain 'expected_draw_value').

        Returns:
            float: An estimated score adjustment based on expected draw value.
        """
        # Multiply the expected value of one draw by the number of tiles drawn
        expected_single_draw_value = pool_analysis.get('expected_draw_value', 0.0)
        estimated_total_draw_value = expected_single_draw_value * num_to_draw
        # print(f"DEBUG estimate_draw_value: Draw {num_to_draw}, Exp Single: {expected_single_draw_value:.2f}, Est Total: {estimated_total_draw_value:.1f}") # Optional debug
        return estimated_total_draw_value






# In Scrabble Game.py

def find_best_exchange_option(rack, current_tiles_grid, current_blanks_set, game_total_blanks_played, bag_count):
    """
    Finds the best set of tiles to exchange from the rack.
    Args:
        rack (list): The player's current rack.
        current_tiles_grid (list of list): The current state of the board's tiles.
        current_blanks_set (set): The set of (r,c) for blanks on the board.
        game_total_blanks_played (int): Total blanks played in the game so far.
        bag_count (int): Number of tiles currently in the bag.
    Returns:
        tuple: (list_of_tiles_to_exchange, estimated_value_of_exchange_option)
               Returns ([], -float('inf')) if no good exchange is found or cannot exchange.
    """
    # print(f"Debug: find_best_exchange_option called with rack: {''.join(sorted(rack))}, bag_count: {bag_count}")
    if not rack or bag_count == 0: # Cannot exchange if rack is empty or bag is empty
        return [], -float('inf')

    best_overall_exchange_tiles = []
    best_overall_estimated_value = -float('inf')

    # Calculate expected draw value using Cython helper
    # This needs the *current* board state to correctly assess the pool for get_remaining_tiles
    expected_single_draw_value = 0.0
    try:
        expected_single_draw_value = get_expected_draw_value_cython(
            rack,                       # Current rack (pool is unseen relative to this)
            current_tiles_grid,         # Pass the current game's tiles grid
            current_blanks_set,         # Pass the current game's blanks set
            game_total_blanks_played,   # Pass the game's total blanks played
            get_remaining_tiles         # Pass the Python helper function object
        )
    except Exception as e_val:
        print(f"Error in find_best_exchange_option calling get_expected_draw_value_cython: {e_val}")
        # If this fails, we can't reliably estimate draw value, so exchanges are risky.
        # Depending on strategy, could return no exchange or proceed with 0 draw value.
        # For now, proceed with 0, which makes exchanges less likely unless leave is very bad.
        expected_single_draw_value = 0.0

    # Iterate through exchanging k=1 to min(bag_count, len(rack)) tiles
    # We can only exchange as many tiles as are in the bag.
    max_exchange_count = min(len(rack), bag_count)

    for k in range(1, max_exchange_count + 1): # Number of tiles to exchange
        best_leave_score_for_k = -float('inf')
        best_kept_subset_for_k = []
        current_best_exchange_tiles_for_k = []

        num_to_keep = len(rack) - k
        if num_to_keep < 0: continue # Should not happen with loop range

        if num_to_keep == 0: # Exchanging all tiles
            best_leave_score_for_k = 0 # Leave value of an empty rack is 0
            best_kept_subset_for_k = []
            current_best_exchange_tiles_for_k = rack[:] # All tiles are exchanged
        else:
            for kept_subset_tuple in combinations(rack, num_to_keep):
                kept_subset_list = list(kept_subset_tuple)
                current_leave_score = evaluate_leave_cython(kept_subset_list) # evaluate_leave is imported
                if current_leave_score > best_leave_score_for_k:
                    best_leave_score_for_k = current_leave_score
                    best_kept_subset_for_k = kept_subset_list

            # Determine tiles to exchange based on best_kept_subset_for_k
            if best_kept_subset_for_k is not None: # Check if any subset was found (should be if num_to_keep > 0)
                temp_rack_counts = Counter(rack)
                temp_kept_counts = Counter(best_kept_subset_for_k)
                tiles_to_exchange_counts = temp_rack_counts - temp_kept_counts
                current_best_exchange_tiles_for_k = list(tiles_to_exchange_counts.elements())
            else: # Should only occur if num_to_keep was 0 and logic above was missed
                current_best_exchange_tiles_for_k = rack[:]


        # Calculate total estimated value for exchanging these 'k' tiles
        estimated_value_of_draw = expected_single_draw_value * k
        leave_score_after_exchange_and_draw = best_leave_score_for_k # This is the leave of what's *kept*
        
        # The total value of this exchange option is the leave of the tiles kept
        # PLUS the expected value of the tiles drawn.
        total_estimated_value_for_this_k_exchange = leave_score_after_exchange_and_draw + estimated_value_of_draw

        if total_estimated_value_for_this_k_exchange > best_overall_estimated_value:
            best_overall_estimated_value = total_estimated_value_for_this_k_exchange
            best_overall_exchange_tiles = current_best_exchange_tiles_for_k
    
    # print(f"Debug: find_best_exchange_option result: Tiles: {''.join(sorted(best_overall_exchange_tiles))}, Value: {best_overall_estimated_value:.2f}")
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







def negamax_endgame(rack_player, rack_opponent, tiles, blanks, board,
                    current_score_diff, alpha, beta, depth, pass_count,
                    max_depth, search_depth_limit): # Added search_depth_limit
    """
    Negamax solver for the endgame (empty bag) with depth limit.
    Returns (best_score_diff_for_player, best_move_sequence).
    Score difference is from the perspective of the player whose turn it is.
    """
    # --- Access Globals ---
    global GADDAG_STRUCTURE, DAWG # Need these for move generation

    # --- Base Cases: Game Over OR Depth Limit Reached ---
    if depth >= search_depth_limit:
        heuristic_score = evaluate_endgame_heuristic(rack_player, rack_opponent, current_score_diff)
        return heuristic_score, []

    if not rack_player or not rack_opponent or pass_count >= 6:
        final_diff = calculate_endgame_score_diff(rack_player, rack_opponent, current_score_diff)
        return final_diff, []

    # --- Generate Moves ---
    # --- MODIFICATION: Call Cython version ---
    try:
        # Ensure GADDAG/DAWG are available
        if GADDAG_STRUCTURE is None or DAWG is None:
             print("ERROR (negamax): GADDAG/DAWG not available for move generation.")
             possible_moves = []
        else:
             possible_moves = generate_all_moves_gaddag_cython(
                 rack_player, tiles, board, blanks, GADDAG_STRUCTURE.root, DAWG
             )
        if possible_moves is None: possible_moves = []
    except Exception as e_gen:
        print(f"ERROR during move generation in negamax: {e_gen}")
        possible_moves = []
    # --- END MODIFICATION ---


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

        if move == "PASS":
            sim_pass_count += 1
            value, subsequent_sequence = negamax_endgame(
                sim_rack_opponent, sim_rack_player, sim_tiles, sim_blanks, board,
                -sim_score_diff, -beta, -alpha, depth + 1, sim_pass_count, max_depth, search_depth_limit
            )
            value = -value
        else: # It's a play move (dictionary)
            sim_pass_count = 0
            newly_placed_details = move.get('newly_placed', [])
            move_blanks = move.get('blanks', set())
            move_score = move.get('score', 0)

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
                 continue

            sim_rack_player = temp_rack
            sim_score_diff += move_score

            value, subsequent_sequence = negamax_endgame(
                sim_rack_opponent, sim_rack_player, sim_tiles, sim_blanks, board,
                -sim_score_diff, -beta, -alpha, depth + 1, sim_pass_count, max_depth, search_depth_limit
            )
            value = -value

        if value > best_value:
            best_value = value
            best_sequence = [current_move_details] + subsequent_sequence

        alpha = max(alpha, value)
        if alpha >= beta:
            break

    if best_sequence is None:
         final_diff = calculate_endgame_score_diff(rack_player, rack_opponent, current_score_diff)
         return final_diff, []

    return best_value, best_sequence










# Function to Replace: solve_endgame
# REASON: Correct indentation for sequence printing block.

# Function to Replace: solve_endgame
# REASON: Suppress print statements during batch mode.

def solve_endgame(rack_player, rack_opponent, tiles, blanks, board, current_score_diff):
    """
    Top-level function to initiate the endgame solver with a depth limit.
    Calls negamax once and prints the single best sequence found (unless in batch mode).
    Returns the best first move to make.
    """
    global is_solving_endgame, endgame_start_time, is_batch_running # Add is_batch_running

    # --- MODIFICATION: Conditional Printing ---
    if not is_batch_running:
        print("--- Starting Endgame Solver ---")
    # --- END MODIFICATION ---
    is_solving_endgame = True
    endgame_start_time = time.time()

    # Make deep copies to avoid modifying original game state during search
    tiles_copy = copy.deepcopy(tiles)
    blanks_copy = blanks.copy()
    rack_player_copy = rack_player[:]
    rack_opponent_copy = rack_opponent[:]

    search_depth_limit = 6
    max_possible_depth = len(rack_player_copy) + len(rack_opponent_copy)
    actual_search_depth = min(search_depth_limit, max_possible_depth)
    # --- MODIFICATION: Conditional Printing ---
    if not is_batch_running:
        print(f"--- Endgame Search Depth Limit: {actual_search_depth} ---")
    # --- END MODIFICATION ---

    # Initial call to negamax, passing the depth limit
    best_score_diff, best_move_sequence = negamax_endgame(
        rack_player_copy, rack_opponent_copy, tiles_copy, blanks_copy, board,
        current_score_diff, -float('inf'), float('inf'), 0, 0,
        max_possible_depth, actual_search_depth # Pass both max possible and actual limit
    )


    solve_duration = time.time() - endgame_start_time
    # --- MODIFICATION: Conditional Printing ---
    if not is_batch_running:
        print(f"--- Endgame Solver Finished ({solve_duration:.2f}s) ---")
    # --- END MODIFICATION ---
    is_solving_endgame = False # Reset flag

    # --- MODIFICATION: Conditional Printing ---
    if not is_batch_running:
        # Print the single best sequence found
        if best_move_sequence:
            print(f"Best Endgame Sequence Found (Score Diff: {best_score_diff:+.0f} at depth {actual_search_depth}):")
            current_player_idx = 0
            turn_num = 1
            temp_rack_p = rack_player[:]
            temp_rack_o = rack_opponent[:]
            for i, move in enumerate(best_move_sequence):
                player_indicator = "P1►" if current_player_idx == 0 else "P2►"
                rack_to_use = temp_rack_p if current_player_idx == 0 else temp_rack_o
                print(f"  {turn_num}. {player_indicator} {format_move_for_debug(move, rack_to_use)}")
                if move != "PASS":
                     if isinstance(move, dict):
                         newly_placed_count = len(move.get('newly_placed', []))
                         if newly_placed_count == 0 and move.get('score',0) > 0:
                             newly_placed_count = move.get('score',0) // 5
                             print(f"    (Warning: Estimating tiles played for print: {newly_placed_count})")
                current_player_idx = 1 - current_player_idx
                if current_player_idx == 0: turn_num += 1
        else:
            print("Endgame Solver: No optimal sequence found (e.g., immediate game end).")
    # --- END MODIFICATION ---

    if best_move_sequence:
        return best_move_sequence[0] # Return only the first move to make
    else:
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



# Function to Replace: run_ai_simulation
# REASON: Suppress print statements during batch mode.

def run_ai_simulation(ai_rack, opponent_rack_len, tiles, blanks, board, bag, gaddag_root, is_first_play,
                      board_tile_counts, blanks_played_count, # <<< ADDED blanks_played_count parameter
                      # Parameters use global defaults defined earlier
                      num_ai_candidates=DEFAULT_AI_CANDIDATES,
                      num_opponent_sims=DEFAULT_OPPONENT_SIMULATIONS,
                      num_post_sim_candidates=DEFAULT_POST_SIM_CANDIDATES):
    """
    Performs a 2-ply simulation to find the best AI move.
    Simulates top N AI moves, estimates opponent's best response M times for each,
    then evaluates the top K results using the LEAVE_LOOKUP_TABLE (float values).
    Uses board_tile_counts and blanks_played_count for pool calculation. Includes debug prints.
    Removed visual feedback for performance.
    Calls generate_all_moves_gaddag_cython.
    Suppresses prints during batch mode.
    """
    # --- Access Globals ---
    global DAWG, is_batch_running # Need DAWG and batch status

    import time # Ensure time module is available
    # --- MODIFICATION: Conditional Printing ---
    if not is_batch_running:
        print("--- Running AI 2-Ply Simulation ---")
        print(f"  Params: AI Cands={num_ai_candidates}, Opp Sims={num_opponent_sims}, Post Sims={num_post_sim_candidates}")
        print(f"  AI Rack: {''.join(sorted(ai_rack))}, Opp Rack Len: {opponent_rack_len}, Bag: {len(bag)}, Blanks Played: {blanks_played_count}")
        print("  Generating initial AI moves...")
    # --- END MODIFICATION ---

    try:
        if gaddag_root is None or DAWG is None:
             if not is_batch_running: print("ERROR (run_ai_simulation): GADDAG/DAWG not available for move generation.")
             all_ai_moves = []
        else:
             all_ai_moves = generate_all_moves_gaddag_cython(
                 ai_rack, tiles, board, blanks, gaddag_root, DAWG
             )
        if all_ai_moves is None: all_ai_moves = []
    except Exception as e_gen:
        # Keep error prints even in batch
        print(f"ERROR during initial AI move generation in simulation: {e_gen}")
        import traceback
        traceback.print_exc()
        all_ai_moves = []
    # --- MODIFICATION: Conditional Printing ---
    if not is_batch_running:
        print(f"  Generated {len(all_ai_moves) if all_ai_moves else 0} initial AI moves.")
    # --- END MODIFICATION ---

    if not all_ai_moves:
        if not is_batch_running: print(f"  Simulation: No initial AI moves found.")
        return []


    # 1. Select Top N AI moves based on raw score
    all_ai_moves.sort(key=lambda m: m.get('score', 0), reverse=True)
    top_ai_moves_candidates = all_ai_moves[:num_ai_candidates]
    if not is_batch_running: print(f"  Simulating top {len(top_ai_moves_candidates)} AI moves...")

    simulation_results = [] # Store tuples: (ai_move, avg_opponent_score)
    num_simulations_per_move = num_opponent_sims

    # --- Pre-calculate unseen tiles for opponent rack simulation ---
    remaining_dict = get_remaining_tiles(ai_rack, board_tile_counts, blanks_played_count)

    unseen_tiles_pool = []
    for tile, count in remaining_dict.items():
        unseen_tiles_pool.extend([tile] * count)
    if not is_batch_running: print(f"  Unseen Pool Size: {len(unseen_tiles_pool)}")

    for i, ai_move in enumerate(top_ai_moves_candidates):

        move_word = ai_move.get('word_with_blanks', ai_move.get('word', '?'))
        move_score = ai_move.get('score', 0)
        if not is_batch_running: print(f"\n  Simulating AI move {i+1}/{len(top_ai_moves_candidates)}: '{move_word}' ({move_score} pts)")

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
        sim_board_counts_after_ai = board_tile_counts.copy()
        for r, c, letter in ai_move.get('newly_placed', []):
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                sim_tiles_after_ai[r][c] = letter
                sim_board_counts_after_ai[letter] += 1 # Update counter
                if (r, c) in move_blanks:
                    sim_blanks_after_ai.add((r, c))
                    if ' ' in sim_rack_after_ai: sim_rack_after_ai.remove(' ')
                    else: valid_placement = False; break
                else:
                    if letter in sim_rack_after_ai: sim_rack_after_ai.remove(letter)
                    else: valid_placement = False; break
            else: valid_placement = False; break
        if not valid_placement:
            if not is_batch_running: print("    Skipping AI move due to invalid placement simulation.")
            continue
        num_to_draw_ai = newly_placed_count
        drawn_ai = []
        for _ in range(num_to_draw_ai):
            if sim_bag_after_ai: drawn_ai.append(sim_bag_after_ai.pop())
        sim_rack_after_ai.extend(drawn_ai)
        if not is_batch_running: print(f"    State after AI move: Bag={len(sim_bag_after_ai)}")

        # --- Loop M times for opponent simulation ---
        if not is_batch_running: print(f"    Starting {num_simulations_per_move} opponent simulations...")
        for sim_run in range(num_simulations_per_move):
            opponent_available_pool = unseen_tiles_pool[:]
            temp_drawn_ai_counts = Counter(drawn_ai)
            temp_pool_copy = opponent_available_pool[:]
            for tile in temp_pool_copy:
                 if temp_drawn_ai_counts.get(tile, 0) > 0:
                      temp_drawn_ai_counts[tile] -= 1
                      try:
                          opponent_available_pool.remove(tile)
                      except ValueError:
                          # Keep warning
                          print(f"        WARNING: Could not remove tile '{tile}' from opponent pool during adjustment.")
            random.shuffle(opponent_available_pool)
            actual_opponent_rack_len = min(opponent_rack_len, len(opponent_available_pool))
            sim_opponent_rack = opponent_available_pool[:actual_opponent_rack_len]

            try:
                if gaddag_root is None or DAWG is None:
                     if not is_batch_running: print("ERROR (run_ai_simulation opp): GADDAG/DAWG not available.")
                     opponent_moves = []
                else:
                     opponent_moves = generate_all_moves_gaddag_cython(
                         sim_opponent_rack, sim_tiles_after_ai, board, sim_blanks_after_ai, gaddag_root, DAWG
                     )
                if opponent_moves is None: opponent_moves = []
            except Exception as e_gen_opp:
                # Keep error print
                print(f"ERROR during opponent move generation in simulation: {e_gen_opp}")
                opponent_moves = []

            best_opponent_score = 0
            if opponent_moves:
                opponent_moves.sort(key=lambda m: m.get('score', 0), reverse=True)
                best_opponent_score = opponent_moves[0].get('score', 0)
            total_opponent_score_for_this_move += best_opponent_score
        if not is_batch_running: print(f"    Finished opponent simulations for AI move {i+1}.")

        average_opponent_score = total_opponent_score_for_this_move / num_simulations_per_move
        ai_move['avg_opp_score'] = average_opponent_score
        simulation_results.append(ai_move)


    if not simulation_results:
         if not is_batch_running: print(f"  Simulation: No results generated. Falling back.")
         return [{'move': m, 'final_score': m.get('score',0)} for m in top_ai_moves_candidates]


    # 2. Select Top K based on (AI Score - Avg Opponent Score)
    simulation_results.sort(key=lambda r: r.get('score', 0) - r.get('avg_opp_score', 0.0), reverse=True)
    top_sim_results = simulation_results[:num_post_sim_candidates]

    # 3. Apply Leave Evaluation (Lookup) to Top K
    if not is_batch_running: print("--- Evaluating Top Simulation Results with Leave Lookup ---")
    final_evaluated_moves = []
    for move_result in top_sim_results:
        ai_move = move_result
        avg_opp_score = ai_move.get('avg_opp_score', 0.0)
        ai_raw_score = ai_move.get('score', 0)
        leave = ai_move.get('leave', [])
        leave_value = evaluate_leave_cython(leave) # Returns float
        leave_str_eval = "".join(sorted(['?' if tile == ' ' else tile for tile in leave]))
        word_eval = ai_move.get('word_with_blanks', ai_move.get('word', '?'))
        if not is_batch_running: print(f"  Evaluating: {word_eval} ({ai_raw_score} pts), Leave: '{leave_str_eval}', Lookup Value: {leave_value:.2f}")

        final_eval_score = float(ai_raw_score) + leave_value - float(avg_opp_score)
        final_evaluated_moves.append({'move': ai_move, 'final_score': final_eval_score})
    if not is_batch_running: print("-" * 20)

    # 4. Sort the final evaluated moves
    final_evaluated_moves.sort(key=lambda m: m['final_score'], reverse=True)

    # --- Print Top 5 Choices (Debug/Info) ---
    if not is_batch_running:
        print("--- Simulation Top 5 Choices (AI Score + LeaveLookup - Avg Opponent Score) ---")
        for i, evaluated_move in enumerate(final_evaluated_moves[:5]):
            move = evaluated_move['move']
            final_score = evaluated_move['final_score']
            word = move.get('word_with_blanks', move.get('word', '?'))
            coord = get_coord(move.get('start', (0,0)), move.get('direction', '?'))
            raw_score = move.get('score', 0)
            leave_list = move.get('leave', [])
            leave_str = "".join(sorted(l if l != ' ' else '?' for l in leave_list))
            avg_opp = move.get('avg_opp_score', 0.0)
            leave_val = evaluate_leave_cython(leave_list)
            print(f"  {i+1}. {word} at {coord} ({raw_score}) L:'{leave_str}' ({leave_val:.2f}) OppAvg:{avg_opp:.1f} -> Final:{final_score:.1f}")
        print("-" * 20)
        print(f"--- AI Simulation Complete ---")

    return final_evaluated_moves










        

# In Scrabble_Game.py

def get_replay_state(target_turn_idx, move_history_sgs, initial_racks_for_replay):
    """
    Recreate the game state up to target_turn_idx ITERATIVELY using rich SGS move_history.
    Returns board, blanks, scores, racks (sorted), board_counts, blanks_played_total.
    MODIFIED: Refined board_counts and blanks_played_total updates.
    """
    tiles_state = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    blanks_state = set() 
    scores_state = [0, 0]
    board_counts_state = Counter() 
    blanks_played_total_state = 0  

    if not isinstance(initial_racks_for_replay, list) or len(initial_racks_for_replay) != 2 or \
       not isinstance(initial_racks_for_replay[0], list) or \
       not isinstance(initial_racks_for_replay[1], list):
        print(f"Error (get_replay_state): Invalid initial_racks_for_replay format: {initial_racks_for_replay}")
        empty_racks = [[], []]
        for rack in empty_racks: rack.sort()
        # Return default counts as well
        return tiles_state, blanks_state, scores_state, empty_racks, board_counts_state, blanks_played_total_state

    racks_state = [list(initial_racks_for_replay[0]), list(initial_racks_for_replay[1])] 

    for i in range(target_turn_idx):
        if i >= len(move_history_sgs): break 
        
        move = move_history_sgs[i]
        player_idx = move.get('player') 
        if player_idx not in [1, 2]: continue
        player_idx -= 1 

        current_rack_copy = list(racks_state[player_idx]) 
        move_type = move.get('move_type')
        tiles_drawn_this_move = move.get('tiles_drawn_after_move', []) 

        if move_type == 'place':
            newly_placed_this_move = move.get('newly_placed_details', []) 
            blanks_coords_this_play = set(move.get('blanks_coords_on_board_this_play', [])) 
            blanks_played_info_this_move = move.get('blanks_played_info', []) 

            for r_place, c_place, letter_on_board in newly_placed_this_move:
                if 0 <= r_place < GRID_SIZE and 0 <= c_place < GRID_SIZE:
                    # --- Refined Update ---
                    is_empty_before = not tiles_state[r_place][c_place] 
                    tiles_state[r_place][c_place] = letter_on_board 
                    if is_empty_before: # Only count if placed on empty square
                         board_counts_state[letter_on_board] += 1
                    # --- End Refined Update ---
                    if (r_place, c_place) in blanks_coords_this_play:
                        blanks_state.add((r_place, c_place))
            
            # --- Refined Update: Increment total blanks played ONCE per move ---
            blanks_played_total_state += len(blanks_played_info_this_move)
            # --- End Refined Update ---

            tiles_actually_taken_from_rack = move.get('tiles_placed_from_rack', [])
            for tile_to_remove in tiles_actually_taken_from_rack:
                try: current_rack_copy.remove(tile_to_remove)
                except ValueError: 
                    print(f"Replay Warning (Place): Tile '{tile_to_remove}' specified in 'tiles_placed_from_rack' not found in reconstructed rack {current_rack_copy} for move {i}, player {player_idx+1}.")

            scores_state[player_idx] += move.get('score', 0)
            current_rack_copy.extend(tiles_drawn_this_move)
            racks_state[player_idx] = current_rack_copy

        elif move_type == 'exchange':
            tiles_exchanged_sgs = move.get('tiles_exchanged', [])
            for tile_to_remove in tiles_exchanged_sgs:
                try: current_rack_copy.remove(tile_to_remove)
                except ValueError: pass 
            current_rack_copy.extend(tiles_drawn_this_move)
            racks_state[player_idx] = current_rack_copy

        elif move_type == 'pass':
            racks_state[player_idx] = current_rack_copy 
        
    final_racks_sorted = [sorted(rack) for rack in racks_state]

    return (tiles_state, blanks_state, scores_state, final_racks_sorted, 
            board_counts_state, blanks_played_total_state)






def perform_leave_lookup(leave_key_str):
    """
    Performs the lookup in the global LEAVE_LOOKUP_TABLE.
    Called by the Cython function.
    """
    # Access the global table (ensure it's loaded and accessible)
    global LEAVE_LOOKUP_TABLE
    try:
        value = LEAVE_LOOKUP_TABLE.get(leave_key_str)
        if value is not None:
            return float(value)
        else:
            # Optional: Add a print here if you want to see keys failing lookup *in Python*
            # print(f"--- DEBUG (Python Lookup): Key '{leave_key_str}' NOT FOUND.")
            return 0.0
    except Exception as e:
        print(f"Error during Python leave lookup for key '{leave_key_str}': {e}")
        return 0.0













def ai_turn(current_rack_for_ai_logic, tiles, blanks, board, scores, racks, turn, bag, first_play,
            pass_count, exchange_count, consecutive_zero_point_turns,
            player_names, dropdown_open, hinting, showing_all_words,
            letter_checks, number_checks, # For practice modes
            move_history, # For SGS saving and context
            initial_racks_sgs, # For SGS saving
            initial_shuffled_bag_order_sgs, # For SGS saving
            board_tile_counts, # This is the main game's Counter, will be modified by play_hint_move
            current_game_blanks_played_count, # This is the game total BEFORE this move
            last_played_highlight_coords, # To be updated
            is_batch_running=False, # For conditional printing/UI
            use_endgame_solver_setting=False, # To enable endgame solver
            use_ai_simulation_setting=False, # To enable simulation
            practice_mode=None, practice_best_move=None, practice_target_moves=None, # For practice logic
            paused_for_power_tile=False, current_power_tile=None, # For power tile practice
            paused_for_bingo_practice=False, # For bingo bango bongo
            practice_probability_max_index=None # For 8-letter practice
            ):
    """Handles the AI's turn, deciding to play, exchange, or pass."""
    global GADDAG_STRUCTURE, DAWG # Access GADDAG and DAWG
    global USE_CYTHON_AI_TURN_LOGIC, USE_CYTHON_STANDARD_EVALUATION, USE_CYTHON_MOVE_GENERATION
    global EXCHANGE_PREFERENCE_THRESHOLD, MIN_SCORE_TO_AVOID_EXCHANGE
    global is_solving_endgame, endgame_start_time # For UI indicator

    start_turn_time = time.time()
    player_idx = turn - 1
    opponent_idx = 1 - player_idx

    # Store the rack before the move for SGS logging
    rack_before_move_sgs = list(racks[player_idx][:]) # Make a copy

    # Make copies of mutable game state items that might be modified by simulation/play_hint_move
    # but only if the original should not be changed directly by this top-level AI call yet.
    # For board_tile_counts and current_game_blanks_played_count, we pass them in and expect
    # play_hint_move to update them and return the new versions.

    debug_prefix = f"AI {turn}" if not is_batch_running else f"AI {turn} (BATCH)"
    if not is_batch_running:
        print(f"\n{GREEN}{debug_prefix}: Turn started. Rack: {''.join(sorted(current_rack_for_ai_logic))}{RESET_COLOR}")

    ai_paused_for_power_tile_this_turn = paused_for_power_tile # Use passed-in pause state
    ai_current_power_tile_this_turn = current_power_tile
    ai_paused_for_bingo_practice_this_turn = paused_for_bingo_practice

    # Preserve the counts at the start of the turn for luck calculation if a play is made
    board_tile_counts_at_turn_start = board_tile_counts.copy()
    blanks_played_count_at_turn_start = current_game_blanks_played_count

    # --- Pre-computation / GADDAG Check ---
    if gaddag_loading_status != 'loaded' or GADDAG_STRUCTURE is None or DAWG is None:
        if not is_batch_running:
            show_message_dialog("AI cannot make a move: GADDAG or DAWG structure not loaded.", "AI Error")
            print(f"{RED}{debug_prefix}: GADDAG/DAWG not loaded. AI passes by default.{RESET_COLOR}")

        action_chosen_gaddag_fail = 'pass'
        status_reason = "GADDAG/DAWG not loaded" if gaddag_loading_status != 'loaded' or DAWG is None else "GADDAG structure is None"

        # Update game state for a pass
        # Ensure these are the main game state variables being updated if they were passed by reference effectively
        # Or ensure the calling function (handle_ai_turn_trigger) updates them based on return
        current_replay_turn = len(move_history) # Should be up-to-date if called after human
        updated_consecutive_zero = consecutive_zero_point_turns + 1
        updated_pass_count = pass_count + 1
        updated_exchange_count = 0 # Pass resets exchange count
        next_turn_val_gfail = 3 - turn # Turn flips on pass

        turn_duration_gaddag_fail = time.time() - start_turn_time
        move_data_gaddag_fail = {
            'player': turn, 'move_type': action_chosen_gaddag_fail, 'score': 0,
            'word': status_reason, 'positions': [], 'blanks': set(), 'coord': '',
            'leave': current_rack_for_ai_logic[:], 'is_bingo': False, 'turn_duration': turn_duration_gaddag_fail,
            'word_with_blanks': status_reason, 'newly_placed': [],
            'tiles_placed_from_rack': [], 'blanks_played_info': [],
            'rack_before_move': rack_before_move_sgs, 'tiles_drawn_after_move': [],
            'luck_factor': 0.0,
            'initial_shuffled_bag_order_sgs': initial_shuffled_bag_order_sgs, # Save for GCG/SGS
            'initial_racks_sgs': initial_racks_sgs, # Save for GCG/SGS
            'next_turn_after_move': next_turn_val_gfail,
            'pass_count_after_move': updated_pass_count,
            'exchange_count_after_move': updated_exchange_count,
            'consecutive_zero_after_move': updated_consecutive_zero,
            'is_first_play_after_move': first_play, # Pass doesn't change first_play status
            'board_tile_counts_after_move': board_tile_counts_at_turn_start.copy(), # No change to board
            'blanks_played_count_after_move': blanks_played_count_at_turn_start # No change to blanks played
        }
        # These main game state variables are directly modified here for the pass scenario
        # For play/exchange, they are modified by play_hint_move or the exchange logic below
        return (action_chosen_gaddag_fail, None, [], move_data_gaddag_fail,
                first_play, updated_pass_count, updated_exchange_count, updated_consecutive_zero,
                set()) # No highlight on pass

    # --- Endgame Solver Logic ---
    # Check if endgame conditions are met (e.g., bag is empty or few tiles left)
    # For now, a simple check: bag is empty and AI is configured to use solver
    should_try_endgame_solver = use_endgame_solver_setting and not bag and practice_mode is None

    if should_try_endgame_solver:
        if not is_batch_running: print(f"{YELLOW}{debug_prefix}: Entering endgame solving phase.{RESET_COLOR}")
        is_solving_endgame = True # For UI indicator
        endgame_start_time = time.time()
        opponent_rack_endgame = racks[opponent_idx][:]
        current_score_diff_endgame = scores[player_idx] - scores[opponent_idx]

        # Call the endgame solver
        best_first_move_endgame = solve_endgame(current_rack_for_ai_logic, opponent_rack_endgame, tiles, blanks, board, current_score_diff_endgame)
        is_solving_endgame = False # Reset UI indicator

        action_chosen_endgame = 'pass'
        best_play_move_dict_endgame = None

        if best_first_move_endgame:
            if not is_batch_running:
                print(f"{YELLOW}{debug_prefix}: Endgame solver found a sequence. First move: {best_first_move_endgame.get('word', 'N/A')}, Score: {best_first_move_endgame.get('score',0)}{RESET_COLOR}")
            action_chosen_endgame = 'play'
            best_play_move_dict_endgame = best_first_move_endgame
        else:
            if not is_batch_running: print(f"{YELLOW}{debug_prefix}: Endgame solver suggests PASS or found no plays.{RESET_COLOR}")
            action_chosen_endgame = 'pass' # Default to pass if solver returns None

        # --- Prepare data for SGS and return for ENDGAME solver path ---
        drawn_tiles_sgs_endgame = []
        newly_placed_details_endgame = []
        move_type_sgs_endgame = action_chosen_endgame
        score_sgs_endgame = 0; word_sgs_endgame = ''; positions_sgs_endgame = []
        blanks_coords_sgs_endgame = set(); coord_sgs_endgame = ''; word_with_blanks_sgs_endgame = ''
        is_bingo_sgs_endgame = False; tiles_placed_from_rack_sgs_endgame = []; blanks_info_sgs_endgame = []
        start_pos_sgs_endgame = None; direction_sgs_endgame = None

        # Updated game flow variables (to be returned)
        next_turn_val_endgame = turn # Placeholder, will be updated by play_hint_move or pass logic
        first_play_val_endgame = first_play
        pass_count_val_endgame = pass_count
        exchange_count_val_endgame = exchange_count
        consecutive_zero_val_endgame = consecutive_zero_point_turns
        last_played_highlight_coords_endgame = set()
        # Local board_tile_counts and blanks_played_count for this path
        # They will be updated by play_hint_move if a play is made
        current_board_tile_counts_endgame = board_tile_counts.copy() # Use copy from start of ai_turn
        current_blanks_played_total_endgame = current_game_blanks_played_count # Use value from start of ai_turn

        if action_chosen_endgame == 'play' and best_play_move_dict_endgame:
            # Call play_hint_move to update board, racks, scores, etc.
            # play_hint_move will also handle drawing tiles.
            # We need to pass the *current* board_tile_counts and current_game_blanks_played_count
            # and get the updated ones back.
            (scores[player_idx], scores[opponent_idx]), \
            racks[player_idx], racks[opponent_idx], \
            tiles, blanks, bag, \
            first_play_val_endgame, next_turn_val_endgame, \
            current_board_tile_counts_endgame, current_blanks_played_total_endgame = play_hint_move( # Capture updated counts
                best_play_move_dict_endgame,
                scores,
                racks,
                player_idx,
                tiles,
                blanks,
                board, # Pass the main board for premium square info
                bag,
                first_play, # Pass current first_play state
                turn, # Pass current turn
                current_board_tile_counts_endgame, # Pass current board_tile_counts
                current_blanks_played_total_endgame # Pass current blanks_played_count
            )
            # board_tile_counts and current_game_blanks_played_count are now updated with return values

            score_sgs_endgame = best_play_move_dict_endgame.get('score', 0)
            word_sgs_endgame = best_play_move_dict_endgame.get('word', 'N/A')
            positions_sgs_endgame = best_play_move_dict_endgame.get('positions', [])
            blanks_coords_sgs_endgame = best_play_move_dict_endgame.get('blanks', set())
            start_pos_sgs_endgame = best_play_move_dict_endgame.get('start', (0,0))
            direction_sgs_endgame = best_play_move_dict_endgame.get('direction', 'right')
            coord_sgs_endgame = get_coord(start_pos_sgs_endgame, direction_sgs_endgame)
            word_with_blanks_sgs_endgame = best_play_move_dict_endgame.get('word_with_blanks', '')
            is_bingo_sgs_endgame = best_play_move_dict_endgame.get('is_bingo', False)
            newly_placed_details_endgame = best_play_move_dict_endgame.get('newly_placed', [])

            # Construct tiles_placed_from_rack_sgs and blanks_info_sgs
            temp_rack_sgs_endgame_logging = Counter(rack_before_move_sgs) # Based on rack *before* play_hint_move
            for r_eg, c_eg, l_eg in newly_placed_details_endgame:
                is_blank_eg = (r_eg, c_eg) in blanks_coords_sgs_endgame
                tile_to_log_removed_eg = ' ' if is_blank_eg else l_eg
                if temp_rack_sgs_endgame_logging[tile_to_log_removed_eg] > 0:
                    tiles_placed_from_rack_sgs_endgame.append(tile_to_log_removed_eg)
                    temp_rack_sgs_endgame_logging[tile_to_log_removed_eg] -= 1
                    if is_blank_eg:
                        blanks_info_sgs_endgame.append({'coord': (r_eg, c_eg), 'assigned_letter': l_eg})
                # else: Error or tile came from board (not handled by this simple logging here)

            # After a play, first_play becomes False, zero counts reset
            # first_play_val_endgame is already updated by play_hint_move
            pass_count_val_endgame = 0; exchange_count_val_endgame = 0
            consecutive_zero_val_endgame = 0 # Reset on successful play
            last_played_highlight_coords_endgame = set((pos[0], pos[1]) for pos in positions_sgs_endgame if pos)
            drawn_tiles_sgs_endgame = [tile for tile in racks[player_idx] if tile not in current_rack_for_ai_logic or current_rack_for_ai_logic.remove(tile) is None] # Simplistic: new tiles in rack

        elif action_chosen_endgame == 'pass':
            # Update game flow for pass
            consecutive_zero_val_endgame = consecutive_zero_point_turns + 1
            pass_count_val_endgame = pass_count + 1; exchange_count_val_endgame = 0
            next_turn_val_endgame = 3 - turn # Turn flips
            # board_tile_counts and blanks_played_count remain as they were at start of ai_turn for a pass
            current_board_tile_counts_endgame = board_tile_counts_at_turn_start.copy()
            current_blanks_played_total_endgame = blanks_played_count_at_turn_start

        turn_duration_endgame = time.time() - start_turn_time
        luck_factor_endgame = 0.0
        if drawn_tiles_sgs_endgame: # Only calculate if tiles were drawn
            try:
                # For luck calculation, use board/blank counts *before* the current move's tiles were added/counted
                luck_factor_endgame = calculate_luck_factor_cython(
                    drawn_tiles_sgs_endgame,
                    rack_before_move_sgs,       # Rack before this AI's move
                    tiles,                      # tiles grid *before* this AI's move (from ai_turn args)
                    blanks,                     # blanks set *before* this AI's move (from ai_turn args)
                    blanks_played_count_at_turn_start, # game total blanks played *before* this AI's move
                    get_remaining_tiles
                )
            except Exception as e_luck_eg_inner: luck_factor_endgame = 0.0

        move_data_sgs_endgame = {
            'player': turn, 'move_type': move_type_sgs_endgame, 'score': score_sgs_endgame,
            'word': word_sgs_endgame, 'positions': positions_sgs_endgame,
            'blanks_coords_on_board_this_play': list(blanks_coords_sgs_endgame), # Store set as list
            'coord': coord_sgs_endgame,
            'leave': racks[player_idx][:] if action_chosen_endgame == 'play' else current_rack_for_ai_logic[:], # Rack after play or current rack if pass
            'is_bingo': is_bingo_sgs_endgame, 'turn_duration': turn_duration_endgame,
            'word_with_blanks': word_with_blanks_sgs_endgame,
            'newly_placed_details': newly_placed_details_endgame, # Actual (r,c,l) of tiles from rack
            'tiles_placed_from_rack': tiles_placed_from_rack_sgs_endgame, # Letters (or ' ') removed from rack
            'blanks_played_info': blanks_info_sgs_endgame, # List of {'coord':(r,c), 'assigned_letter': 'A'}
            'rack_before_move': rack_before_move_sgs, # Rack state before this move
            'tiles_drawn_after_move': drawn_tiles_sgs_endgame, # Actual tiles drawn
            'luck_factor': luck_factor_endgame,
            'initial_shuffled_bag_order_sgs': initial_shuffled_bag_order_sgs,
            'initial_racks_sgs': initial_racks_sgs,
            'next_turn_after_move': next_turn_val_endgame,
            'pass_count_after_move': pass_count_val_endgame,
            'exchange_count_after_move': exchange_count_val_endgame,
            'consecutive_zero_after_move': consecutive_zero_val_endgame,
            'is_first_play_after_move': first_play_val_endgame, # State of first_play *after* this move
            'board_tile_counts_after_move': current_board_tile_counts_endgame.copy(), # Board counts *after* this move
            'blanks_played_count_after_move': current_blanks_played_total_endgame # Blanks played *after* this move
        }
        return (action_chosen_endgame, best_play_move_dict_endgame, [], move_data_sgs_endgame,
                first_play_val_endgame, pass_count_val_endgame, exchange_count_val_endgame, consecutive_zero_val_endgame,
                last_played_highlight_coords_endgame, current_board_tile_counts_endgame, current_blanks_played_total_endgame) # Return updated counts

    # --- Standard Move Generation & Evaluation (Not Endgame) ---
    if not is_batch_running: print(f"{debug_prefix}: Preparing to generate moves. Rack: {''.join(sorted(current_rack_for_ai_logic))}, First Play Flag: {first_play}")
    all_ai_moves_generated = []
    if USE_CYTHON_MOVE_GENERATION and GADDAG_STRUCTURE and GADDAG_STRUCTURE.root and DAWG:
        if not is_batch_running: print(f"{debug_prefix}: Generating moves via Cython GADDAG...")
        all_ai_moves_generated = generate_all_moves_gaddag_cython(
            current_rack_for_ai_logic, tiles, board, blanks, GADDAG_STRUCTURE.root, DAWG
        )
    else: # Fallback or if Cython is disabled
        # This part should ideally not be reached if Cython is the primary path
        # and GADDAG_STRUCTURE.root or DAWG is None (handled by the pass condition earlier)
        if not is_batch_running: print(f"{debug_prefix}: Generating moves via Python (fallback)...")
        # Fallback: all_ai_moves_generated = generate_all_moves(current_rack_for_ai_logic, tiles, board, blanks, lexicon.DAWG) # Example fallback
        all_ai_moves_generated = [] # Ensure it's a list

    if all_ai_moves_generated is None: # Safety check
        all_ai_moves_generated = []

    if not is_batch_running:
        print(f"{debug_prefix}: Generated {len(all_ai_moves_generated)} raw moves.")
        if all_ai_moves_generated:
            print(f"{debug_prefix}: Sample of first few generated moves (raw scores):")
            for i, move_sample in enumerate(all_ai_moves_generated[:3]):
                print(f"  Move {i+1}: Word='{move_sample.get('word','N/A')}', Score={move_sample.get('score',0)}, Start={move_sample.get('start',(0,0))}, Dir={move_sample.get('direction','?')}")

    # --- Practice Mode Checks & Potential Pauses (before full AI logic if not batch) ---
    if not is_batch_running and practice_mode:
        if practice_mode == "power_tiles" and not ai_paused_for_power_tile_this_turn:
            # Check if any power tile from letter_checks is on the rack
            checked_power_tiles_ai = {letter for i, letter in enumerate(['J', 'Q', 'X', 'Z']) if letter_checks[i]}
            power_tiles_on_rack_ai = sorted([tile for tile in current_rack_for_ai_logic if tile in checked_power_tiles_ai])
            if power_tiles_on_rack_ai:
                ai_current_power_tile_this_turn = power_tiles_on_rack_ai[0] # Pick the first one for the pause
                ai_paused_for_power_tile_this_turn = True
                # Return a special status to indicate pause, AI hasn't made a move decision yet
                return ("pause_power_tile", ai_current_power_tile_this_turn, [], {},
                        first_play, pass_count, exchange_count, consecutive_zero_point_turns,
                        last_played_highlight_coords, board_tile_counts, current_game_blanks_played_count) # Pass back current counts

        elif practice_mode == "bingo_bango_bongo" and not ai_paused_for_bingo_practice_this_turn:
            if all_ai_moves_generated is None: all_ai_moves_generated = [] # Ensure it's iterable
            found_bingo_ai = any(move.get('is_bingo', False) for move in all_ai_moves_generated)
            if found_bingo_ai:
                ai_paused_for_bingo_practice_this_turn = True
                return ("pause_bingo_practice", None, [], {},
                        first_play, pass_count, exchange_count, consecutive_zero_point_turns,
                        last_played_highlight_coords, board_tile_counts, current_game_blanks_played_count) # Pass back current counts

    # --- AI Decision Logic (Play, Exchange, Pass) ---
    best_play_move_ai_reg = None
    best_exchange_tiles_ai_reg = []
    action_chosen_ai_reg = 'pass' # Default action

    can_play_ai_reg = bool(all_ai_moves_generated)
    run_simulation_ai_reg = (use_ai_simulation_setting and game_mode in [MODE_HVA, MODE_AVA] and practice_mode is None and can_play_ai_reg)

    if not is_batch_running: print(f"{debug_prefix}: Running standard evaluation/decision logic" + (" via Cython..." if USE_CYTHON_AI_TURN_LOGIC else "..."))

    if run_simulation_ai_reg:
        if not is_batch_running: print(f"{debug_prefix}: AI Simulation is ON. Running 2-ply lookahead.")
        opponent_rack_len_sim_ai_reg = len(racks[opponent_idx]) if opponent_idx < len(racks) else 7 # Default if opponent rack not available
        # Pass board_tile_counts and current_game_blanks_played_count to run_ai_simulation
        simulation_results_ai_reg = run_ai_simulation(
            ai_rack=current_rack_for_ai_logic,
            opponent_rack_len=opponent_rack_len_sim_ai_reg,
            tiles=tiles, blanks=blanks, board=board, bag=bag,
            gaddag_root=GADDAG_STRUCTURE.root, is_first_play=first_play,
            board_tile_counts=board_tile_counts, # Pass the current board counts
            blanks_played_count=current_game_blanks_played_count, # Pass the current blanks played count
            num_ai_candidates=DEFAULT_AI_CANDIDATES,
            num_opponent_sims=DEFAULT_OPPONENT_SIMULATIONS,
            num_post_sim_candidates=DEFAULT_POST_SIM_CANDIDATES
        )
        if simulation_results_ai_reg:
            best_play_move_ai_reg = simulation_results_ai_reg[0]['move']
            action_chosen_ai_reg = 'play'
            if not is_batch_running:
                best_play_eval_sim = simulation_results_ai_reg[0]['final_score']
                print(f"{debug_prefix}: Simulation top choice: {best_play_move_ai_reg.get('word','N/A')} (Final Eval: {best_play_eval_sim:.2f})")
        else: # Simulation yielded no moves (shouldn't happen if can_play_ai_reg was true)
            action_chosen_ai_reg = 'pass' # Fallback if simulation fails unexpectedly
            if not is_batch_running: print(f"{debug_prefix}: AI Simulation returned no moves, defaulting to pass.")

    elif USE_CYTHON_AI_TURN_LOGIC:
        action_chosen_ai_reg, best_move_data_ai_reg = ai_turn_logic_cython(
                all_ai_moves_generated,
                current_rack_for_ai_logic,
                tiles,  # Pass current tiles grid
                blanks, # Pass current blanks set
                current_game_blanks_played_count, # Game total
                len(bag),
                get_remaining_tiles, 
                find_best_exchange_option, 
                EXCHANGE_PREFERENCE_THRESHOLD,
                MIN_SCORE_TO_AVOID_EXCHANGE
            )
        if action_chosen_ai_reg == 'play': best_play_move_ai_reg = best_move_data_ai_reg
        elif action_chosen_ai_reg == 'exchange': best_exchange_tiles_ai_reg = best_move_data_ai_reg
    else: # Python fallback for AI decision (should match Cython's logic if possible)
        # Simplified Python fallback:
        # evaluated_options = standard_evaluation_python(all_ai_moves_generated) # Assuming a python equivalent
        # ... logic to choose play/exchange/pass ...
        if can_play_ai_reg:
            # Minimal Python fallback: pick the highest raw score if Cython is off
            # This does not include leave eval or exchange logic of the Cython version.
            evaluated_options = sorted(all_ai_moves_generated, key=lambda m: m.get('score', -float('inf')), reverse=True)
            if evaluated_options:
                best_play_move_ai_reg = evaluated_options[0]
                action_chosen_ai_reg = 'play'
            else:
                action_chosen_ai_reg = 'pass'
        else:
            action_chosen_ai_reg = 'pass'

    if not is_batch_running:
        if action_chosen_ai_reg == 'play' and best_play_move_ai_reg:
            print(f"{debug_prefix}: Cython logic chose action: PLAY")
            print(f"  Play Details: '{best_play_move_ai_reg.get('word_with_blanks', best_play_move_ai_reg.get('word','?'))}' Score: {best_play_move_ai_reg.get('score',0)}")
        elif action_chosen_ai_reg == 'exchange':
            print(f"{debug_prefix}: Cython logic chose action: EXCHANGE. Tiles: {''.join(sorted(best_exchange_tiles_ai_reg))}")
        else:
            print(f"{debug_prefix}: Cython logic chose action: PASS")

    # --- Prepare data for SGS and update game state based on AI REGULAR decision ---
    next_turn_val_reg = turn # Placeholder
    drawn_tiles_sgs_reg = []
    newly_placed_details_sgs_reg = []
    move_type_sgs_reg = action_chosen_ai_reg
    score_sgs_reg = 0; word_sgs_reg = ''; positions_sgs_reg = []
    blanks_coords_sgs_reg = set(); coord_sgs_reg = ''; word_with_blanks_sgs_reg = ''
    is_bingo_sgs_reg = False; tiles_placed_from_rack_sgs_reg = []; blanks_info_sgs_reg = []
    start_pos_sgs_reg = None; direction_sgs_reg = None; exchanged_tiles_sgs_reg = []

    # Game flow variables to be returned/updated
    first_play_val_reg = first_play
    pass_count_val_reg = pass_count
    exchange_count_val_reg = exchange_count
    consecutive_zero_val_reg = consecutive_zero_point_turns
    last_played_highlight_coords_reg = set()

    # Local board_tile_counts and blanks_played_count for this path
    current_board_tile_counts_reg = board_tile_counts.copy()
    current_blanks_played_total_reg = current_game_blanks_played_count

    if action_chosen_ai_reg == 'play' and best_play_move_ai_reg:
        move_type_sgs_reg = 'place'
        if not is_batch_running: print(f"{debug_prefix} playing move: '{best_play_move_ai_reg.get('word','N/A')}' at {get_coord(best_play_move_ai_reg.get('start',(0,0)), best_play_move_ai_reg.get('direction','right'))}")

        # Call play_hint_move to update board, racks, scores, etc.
        (scores[player_idx], scores[opponent_idx]), \
        racks[player_idx], racks[opponent_idx], \
        tiles, blanks, bag, \
        first_play_val_reg, next_turn_val_reg, \
        current_board_tile_counts_reg, current_blanks_played_total_reg = play_hint_move( # Capture updated counts
            best_play_move_ai_reg,
            scores,
            racks,
            player_idx,
            tiles,
            blanks,
            board,
            bag,
            first_play, # Pass current first_play
            turn, # Pass current turn
            current_board_tile_counts_reg, # Pass current board_tile_counts
            current_blanks_played_total_reg # Pass current blanks_played_count
        )
        # board_tile_counts and current_game_blanks_played_count are now updated with return values

        score_sgs_reg = best_play_move_ai_reg.get('score', 0)
        word_sgs_reg = best_play_move_ai_reg.get('word', 'N/A')
        positions_sgs_reg = best_play_move_ai_reg.get('positions', [])
        blanks_coords_sgs_reg = best_play_move_ai_reg.get('blanks', set())
        start_pos_sgs_reg = best_play_move_ai_reg.get('start', (0,0))
        direction_sgs_reg = best_play_move_ai_reg.get('direction', 'right')
        coord_sgs_reg = get_coord(start_pos_sgs_reg, direction_sgs_reg)
        word_with_blanks_sgs_reg = best_play_move_ai_reg.get('word_with_blanks', '')
        is_bingo_sgs_reg = best_play_move_ai_reg.get('is_bingo', False)

        # 'newly_placed' is crucial for SGS logging and luck factor
        newly_placed_for_sgs_construction = best_play_move_ai_reg.get('newly_placed', [])
        newly_placed_details_sgs_reg = newly_placed_for_sgs_construction

        # Construct tiles_placed_from_rack_sgs and blanks_info_sgs
        temp_rack_sgs_ai_reg_logging = Counter(rack_before_move_sgs) # Rack before play_hint_move
        for r_air, c_air, l_air in newly_placed_details_sgs_reg:
            is_blank_air = (r_air, c_air) in blanks_coords_sgs_reg
            tile_to_log_removed_air = ' ' if is_blank_air else l_air
            if temp_rack_sgs_ai_reg_logging[tile_to_log_removed_air] > 0:
                tiles_placed_from_rack_sgs_reg.append(tile_to_log_removed_air)
                temp_rack_sgs_ai_reg_logging[tile_to_log_removed_air] -= 1
                if is_blank_air:
                    blanks_info_sgs_reg.append({'coord': (r_air, c_air), 'assigned_letter': l_air})

        # After a play, first_play becomes False, zero counts reset
        # first_play_val_reg is already updated by play_hint_move
        pass_count_val_reg = 0; exchange_count_val_reg = 0
        consecutive_zero_val_reg = 0 # Reset on successful play
        last_played_highlight_coords_reg = set((pos[0], pos[1]) for pos in positions_sgs_reg if pos)
        # Determine drawn_tiles for SGS (tiles added to AI's rack after play_hint_move)
        # This is a bit tricky as play_hint_move modifies racks[player_idx] in place.
        # A simple way: compare rack_before_move_sgs with the rack after play_hint_move,
        # considering the tiles_placed_from_rack_sgs.
        # However, play_hint_move already returns the updated rack.
        # drawn_tiles_sgs_reg is filled by play_hint_move's draw logic.
        rack_after_play_and_draw = racks[player_idx][:]
        temp_rack_before_counts = Counter(rack_before_move_sgs)
        temp_placed_counts = Counter(tiles_placed_from_rack_sgs_reg)
        rack_content_before_draw = list((temp_rack_before_counts - temp_placed_counts).elements())
        
        drawn_tiles_sgs_reg = []
        rack_after_draw_counts = Counter(rack_after_play_and_draw)
        rack_before_draw_counts = Counter(rack_content_before_draw)
        diff_counts = rack_after_draw_counts - rack_before_draw_counts
        drawn_tiles_sgs_reg = list(diff_counts.elements())

    elif action_chosen_ai_reg == 'exchange':
        move_type_sgs_reg = 'exchange'
        if not is_batch_running: print(f"{debug_prefix} exchanging tiles: {''.join(sorted(best_exchange_tiles_ai_reg))}")
        exchanged_tiles_sgs_reg = list(best_exchange_tiles_ai_reg) # Ensure it's a list

        # Perform exchange on a copy of the rack first to determine drawn tiles
        rack_copy_for_exchange_reg = racks[player_idx][:]
        temp_rack_after_exchange_reg = []
        exchange_counts_temp_reg = Counter(exchanged_tiles_sgs_reg)

        for tile_in_rack in rack_copy_for_exchange_reg:
            if exchange_counts_temp_reg[tile_in_rack] > 0:
                exchange_counts_temp_reg[tile_in_rack] -= 1
            else:
                temp_rack_after_exchange_reg.append(tile_in_rack)

        num_to_draw_exch_reg = len(exchanged_tiles_sgs_reg)
        drawn_tiles_sgs_reg = [bag.pop(0) for _ in range(num_to_draw_exch_reg) if bag]
        temp_rack_after_exchange_reg.extend(drawn_tiles_sgs_reg)
        racks[player_idx] = temp_rack_after_exchange_reg[:] # Update the actual rack
        bag.extend(exchanged_tiles_sgs_reg) # Return exchanged tiles to bag
        random.shuffle(bag) # Shuffle bag

        score_sgs_reg = 0 # No score for exchange
        consecutive_zero_val_reg = consecutive_zero_point_turns + 1
        exchange_count_val_reg = exchange_count + 1; pass_count_val_reg = 0
        next_turn_val_reg = 3 - turn # Turn flips
        # board_tile_counts and blanks_played_count are not changed by an exchange
        current_board_tile_counts_reg = board_tile_counts_at_turn_start.copy()
        current_blanks_played_total_reg = blanks_played_count_at_turn_start

    else: # Pass
        move_type_sgs_reg = 'pass'
        if not is_batch_running: print(f"{debug_prefix} passed.")
        score_sgs_reg = 0
        consecutive_zero_val_reg = consecutive_zero_point_turns + 1
        pass_count_val_reg = pass_count + 1; exchange_count_val_reg = 0
        next_turn_val_reg = 3 - turn # Turn flips
        # board_tile_counts and blanks_played_count are not changed by a pass
        current_board_tile_counts_reg = board_tile_counts_at_turn_start.copy()
        current_blanks_played_total_reg = blanks_played_count_at_turn_start

    # --- Calculate Luck Factor for Play/Exchange ---
    luck_factor_sgs_reg = 0.0
    if drawn_tiles_sgs_reg : # Only if tiles were drawn (play or exchange)
        try:
            luck_factor_sgs_reg = calculate_luck_factor_cython(
                drawn_tiles_sgs_reg,
                rack_before_move_sgs,
                tiles,                      # tiles grid *before* this AI's move (from ai_turn args)
                blanks,                     # blanks set *before* this AI's move (from ai_turn args)
                blanks_played_count_at_turn_start, # game total blanks played *before* this AI's move
                get_remaining_tiles
            )
            if not is_batch_running:
                drawn_tiles_str_luck_reg = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles_sgs_reg))
                print(f"  {debug_prefix}: Drew: {drawn_tiles_str_luck_reg}, Luck (Cython): {luck_factor_sgs_reg:+.2f}")
        except Exception as e_luck_reg_inner:
            if not is_batch_running: print(f"Error in AI turn calling calculate_luck_factor_cython: {e_luck_reg_inner}")
            luck_factor_sgs_reg = 0.0

    turn_duration_sgs_reg = time.time() - start_turn_time
    if not is_batch_running: print(f"{debug_prefix}: Turn duration: {turn_duration_sgs_reg:.3f}s")

    # --- Construct move_data for SGS logging ---
    final_leave_sgs_reg = []
    if move_type_sgs_reg == 'place':
        final_leave_sgs_reg = best_play_move_ai_reg.get('leave', racks[player_idx][:]) # Prefer leave from move dict
    elif move_type_sgs_reg == 'exchange':
        final_leave_sgs_reg = racks[player_idx][:] # Rack after exchange and draw
    else: # Pass
        final_leave_sgs_reg = current_rack_for_ai_logic[:] # Rack remains unchanged

    move_data_sgs_reg = {
        'player': turn,
        'move_type': move_type_sgs_reg,
        'score': score_sgs_reg,
        'word': word_sgs_reg,
        'positions': positions_sgs_reg,
        'blanks_coords_on_board_this_play': list(blanks_coords_sgs_reg),
        'coord': coord_sgs_reg,
        'leave': final_leave_sgs_reg,
        'is_bingo': is_bingo_sgs_reg,
        'turn_duration': turn_duration_sgs_reg,
        'word_with_blanks': word_with_blanks_sgs_reg,
        'newly_placed_details': newly_placed_details_sgs_reg,
        'tiles_placed_from_rack': tiles_placed_from_rack_sgs_reg,
        'blanks_played_info': blanks_info_sgs_reg,
        'rack_before_move': rack_before_move_sgs,
        'tiles_drawn_after_move': drawn_tiles_sgs_reg,
        'exchanged_tiles': exchanged_tiles_sgs_reg if move_type_sgs_reg == 'exchange' else [],
        'luck_factor': luck_factor_sgs_reg,
        'initial_shuffled_bag_order_sgs': initial_shuffled_bag_order_sgs,
        'initial_racks_sgs': initial_racks_sgs,
        'next_turn_after_move': next_turn_val_reg,
        'pass_count_after_move': pass_count_val_reg,
        'exchange_count_after_move': exchange_count_val_reg,
        'consecutive_zero_after_move': consecutive_zero_val_reg,
        'is_first_play_after_move': first_play_val_reg,
        'board_tile_counts_after_move': current_board_tile_counts_reg.copy(),
        'blanks_played_count_after_move': current_blanks_played_total_reg
    }

    # Update main game state variables that were passed by value essentially
    # The calling function (handle_ai_turn_trigger) will use these to update its state.
    # Racks, tiles, blanks, bag, scores are modified in-place if play_hint_move was called or exchange happened.
    # board_tile_counts and current_game_blanks_played_count are returned to reflect changes.

    return (action_chosen_ai_reg, best_play_move_ai_reg, best_exchange_tiles_ai_reg, move_data_sgs_reg,
            first_play_val_reg, pass_count_val_reg, exchange_count_val_reg, consecutive_zero_val_reg,
            last_played_highlight_coords_reg, current_board_tile_counts_reg, current_blanks_played_total_reg) # Return updated counts 




# In Scrabble Game.py

def initialize_game(selected_mode_result, return_data, main_called_flag):
    """Initializes or re-initializes the game state based on mode selection."""
    global board, tiles, racks, blanks, scores, turn, first_play, game_mode, is_ai
    global player_names, bag, move_history, replay_mode, current_replay_turn
    global pass_count, exchange_count, consecutive_zero_point_turns, is_loaded_game
    global last_played_highlight_coords, USE_ENDGAME_SOLVER, USE_AI_SIMULATION
    global DEVELOPER_PROFILE_ENABLED, practice_mode
    global is_batch_running, total_batch_games, current_batch_game_num, batch_results, initial_game_config
    global practice_target_moves, practice_best_move, practice_solved, practice_end_message
    global all_moves 
    global gaddag_loading_status, gaddag_load_thread, GADDAG_STRUCTURE, DAWG
    global board_tile_counts, blanks_played_count
    global letter_checks, number_checks 
    global initial_shuffled_bag_order_sgs, initial_racks_sgs 
    global replay_initial_shuffled_bag 
    global endgame_start_time 
    global gaddag_loaded_event
    # MODIFICATION START: Ensure human_player is global and will be set
    global human_player 
    # MODIFICATION END

    is_batch_running = False 
    human_player = 1 # Default for this scope, will be updated by specific modes

    if gaddag_loaded_event is None: 
        gaddag_loaded_event = threading.Event()
    gaddag_loaded_event.clear() 

    if main_called_flag and (gaddag_loading_status == 'loaded' or gaddag_loading_status == 'error'):
        gaddag_loading_status = 'idle' 

    if gaddag_loading_status == 'idle': 
        try:
            if DAWG is None: 
                print("Loading DAWG dictionary...")
                with open("dawg.pkl", "rb") as f_dawg: 
                    DAWG = pickle.load(f_dawg)
                print("DAWG dictionary loaded successfully.")
        except Exception as e_dawg:
            print(f"ERROR: Failed to load DAWG dictionary: {e_dawg}")
            DAWG = None 
            gaddag_loading_status = 'error' 
            if gaddag_loaded_event:
                gaddag_loaded_event.set()
        
        if DAWG is not None: 
            print("--- initialize_game(): Starting GADDAG background load ---")
            gaddag_loading_status = 'loading' 
            gaddag_load_thread = threading.Thread(target=_load_gaddag_background, args=(gaddag_loaded_event,), daemon=True)
            gaddag_load_thread.start()
        else:
            gaddag_loading_status = 'error'
            if gaddag_loaded_event: 
                gaddag_loaded_event.set()

    board_init, labels_init, tiles_init = create_board()
    board = board_init 
    tiles = tiles_init 
    scores = [0, 0]
    blanks = set()
    racks = [[], []]
    turn = 1
    first_play = True
    pass_count = 0
    exchange_count = 0
    consecutive_zero_point_turns = 0
    player_names = ["Player 1", "Player 2"] # Default, will be overwritten
    is_ai = [False, False] # Default, will be overwritten
    move_history = []
    replay_mode = False
    current_replay_turn = 0
    is_loaded_game = False 
    last_played_highlight_coords = set()
    USE_ENDGAME_SOLVER = False
    USE_AI_SIMULATION = False
    practice_mode = None
    practice_target_moves = []
    practice_best_move = None
    practice_solved = False
    practice_end_message = ""
    all_moves = []
    endgame_start_time = 0
    board_tile_counts = Counter() 
    blanks_played_count = 0    
    letter_checks = [True] * 4 
    number_checks = [True, True, True, True, False, False]
    practice_probability_max_index = None
    initial_shuffled_bag_order_sgs = []
    initial_racks_sgs = [[],[]]
    replay_initial_shuffled_bag = None 

    if selected_mode_result == "LOADED_SGS_GAME":
        is_loaded_game = True 
        loaded_data = return_data 
        if not isinstance(loaded_data, dict):
            print("Error: Loaded game data is not in the expected format. Starting new game.")
            selected_mode_result = MODE_HVH 
        else:
            player_names = loaded_data.get('player_names', ["P1", "P2"])
            game_mode = loaded_data.get('game_mode', MODE_HVH) 
            is_ai = loaded_data.get('is_ai', [False, False])
            practice_mode = loaded_data.get('practice_mode', None)
            USE_ENDGAME_SOLVER = loaded_data.get('use_endgame_solver_setting', False)
            USE_AI_SIMULATION = loaded_data.get('use_ai_simulation_setting', False)
            initial_shuffled_bag_order_sgs = loaded_data.get('initial_shuffled_bag_order_sgs', [])
            initial_racks_sgs = loaded_data.get('initial_racks_sgs', [[],[]])
            move_history = loaded_data.get('move_history', [])
            # MODIFICATION START: Set global human_player based on loaded game_mode if HVA
            if game_mode == MODE_HVA:
                if not is_ai[0]: human_player = 1
                elif not is_ai[1]: human_player = 2
                else: human_player = 1 # Default if is_ai is inconsistent for HVA
            elif game_mode == MODE_HVH:
                human_player = 1 # Doesn't strictly matter for HVH but set a default
            # For AVA, human_player doesn't apply in the same way
            # MODIFICATION END
            replay_mode = True 
            current_replay_turn = len(move_history) 

            if not initial_shuffled_bag_order_sgs and move_history: 
                initial_shuffled_bag_order_sgs = move_history[0].get('initial_shuffled_bag_order_sgs', [])

            sim_board, sim_tiles, sim_racks, sim_blanks, sim_scores, sim_bag, sim_turn, sim_first_play, sim_board_counts, sim_blanks_played = simulate_game_up_to(
                len(move_history), 
                move_history,
                initial_shuffled_bag_order_sgs[:], 
                initial_racks_sgs 
            )
            board = sim_board; tiles = sim_tiles; racks = sim_racks; blanks = sim_blanks; scores = sim_scores; bag = sim_bag
            turn = sim_turn; first_play = sim_first_play
            board_tile_counts = sim_board_counts; blanks_played_count = sim_blanks_played
            print(f"--- Game loaded from SGS. Replay set to end. Moves: {len(move_history)} ---")
            if not main_called_flag: main_called_flag = True 
            return True

    elif selected_mode_result == "BATCH_MODE":
        is_batch_running = True 
        if not isinstance(return_data, tuple) or len(return_data) < 8: 
            print("Error: Batch mode data is not in the expected format. Cannot start batch.")
            return False 
        game_mode_batch, player_names_batch, human_player_batch_val, use_endgame_solver_checked, use_ai_simulation_checked, num_games, visualize_batch_checked, cprofile_checked = return_data[:8]

        DEVELOPER_PROFILE_ENABLED = cprofile_checked 
        total_batch_games = num_games
        current_batch_game_num = 0 
        batch_results = []
        USE_ENDGAME_SOLVER = use_endgame_solver_checked
        USE_AI_SIMULATION = use_ai_simulation_checked
        
        # MODIFICATION START: Set global human_player for batch mode
        human_player = human_player_batch_val
        # MODIFICATION END
        
        batch_now_b = datetime.datetime.now() 
        batch_date_str_b = batch_now_b.strftime("%d%b%y").upper()
        batch_time_str_b = batch_now_b.strftime("%H%M")
        batch_seq_num_b = 1; max_existing_batch_num_b = 0
        for filename_iter_b in os.listdir("."): 
            if filename_iter_b.startswith(batch_date_str_b) and filename_iter_b.endswith(".txt") and "-BATCH-" in filename_iter_b:
                try: parts_b = filename_iter_b[:-4].split('-'); num_b = int(parts_b[-1]); max_existing_batch_num_b = max(max_existing_batch_num_b, num_b)
                except (IndexError, ValueError) : pass
        batch_seq_num_b = max_existing_batch_num_b + 1
        batch_base_filename_prefix_b = f"{batch_date_str_b}-{batch_time_str_b}-BATCH-{batch_seq_num_b}"
        
        is_ai_batch_init = [False, False] 
        if game_mode_batch == MODE_HVA: is_ai_batch_init = [False, True] if human_player == 1 else [True, False]
        elif game_mode_batch == MODE_AVA: is_ai_batch_init = [True, True]
        
        initial_game_config = {
            'game_mode': game_mode_batch, 'player_names': player_names_batch,
            'is_ai': is_ai_batch_init, 'human_player': human_player, # Use the now set global
            'use_endgame_solver': USE_ENDGAME_SOLVER, 'use_ai_simulation': USE_AI_SIMULATION,
            'batch_filename_prefix': batch_base_filename_prefix_b,
            'visualize_batch': visualize_batch_checked
        }
        is_ai = is_ai_batch_init; player_names = player_names_batch; game_mode = game_mode_batch
        
        bag_for_dealing_sgs_batch_init = create_standard_bag(); random.shuffle(bag_for_dealing_sgs_batch_init) 
        initial_shuffled_bag_order_sgs = list(bag_for_dealing_sgs_batch_init)
        bag = list(initial_shuffled_bag_order_sgs) 
        racks_batch_init_local = [[], []] 
        for i_init in range(2): 
            try: racks_batch_init_local[i_init] = [bag.pop(0) for _ in range(7)]
            except IndexError: print("Error: Not enough tiles in bag for initial deal in batch."); return False 
        initial_racks_sgs = [list(racks_batch_init_local[0]), list(racks_batch_init_local[1])]
        racks = racks_batch_init_local
        if not main_called_flag: main_called_flag = True
        return True

    else: # New Game or Practice Mode (not loaded, not batch)
        if not isinstance(return_data, tuple) or len(return_data) != 11:
            print(f"Error: Incorrect data format from mode selection. Expected 11 items, got {len(return_data) if isinstance(return_data, tuple) else 'Non-tuple'}. Starting default HVH game.")
            game_mode = MODE_HVH
            player_names = ["Player 1", "Player 2"]
            is_ai = [False, False]
            human_player = 1 # Default for HVH
            practice_mode = None
            USE_ENDGAME_SOLVER = False
            USE_AI_SIMULATION = False
        else:
            player_names_new, human_player_new_val, practice_mode_new, letter_checks_new, number_checks_new, \
            use_endgame_solver_checked_init, use_ai_simulation_checked_init, practice_state, \
            visualize_batch_checked_init, cprofile_checked_init, practice_prob_max_idx = return_data 

            # MODIFICATION START: Set global human_player
            human_player = human_player_new_val
            # MODIFICATION END
            game_mode = selected_mode_result 
            player_names = player_names_new
            practice_mode = practice_mode_new
            letter_checks = letter_checks_new
            number_checks = number_checks_new
            USE_ENDGAME_SOLVER = use_endgame_solver_checked_init
            USE_AI_SIMULATION = use_ai_simulation_checked_init
            DEVELOPER_PROFILE_ENABLED = cprofile_checked_init
            practice_probability_max_index = practice_prob_max_idx

            if practice_mode == "eight_letter" and practice_state:
                board = practice_state.get('board', board)
                tiles = practice_state.get('tiles', tiles)
                racks = practice_state.get('racks', racks)
                blanks = practice_state.get('blanks', blanks)
                bag = practice_state.get('bag', []) 
                first_play = practice_state.get('first_play', False)
                scores = practice_state.get('scores', [0,0])
                turn = practice_state.get('turn', 1)
                board_tile_counts = Counter(t_ps for row_t_ps in tiles for t_ps in row_t_ps if t_ps) 
                blanks_played_count = sum(1 for r_b_ps,c_b_ps in blanks if tiles[r_b_ps][c_b_ps]) 
                initial_shuffled_bag_order_sgs = [] 
                initial_racks_sgs = [list(racks[0]), list(racks[1])] if racks and len(racks) == 2 else [[],[]]
                practice_best_move = practice_state.get('best_move') 
                practice_target_moves = practice_state.get('target_moves', [])
            elif practice_state: 
                board = practice_state.get('board', board); tiles = practice_state.get('tiles', tiles)
                racks = practice_state.get('racks', racks); blanks = practice_state.get('blanks', blanks)
                bag = practice_state.get('bag', bag); first_play = practice_state.get('first_play', first_play)
                scores = practice_state.get('scores', scores); turn = practice_state.get('turn', turn)
                board_tile_counts = Counter(t_ps_o for row_t_ps_o in tiles for t_ps_o in row_t_ps_o if t_ps_o) 
                blanks_played_count = sum(1 for r_b_ps_o,c_b_ps_o in blanks if tiles[r_b_ps_o][c_b_ps_o]) 
                initial_shuffled_bag_order_sgs = list(bag) if bag else []
                initial_racks_sgs = [list(racks[0]), list(racks[1])] if racks and len(racks) == 2 else [[],[]]
            else: 
                bag_for_dealing_sgs_new_game = create_standard_bag(); random.shuffle(bag_for_dealing_sgs_new_game) 
                initial_shuffled_bag_order_sgs = list(bag_for_dealing_sgs_new_game)
                bag = list(initial_shuffled_bag_order_sgs)
                for i_new_game_deal in range(2): 
                    try: racks[i_new_game_deal] = [bag.pop(0) for _ in range(7)]
                    except IndexError: print("Error: Not enough tiles in bag for initial deal."); return False
                initial_racks_sgs = [list(racks[0]), list(racks[1])]
        
        # Determine is_ai based on game_mode and the now set global human_player
        if game_mode == MODE_HVH: is_ai = [False, False]
        elif game_mode == MODE_HVA: 
            is_ai = [False, True] if human_player == 1 else [True, False] 
        elif game_mode == MODE_AVA: is_ai = [True, True]
    
    if not is_loaded_game and not is_batch_running : 
        if not is_ai[0] and racks[0]: racks[0].sort()
        if not is_ai[1] and racks[1] and practice_mode != "eight_letter": racks[1].sort()

    if not main_called_flag: main_called_flag = True 

    if not initial_shuffled_bag_order_sgs and not replay_mode and not is_batch_running and not practice_mode: 
        temp_bag_sgs_fallback_init = create_standard_bag(); random.shuffle(temp_bag_sgs_fallback_init) 
        initial_shuffled_bag_order_sgs = temp_bag_s_fallback_init[:]
        if not bag : bag = list(initial_shuffled_bag_order_sgs) 

    return True








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











def does_move_form_five_letter_word(move, current_tiles, current_blanks):
    """
    Checks if a given move forms at least one 5-letter word.
    Calls the Cython version of find_all_words_formed.

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

    # --- MODIFICATION: Call Cython version ---
    words_formed_details = find_all_words_formed_cython(newly_placed_details, temp_tiles)
    # --- END MODIFICATION ---

    # Check if any formed word has length 5
    for word_detail in words_formed_details:
        # Ensure word_detail is iterable (list of tuples) before joining
        if isinstance(word_detail, list):
            word_len = len("".join(t[2] for t in word_detail if isinstance(t, tuple) and len(t) >= 3))
            if word_len == 5:
                return True # Found at least one 5-letter word
        else:
            print(f"Warning (does_move_form_five_letter_word): Unexpected item in words_formed_details: {word_detail}")


    return False # No 5-letter words were formed












def draw_game_screen(screen, state):
    # Unpack all necessary state variables
    board = state['board']; tiles = state['tiles']; blanks = state['blanks'] 
    scores = state['scores']; racks = state['racks']; turn = state['turn']
    player_names = state['player_names']; dragged_tile = state['dragged_tile']
    drag_pos = state['drag_pos']; drag_offset = state['drag_offset']
    practice_mode = state['practice_mode']; bag = state['bag']
    move_history = state['move_history']; scroll_offset = state['scroll_offset']
    is_ai = state['is_ai']; final_scores = state['final_scores']
    game_over_state = state['game_over_state']; replay_mode = state['replay_mode']
    current_replay_turn = state['current_replay_turn']; is_loaded_game = state['is_loaded_game']
    initial_racks_sgs = state.get('initial_racks_sgs', [[],[]])
    last_played_highlight_coords = state.get('last_played_highlight_coords', set()) # Use .get for safety
    selected_square = state['selected_square']; typing = state['typing']
    current_r = state['current_r']; current_c = state['current_c']
    typing_direction = state['typing_direction']
    typing_start = state['typing_start']
    preview_score_enabled = state['preview_score_enabled']
    current_preview_score = state['current_preview_score']
    is_solving_endgame = state['is_solving_endgame']
    is_batch_running = state['is_batch_running']
    current_batch_game_num = state['current_batch_game_num']
    total_batch_games = state['total_batch_games']
    showing_simulation_config = state['showing_simulation_config']
    simulation_config_inputs = state['simulation_config_inputs']
    simulation_config_active_input = state['simulation_config_active_input']
    specifying_rack = state['specifying_rack']
    specify_rack_inputs = state['specify_rack_inputs']
    specify_rack_active_input = state['specify_rack_active_input']
    specify_rack_original_racks = state['specify_rack_original_racks']
    confirming_override = state['confirming_override']
    exchanging = state['exchanging']; selected_tiles = state['selected_tiles']
    hinting = state['hinting']; hint_moves = state['hint_moves']
    selected_hint_index = state['selected_hint_index']
    showing_all_words = state['showing_all_words']; all_moves = state.get('all_moves', []) # Use .get for safety
    practice_target_moves = state.get('practice_target_moves', []) 
    paused_for_power_tile = state['paused_for_power_tile']
    current_power_tile = state['current_power_tile']
    number_checks = state['number_checks']
    paused_for_bingo_practice = state['paused_for_bingo_practice']
    all_words_scroll_offset = state['all_words_scroll_offset']
    showing_practice_end_dialog = state['showing_practice_end_dialog']
    practice_end_message = state['practice_end_message']
    dialog_x = state['dialog_x']; dialog_y = state['dialog_y']
    reason = state.get('reason', '')
    showing_stats = state['showing_stats']; stats_dialog_x = state['stats_dialog_x']
    stats_dialog_y = state['stats_dialog_y']; stats_scroll_offset = state['stats_scroll_offset']
    dropdown_open = state['dropdown_open']
    live_board_tile_counts = state['board_tile_counts'] # This is the Counter
    live_blanks_played_count = state.get('blanks_played_count', 0)
    best_exchange_for_hint = state.get('best_exchange_for_hint')
    best_exchange_score_for_hint = state.get('best_exchange_score_for_hint', -float('inf'))
    replay_initial_shuffled_bag = state.get('replay_initial_shuffled_bag', None) # For GCG replay
    # initial_racks is already unpacked, used for SGS replay if initial_racks_sgs is not present
    gaddag_loading_status_from_state = state.get('gaddag_loading_status', 'idle')


    bag_count = len(bag)
    drawn_rects = {} # Store all drawn rects here for event handling

    screen.fill(WHITE)
    draw_board_labels(screen, ui_font)

    # Determine what state to display (live, replay, etc.)
    tiles_to_display = tiles
    blanks_to_display = blanks
    racks_to_display = racks
    scores_to_display = scores
    turn_to_display = turn
    board_counts_for_rem_tiles = live_board_tile_counts # Default to live game's counts
    blanks_played_count_for_rem_tiles = live_blanks_played_count # Default to live

    if replay_mode:
        if is_loaded_game and move_history and move_history[0].get('move_type') == "gcg_placeholder": # GCG Replay
            sim_board, sim_tiles_replay, sim_racks_replay, sim_blanks_replay, sim_scores_replay, sim_bag_replay, sim_turn_replay, sim_first_play_replay, _, _ = simulate_game_up_to(
                current_replay_turn, move_history, replay_initial_shuffled_bag
            )
            tiles_to_display = sim_tiles_replay
            blanks_to_display = sim_blanks_replay
            racks_to_display = sim_racks_replay
            scores_to_display = sim_scores_replay
            turn_to_display = sim_turn_replay
            # For GCG replay, remaining tiles needs careful calculation based on simulated state
            temp_board_counts_replay_gcg = Counter()
            temp_blanks_played_replay_gcg = 0
            for r_rem_gcg in range(GRID_SIZE):
                for c_rem_gcg in range(GRID_SIZE):
                    tile_on_board_gcg = sim_tiles_replay[r_rem_gcg][c_rem_gcg]
                    if tile_on_board_gcg:
                        temp_board_counts_replay_gcg[tile_on_board_gcg] += 1
                        if (r_rem_gcg, c_rem_gcg) in sim_blanks_replay:
                            temp_blanks_played_replay_gcg +=1 # Count blanks on board
            board_counts_for_rem_tiles = temp_board_counts_replay_gcg
            blanks_played_count_for_rem_tiles = temp_blanks_played_replay_gcg

        else: # SGS Replay (or new game replay before first move)
            current_move_history_for_replay_sgs = move_history 
            current_initial_racks_sgs_for_replay_sgs = initial_racks_sgs 
            
            if not current_move_history_for_replay_sgs and not current_initial_racks_sgs_for_replay_sgs[0] and not current_initial_racks_sgs_for_replay_sgs[1]:
                 # Case: New game, no moves yet, but replay mode somehow active (e.g. after "Replay" from game over of a new game)
                 # Show initial empty board state, use initial racks if available from a fresh game setup
                current_initial_shuffled_bag_draw = state.get('initial_shuffled_bag_order_sgs', []) 
                current_initial_racks_for_new_replay_draw = state.get('initial_racks_sgs', [[],[]]) 
                if not current_initial_shuffled_bag_draw: # Fallback if no shuffled bag order
                    current_initial_shuffled_bag_draw = create_standard_bag() # Won't be shuffled unless explicitly done

                sgs_replay_state_new_game_draw = get_replay_state(current_replay_turn, move_history, current_initial_racks_for_new_replay_draw) 
                tiles_to_display = sgs_replay_state_new_game_draw['tiles']
                blanks_to_display = sgs_replay_state_new_game_draw['blanks']
                racks_to_display = sgs_replay_state_new_game_draw['racks']
                scores_to_display = sgs_replay_state_new_game_draw['scores']
                board_counts_for_rem_tiles = sgs_replay_state_new_game_draw['board_tile_counts']
                blanks_played_count_for_rem_tiles = sgs_replay_state_new_game_draw['blanks_played_count']
                if current_replay_turn == 0:
                    turn_to_display = 1 # Start of game
                elif current_replay_turn > 0 and current_replay_turn <= len(move_history):
                     # Get turn from the *next* turn field of the *previous* move
                    turn_to_display = move_history[current_replay_turn-1].get('next_turn_after_move', 1)


            elif current_move_history_for_replay_sgs : # Standard SGS replay with history
                sgs_replay_state = get_replay_state(current_replay_turn, current_move_history_for_replay_sgs, current_initial_racks_sgs_for_replay_sgs)
                tiles_to_display = sgs_replay_state['tiles']
                blanks_to_display = sgs_replay_state['blanks']
                racks_to_display = sgs_replay_state['racks']
                scores_to_display = sgs_replay_state['scores']
                board_counts_for_rem_tiles = sgs_replay_state['board_tile_counts']
                blanks_played_count_for_rem_tiles = sgs_replay_state['blanks_played_count']
                if current_replay_turn == 0:
                    turn_to_display = 1 # Start of game
                elif current_replay_turn > 0 and current_replay_turn <= len(move_history):
                    turn_to_display = move_history[current_replay_turn-1].get('next_turn_after_move', 1)

            else: # Fallback if no history but initial_racks_sgs might exist (e.g. loaded game not yet played)
                tiles_to_display = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                blanks_to_display = set()
                racks_to_display = initial_racks_sgs if initial_racks_sgs and (initial_racks_sgs[0] or initial_racks_sgs[1]) else [[],[]]
                scores_to_display = [0,0]
                turn_to_display = 1
                board_counts_for_rem_tiles = Counter()
                blanks_played_count_for_rem_tiles = 0


    elif game_over_state:
        scores_to_display = final_scores if final_scores else scores # Show final scores if available
        # For remaining tiles, use the live game state at the point game over was triggered
        board_counts_for_rem_tiles = live_board_tile_counts
        blanks_played_count_for_rem_tiles = live_blanks_played_count
    # Else, it's live play, defaults are already set

    # Draw Board and Tiles
    for r_draw in range(GRID_SIZE):
        for c_draw in range(GRID_SIZE):
            square_rect_draw = pygame.Rect(40 + c_draw * SQUARE_SIZE, 40 + r_draw * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, board[r_draw][c_draw], square_rect_draw)
            pygame.draw.rect(screen, BLACK, square_rect_draw, 1)
            if tiles_to_display[r_draw][c_draw]:
                tile_char_draw = tiles_to_display[r_draw][c_draw]
                is_blank_on_board_draw = (r_draw, c_draw) in blanks_to_display
                is_last_played_draw = (r_draw, c_draw) in last_played_highlight_coords and not replay_mode 
                
                tile_bg_color_draw = PALE_YELLOW if is_last_played_draw else GREEN
                tile_rect_draw = pygame.Rect(square_rect_draw.left + 2, square_rect_draw.top + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4)
                pygame.draw.rect(screen, tile_bg_color_draw, tile_rect_draw)
                
                text_surf_draw = None
                if is_blank_on_board_draw:
                    center_draw = tile_rect_draw.center
                    radius_draw = SQUARE_SIZE // 2 - 3
                    pygame.draw.circle(screen, BLACK, center_draw, radius_draw) # Black circle for blank
                    text_surf_draw = TILE_LETTER_CACHE['blank_assigned'].get(tile_char_draw)
                    if text_surf_draw:
                        text_rect_draw = text_surf_draw.get_rect(center=center_draw)
                        screen.blit(text_surf_draw, text_rect_draw)
                else:
                    text_surf_draw = TILE_LETTER_CACHE['regular'].get(tile_char_draw)
                    if text_surf_draw:
                        text_rect_draw = text_surf_draw.get_rect(center=tile_rect_draw.center)
                        screen.blit(text_surf_draw, text_rect_draw)

            if selected_square == (r_draw, c_draw) and typing:
                pygame.draw.rect(screen, HIGHLIGHT_BLUE, square_rect_draw, 3)

    # Highlight last played word in Replay Mode (SGS)
    if replay_mode and current_replay_turn > 0 and current_replay_turn <= len(move_history):
        last_move_data_hl_draw = move_history[current_replay_turn - 1]
        if last_move_data_hl_draw.get('move_type') == 'place':
            newly_placed_details_for_highlight_draw = last_move_data_hl_draw.get('newly_placed_details')
            if newly_placed_details_for_highlight_draw: # Check if it exists
                highlight_coords_set_replay_draw = set()
                # Highlight all words formed by newly_placed_details
                words_formed_hl_draw = find_all_words_formed_cython(newly_placed_details_for_highlight_draw, tiles_to_display)
                if words_formed_hl_draw:
                    for word_detail_hl_draw in words_formed_hl_draw:
                        for r_hl_d, c_hl_d, _ in word_detail_hl_draw:
                            highlight_coords_set_replay_draw.add((r_hl_d, c_hl_d))
                else: # Fallback if find_all_words_formed returns empty (e.g. single tile play not forming a 2+ letter word)
                      # Highlight just the newly placed tiles themselves.
                    for r_hl_d_np, c_hl_d_np, _ in newly_placed_details_for_highlight_draw:
                        highlight_coords_set_replay_draw.add((r_hl_d_np, c_hl_d_np))
                
                for r_hl_d2, c_hl_d2 in highlight_coords_set_replay_draw:
                    pygame.draw.rect(screen, YELLOW, (40 + c_hl_d2 * SQUARE_SIZE, 40 + r_hl_d2 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)
            elif 'positions' in last_move_data_hl_draw: # Fallback for GCG-like data if newly_placed_details is missing
                positions_in_move_hl_draw = last_move_data_hl_draw.get('positions', [])
                for r_hl_d3, c_hl_d3, _ in positions_in_move_hl_draw:
                     pygame.draw.rect(screen, YELLOW, (40 + c_hl_d3 * SQUARE_SIZE, 40 + r_hl_d3 * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)


    # Draw Racks
    p1_alpha_rect_draw, p1_rand_rect_draw = draw_rack(1, racks_to_display[0] if len(racks_to_display) > 0 else [], scores_to_display, turn_to_display, player_names, dragged_tile, drag_pos, practice_mode) # Pass full scores
    p2_alpha_rect_draw, p2_rand_rect_draw = draw_rack(2, racks_to_display[1] if len(racks_to_display) > 1 else [], scores_to_display, turn_to_display, player_names, dragged_tile, drag_pos, practice_mode) # Pass full scores
    drawn_rects['p1_alpha_rect'] = p1_alpha_rect_draw
    drawn_rects['p1_rand_rect'] = p1_rand_rect_draw
    drawn_rects['p2_alpha_rect'] = p2_alpha_rect_draw
    drawn_rects['p2_rand_rect'] = p2_rand_rect_draw
    
    # Draw Remaining Tiles (conditionally, not during active dialogs that obscure it)
    if not (hinting or showing_all_words or exchanging or specifying_rack or showing_simulation_config or game_over_state or showing_practice_end_dialog):
        current_player_idx_for_rem_tiles_draw = turn_to_display - 1
        rack_for_rem_tiles_calc_draw = []
        if 0 <= current_player_idx_for_rem_tiles_draw < len(racks_to_display):
            rack_for_rem_tiles_calc_draw = racks_to_display[current_player_idx_for_rem_tiles_draw]
        
        remaining_for_display_draw = get_remaining_tiles(
            rack_for_rem_tiles_calc_draw, 
            tiles_to_display, # Use the tiles currently being displayed
            blanks_to_display, # Use the blanks currently being displayed
            blanks_played_count_for_rem_tiles # Use the count relevant to the displayed state
        )
        draw_remaining_tiles(remaining_for_display_draw, turn_to_display)

    # Draw Scoreboard
    history_to_draw_sb = move_history[:current_replay_turn] if replay_mode else move_history
    is_final_turn_in_replay_sb = replay_mode and current_replay_turn == len(move_history)
    final_scores_for_sb = final_scores if game_over_state or is_final_turn_in_replay_sb else None
    
    sb_x_draw = BOARD_SIZE + 275; sb_y_draw = 40 
    sb_w_draw = max(200, WINDOW_WIDTH - sb_x_draw - 20); sb_h_draw = WINDOW_HEIGHT - 80 
    if sb_x_draw + sb_w_draw > WINDOW_WIDTH - 10: sb_w_draw = WINDOW_WIDTH - sb_x_draw - 10
    if sb_w_draw < 150: sb_x_draw = WINDOW_WIDTH - 160; sb_w_draw = 150
    
    drawn_rects['scoreboard_rect'] = pygame.Rect(sb_x_draw, sb_y_draw, sb_w_draw, sb_h_draw) # Store for scroll
    draw_scoreboard(screen, history_to_draw_sb, scroll_offset, scores_to_display, is_ai, player_names, final_scores_for_sb, game_over_state or is_final_turn_in_replay_sb)

    # Draw Typing Cursor
    if typing and selected_square:
        cursor_x_typing_draw = 40 + current_c * SQUARE_SIZE + SQUARE_SIZE // 2 
        cursor_y_typing_draw = 40 + current_r * SQUARE_SIZE + SQUARE_SIZE - 5 
        if typing_direction == "right":
            pygame.draw.line(screen, BLACK, (cursor_x_typing_draw - 5, cursor_y_typing_draw), (cursor_x_typing_draw + 5, cursor_y_typing_draw), 2)
        else: # down
            pygame.draw.line(screen, BLACK, (cursor_x_typing_draw, cursor_y_typing_draw - 5), (cursor_x_typing_draw, cursor_y_typing_draw + 5), 2)
    
    # Draw Options Menu, Suggest Button, Simulate Button, Preview Score (conditionally)
    suggest_rect_base_draw = None; simulate_button_rect_draw = None; preview_checkbox_rect_draw = None 
    is_human_turn_or_paused_draw = (0 <= turn-1 < len(is_ai)) and (not is_ai[turn-1] or paused_for_power_tile or paused_for_bingo_practice) 

    if not game_over_state and not replay_mode and not is_batch_running:
        options_rect_base_ui, dropdown_rects_base_ui = draw_options_menu(turn, dropdown_open, bag_count, replay_mode)
        drawn_rects['options_rect_base'] = options_rect_base_ui
        drawn_rects['dropdown_rects_base'] = dropdown_rects_base_ui

        if is_human_turn_or_paused_draw and not (exchanging or hinting or showing_all_words or specifying_rack or showing_simulation_config):
            suggest_rect_base_draw = draw_suggest_button()
            drawn_rects['suggest_rect_base'] = suggest_rect_base_draw

            if suggest_rect_base_draw: # Only draw simulate if suggest is drawn
                simulate_button_rect_draw = pygame.Rect(suggest_rect_base_draw.x, suggest_rect_base_draw.bottom + BUTTON_GAP, OPTIONS_WIDTH, OPTIONS_HEIGHT)
                hover_sim_draw = simulate_button_rect_draw.collidepoint(pygame.mouse.get_pos()) 
                color_sim_draw = BUTTON_HOVER if hover_sim_draw else BUTTON_COLOR 
                pygame.draw.rect(screen, color_sim_draw, simulate_button_rect_draw)
                simulate_text_surf_draw = button_font.render("Simulate", True, BLACK) 
                simulate_text_rect_render_draw = simulate_text_surf_draw.get_rect(center=simulate_button_rect_draw.center) 
                screen.blit(simulate_text_surf_draw, simulate_text_rect_render_draw)
                drawn_rects['simulate_button_rect'] = simulate_button_rect_draw
            
            # Score Preview Checkbox (position relative to randomize button of current player)
            relevant_rand_rect_ui_draw = p1_rand_rect_draw if turn == 1 else p2_rand_rect_draw 
            if relevant_rand_rect_ui_draw:
                preview_checkbox_height_ui_draw = 20 
                checkbox_x_ui_draw = relevant_rand_rect_ui_draw.left 
                checkbox_y_ui_draw = relevant_rand_rect_ui_draw.top - preview_checkbox_height_ui_draw - BUTTON_GAP 
                preview_checkbox_rect_draw = pygame.Rect(checkbox_x_ui_draw, checkbox_y_ui_draw, 20, preview_checkbox_height_ui_draw)
                pygame.draw.rect(screen, WHITE, preview_checkbox_rect_draw)
                pygame.draw.rect(screen, BLACK, preview_checkbox_rect_draw, 1)
                if preview_score_enabled:
                    pygame.draw.line(screen, BLACK, (checkbox_x_ui_draw + 3, checkbox_y_ui_draw + preview_checkbox_height_ui_draw // 2), (checkbox_x_ui_draw + preview_checkbox_height_ui_draw // 2 -2 , checkbox_y_ui_draw + preview_checkbox_height_ui_draw -3), 2)
                    pygame.draw.line(screen, BLACK, (checkbox_x_ui_draw + preview_checkbox_height_ui_draw // 2 -2, checkbox_y_ui_draw + preview_checkbox_height_ui_draw -3), (checkbox_x_ui_draw + preview_checkbox_height_ui_draw -3, checkbox_y_ui_draw + 3 ), 2)
                
                label_text_ui_draw = "Score Preview: " 
                label_surf_ui_draw = ui_font.render(label_text_ui_draw, True, BLACK) 
                label_x_ui_draw = checkbox_x_ui_draw + 25 
                label_y_ui_draw = checkbox_y_ui_draw + (preview_checkbox_rect_draw.height - label_surf_ui_draw.get_height()) // 2 
                screen.blit(label_surf_ui_draw, (label_x_ui_draw, label_y_ui_draw))
                drawn_rects['preview_checkbox_rect'] = preview_checkbox_rect_draw

                if preview_score_enabled and typing and word_positions:
                    score_text_ui_draw = str(current_preview_score) 
                    score_surf_ui_draw = ui_font.render(score_text_ui_draw, True, BLACK) 
                    score_x_ui_draw = label_x_ui_draw + label_surf_ui_draw.get_width() + 2 
                    score_y_ui_draw = label_y_ui_draw 
                    screen.blit(score_surf_ui_draw, (score_x_ui_draw, score_y_ui_draw))


    # Draw Loading Indicator if GADDAG is loading
    if gaddag_loading_status_from_state == 'loading' or (gaddag_loading_status_from_state == 'idle' and not gaddag_loaded_event.is_set()):
         draw_loading_indicator(sb_x_draw, sb_y_draw, sb_w_draw) # Pass scoreboard rect for positioning

    # Draw Batch Game Indicator
    if is_batch_running and not game_over_state :
        batch_text_ind_draw = f"Running Game: {current_batch_game_num} / {total_batch_games}" 
        batch_surf_ind_draw = ui_font.render(batch_text_ind_draw, True, BLUE) 
        indicator_center_x_ind_draw = sb_x_draw + sb_w_draw // 2 
        indicator_top_y_ind_draw = sb_y_draw - batch_surf_ind_draw.get_height() - 5 
        batch_rect_ind_draw = batch_surf_ind_draw.get_rect(centerx=indicator_center_x_ind_draw, top=max(5, indicator_top_y_ind_draw)) 
        screen.blit(batch_surf_ind_draw, batch_rect_ind_draw)
    
    # Draw Endgame Solving Indicator
    if is_solving_endgame:
        draw_endgame_solving_indicator()

    # --- Dialogs on top ---
    if showing_practice_end_dialog:
        rects_ped_draw = draw_practice_end_dialog(practice_end_message)
        drawn_rects['practice_play_again_rect'] = rects_ped_draw[0]
        drawn_rects['practice_main_menu_rect'] = rects_ped_draw[1]
        drawn_rects['practice_quit_rect'] = rects_ped_draw[2]

    if specifying_rack:
        p1_name_disp_sr_draw = player_names[0] if player_names and player_names[0] else "Player 1" 
        p2_name_disp_sr_draw = player_names[1] if player_names and player_names[1] else "Player 2" 
        rects_sr_draw = draw_specify_rack_dialog(p1_name_disp_sr_draw, p2_name_disp_sr_draw, specify_rack_inputs, specify_rack_active_input, specify_rack_original_racks) 
        drawn_rects['p1_input_rect_sr'] = rects_sr_draw[0]
        drawn_rects['p2_input_rect_sr'] = rects_sr_draw[1]
        drawn_rects['p1_reset_rect_sr'] = rects_sr_draw[2]
        drawn_rects['p2_reset_rect_sr'] = rects_sr_draw[3]
        drawn_rects['confirm_rect_sr'] = rects_sr_draw[4]
        drawn_rects['cancel_rect_sr'] = rects_sr_draw[5]

    if confirming_override:
        rects_ov_draw = draw_override_confirmation_dialog()
        drawn_rects['go_back_rect_ov'] = rects_ov_draw[0]
        drawn_rects['override_rect_ov'] = rects_ov_draw[1]
        
    if showing_simulation_config:
        rects_sim_cfg_draw = draw_simulation_config_dialog(simulation_config_inputs, simulation_config_active_input)
        drawn_rects['sim_input_rects'] = rects_sim_cfg_draw[0] # This is a list of rects
        drawn_rects['sim_simulate_rect'] = rects_sim_cfg_draw[1]
        drawn_rects['sim_cancel_rect'] = rects_sim_cfg_draw[2]

    if exchanging:
        current_rack_for_exchange_draw = racks[turn-1] if 0 <= turn-1 < len(racks) else [] 
        dialog_rects_exch_draw = draw_exchange_dialog(current_rack_for_exchange_draw, selected_tiles) 
        drawn_rects['tile_rects'] = dialog_rects_exch_draw[0] # List of tile rects
        drawn_rects['exchange_button_rect'] = dialog_rects_exch_draw[1]
        drawn_rects['cancel_button_rect'] = dialog_rects_exch_draw[2]

    if hinting:
        is_sim_res_hint_draw = bool(hint_moves and isinstance(hint_moves[0], dict) and 'final_score' in hint_moves[0]) 
        dialog_elements_hint = draw_hint_dialog(hint_moves, selected_hint_index, is_simulation_result=is_sim_res_hint_draw, best_exchange_tiles=best_exchange_for_hint, best_exchange_score=best_exchange_score_for_hint)
        drawn_rects['play_button_rect'] = dialog_elements_hint.get('play_button_rect') # Key for play/exchange
        drawn_rects['all_words_button_rect'] = dialog_elements_hint.get('all_words_button_rect')
        drawn_rects['ok_button_rect'] = dialog_elements_hint.get('ok_button_rect')
        drawn_rects['hint_rects'] = dialog_elements_hint.get('hint_rects', []) # List of (rect, index)

    if showing_all_words:
        moves_for_all_dlg_draw = all_moves 
        if practice_mode == "eight_letter":
            moves_for_all_dlg_draw = practice_target_moves if practice_target_moves else []
        elif practice_mode == "power_tiles" and paused_for_power_tile:
            moves_for_all_dlg_draw = sorted([m for m in all_moves if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)], key=lambda m: m.get('score',0), reverse=True)
        elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice:
            moves_for_all_dlg_draw = sorted([m for m in all_moves if m.get('is_bingo', False)], key=lambda m: m.get('score',0), reverse=True)
        
        current_selected_hint_index_for_aw_dlg = selected_hint_index
        if not moves_for_all_dlg_draw:
            current_selected_hint_index_for_aw_dlg = None
        elif selected_hint_index is None or selected_hint_index >= len(moves_for_all_dlg_draw):
            current_selected_hint_index_for_aw_dlg = 0 if moves_for_all_dlg_draw else None
        
        all_words_dialog_data = draw_all_words_dialog(
            moves_for_all_dlg_draw, 
            current_selected_hint_index_for_aw_dlg, 
            all_words_scroll_offset
        )
        drawn_rects['all_words_play_rect'] = all_words_dialog_data.get('all_words_play_rect')
        drawn_rects['all_words_ok_rect'] = all_words_dialog_data.get('all_words_ok_rect')
        drawn_rects['all_words_rects'] = all_words_dialog_data.get('all_words_clickable_items', []) 
        drawn_rects['all_words_content_height'] = all_words_dialog_data.get('all_words_content_height', 0)


    if game_over_state:
        rects_go_ui = draw_game_over_dialog(dialog_x, dialog_y, final_scores, reason, player_names) 
        drawn_rects['save_rect'] = rects_go_ui[0]
        drawn_rects['quit_rect'] = rects_go_ui[1]
        drawn_rects['replay_rect'] = rects_go_ui[2]
        drawn_rects['play_again_rect'] = rects_go_ui[3]
        drawn_rects['stats_rect'] = rects_go_ui[4]
        drawn_rects['game_over_dialog_rect'] = pygame.Rect(dialog_x, dialog_y, DIALOG_WIDTH, DIALOG_HEIGHT)


    if showing_stats:
        stats_ok_rect_ui, total_content_h_stats_ui = draw_stats_dialog(stats_dialog_x, stats_dialog_y, player_names, final_scores, tiles_to_display, stats_scroll_offset) 
        drawn_rects['stats_ok_button_rect'] = stats_ok_rect_ui
        drawn_rects['stats_total_content_height'] = total_content_h_stats_ui # For scrolling
        drawn_rects['stats_dialog_rect'] = pygame.Rect(stats_dialog_x, stats_dialog_y, 480, 600)


    # Draw Replay Controls if in replay_mode and not game over screen (which has its own replay button)
    if replay_mode and not game_over_state:
        replay_start_rect_ui = state['replay_start_rect']; replay_prev_rect_ui = state['replay_prev_rect'] 
        replay_next_rect_ui = state['replay_next_rect']; replay_end_rect_ui = state['replay_end_rect'] 
        replay_controls_ui = [(replay_start_rect_ui, "start"), (replay_prev_rect_ui, "prev"), (replay_next_rect_ui, "next"), (replay_end_rect_ui, "end")] 
        for rect_rc_ui, icon_type_rc_ui in replay_controls_ui: 
            if rect_rc_ui: # Ensure rect is not None
                hover_rc_ui = rect_rc_ui.collidepoint(pygame.mouse.get_pos()) 
                color_rc_ui = BUTTON_HOVER if hover_rc_ui else BUTTON_COLOR 
                pygame.draw.rect(screen, color_rc_ui, rect_rc_ui)
                draw_replay_icon(screen, rect_rc_ui, icon_type_rc_ui)
        # Store these in drawn_rects if they are not already updated in state by initialize_game
        drawn_rects['replay_start_rect'] = replay_start_rect_ui
        drawn_rects['replay_prev_rect'] = replay_prev_rect_ui
        drawn_rects['replay_next_rect'] = replay_next_rect_ui
        drawn_rects['replay_end_rect'] = replay_end_rect_ui


    # Draw dragged tile last so it's on top
    if dragged_tile and drag_pos:
        player_idx_drag_draw = dragged_tile[0]-1 
        tile_val_drag_draw = None 
        current_racks_for_drag_draw = racks_to_display 
        if 0 <= player_idx_drag_draw < len(current_racks_for_drag_draw) and \
           current_racks_for_drag_draw[player_idx_drag_draw] is not None and \
           0 <= dragged_tile[1] < len(current_racks_for_drag_draw[player_idx_drag_draw]):
            tile_val_drag_draw = current_racks_for_drag_draw[player_idx_drag_draw][dragged_tile[1]]
        
        if tile_val_drag_draw:
            center_x_drag_draw = drag_pos[0] - drag_offset[0] 
            center_y_drag_draw = drag_pos[1] - drag_offset[1] 
            draw_x_drag_draw = center_x_drag_draw - TILE_WIDTH // 2 
            draw_y_drag_draw = center_y_drag_draw - TILE_HEIGHT // 2 
            
            pygame.draw.rect(screen, GREEN, (draw_x_drag_draw, draw_y_drag_draw, TILE_WIDTH, TILE_HEIGHT))
            text_surf_drag_tile = None 
            if tile_val_drag_draw == ' ':
                radius_drag_tile_draw = TILE_WIDTH // 2 - 2 
                pygame.draw.circle(screen, BLACK, (center_x_drag_draw, center_y_drag_draw), radius_drag_tile_draw)
                text_surf_drag_tile = TILE_LETTER_CACHE['blank'].get('?')
                if text_surf_drag_tile:
                    text_rect_drag_draw = text_surf_drag_tile.get_rect(center=(center_x_drag_draw, center_y_drag_draw)) 
                    screen.blit(text_surf_drag_tile, text_rect_drag_draw)
            else:
                text_surf_drag_tile = TILE_LETTER_CACHE['regular'].get(tile_val_drag_draw)
                if text_surf_drag_tile:
                    text_rect_drag_draw = text_surf_drag_tile.get_rect(center=(center_x_drag_draw, center_y_drag_draw))
                    screen.blit(text_surf_drag_tile, text_rect_drag_draw)
    
    pygame.display.flip()
    return drawn_rects











def process_game_events(state, drawn_rects):
    """Processes Pygame events and updates game state."""
    global DAWG # Ensure DAWG is accessible for is_valid_play_cython
    # (Other globals you might need access to within this function, if any, though most state is passed in)

    # Unpack frequently used control flow flags
    running_inner = state['running_inner']
    return_to_mode_selection = state['return_to_mode_selection']
    pyperclip_available = state.get('pyperclip_available', False) # Safely get
    pyperclip = state.get('pyperclip', None) # Safely get

    # Unpack cursor and board state variables
    current_r = state.get('current_r')
    current_c = state.get('current_c')
    typing_direction = state.get('typing_direction')
    typing_start = state.get('typing_start')
    board_tile_counts = state['board_tile_counts'] # This is the Counter, modified by play_hint_move
    practice_probability_max_index = state.get('practice_probability_max_index')
    blanks_played_count = state.get('blanks_played_count', 0) # Game total

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if state.get('is_batch_running'): # If in batch, set flag to stop batch
                state['batch_stop_requested'] = True
                # Don't immediately set running_inner to False, let batch complete current game if desired by logic
            else:
                running_inner = False # For non-batch, quit directly

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down_result = handle_mouse_down_event(event, state, drawn_rects)
            state.update(mouse_down_result) # Update state with any changes from mouse_down
            # Update local vars that might have changed in state if they are used directly below
            running_inner = state['running_inner']
            return_to_mode_selection = state['return_to_mode_selection']
            # other flags like 'typing', 'selected_square' etc. are now in state

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: # Left mouse button up
                if state.get('dragged_tile'):
                    x_up, y_up = event.pos
                    player_idx_up = state['dragged_tile'][0] - 1
                    rack_y_up = BOARD_SIZE + 80 if state['dragged_tile'][0] == 1 else BOARD_SIZE + 150
                    rack_width_calc_up = 7 * (TILE_WIDTH + TILE_GAP) - TILE_GAP
                    replay_area_end_x_up = 10 + 4 * (REPLAY_BUTTON_WIDTH + REPLAY_BUTTON_GAP)
                    min_rack_start_x_up = replay_area_end_x_up + BUTTON_GAP + 20
                    rack_start_x_calc_up = max(min_rack_start_x_up, (BOARD_SIZE - rack_width_calc_up) // 2)
                    rack_area_rect_up = pygame.Rect(rack_start_x_calc_up, rack_y_up, rack_width_calc_up, TILE_HEIGHT)

                    if not rack_area_rect_up.collidepoint(x_up, y_up): # Dropped outside rack area
                        # For now, simply return tile to original position (no board drop yet)
                        pass # Tile will snap back as drag_pos is reset
                    else: # Dropped on rack area, reorder
                        player_rack_up = state['racks'][player_idx_up]
                        rack_len_up = len(player_rack_up)
                        insert_idx_raw_up = get_insertion_index(x_up, rack_start_x_calc_up, rack_len_up)
                        original_tile_idx_up = state['dragged_tile'][1]

                        if 0 <= original_tile_idx_up < len(player_rack_up):
                            tile_to_move_up = player_rack_up.pop(original_tile_idx_up)
                            insert_idx_adjusted_up = insert_idx_raw_up
                            if insert_idx_raw_up > original_tile_idx_up: # Accounts for shift due to pop
                                insert_idx_adjusted_up -=1
                            insert_idx_final_up = max(0, min(insert_idx_adjusted_up, len(player_rack_up)))
                            player_rack_up.insert(insert_idx_final_up, tile_to_move_up)

                    state['dragged_tile'] = None; state['drag_pos'] = None
                state['stats_dialog_dragging'] = False
                state['dragging'] = False

        elif event.type == pygame.MOUSEMOTION:
            if state.get('dragged_tile') and state.get('drag_pos'):
                state['drag_pos'] = event.pos
            elif state.get('stats_dialog_dragging'):
                new_x = event.pos[0] - state['stats_dialog_drag_offset'][0]
                new_y = event.pos[1] - state['stats_dialog_drag_offset'][1]
                state['stats_dialog_x'] = max(0, min(new_x, WINDOW_WIDTH - 480))
                state['stats_dialog_y'] = max(0, min(new_y, WINDOW_HEIGHT - 600))
            elif state.get('dragging'): # For game over dialog
                new_x = event.pos[0] - state['drag_offset'][0]
                new_y = event.pos[1] - state['drag_offset'][1]
                state['dialog_x'] = max(0, min(new_x, WINDOW_WIDTH - DIALOG_WIDTH))
                state['dialog_y'] = max(0, min(new_y, WINDOW_HEIGHT - DIALOG_HEIGHT))

        elif event.type == pygame.MOUSEWHEEL:
            dialog_rect_all_wheel = pygame.Rect((WINDOW_WIDTH - ALL_WORDS_DIALOG_WIDTH) // 2, (WINDOW_HEIGHT - ALL_WORDS_DIALOG_HEIGHT) // 2, ALL_WORDS_DIALOG_WIDTH, ALL_WORDS_DIALOG_HEIGHT)
            if state.get('showing_all_words') and dialog_rect_all_wheel.collidepoint(pygame.mouse.get_pos()):
                moves_for_scroll_wheel = []
                if state['practice_mode'] == "eight_letter": moves_for_scroll_wheel = state.get('practice_target_moves', [])
                elif state['practice_mode'] == "power_tiles" and state.get('paused_for_power_tile'): moves_for_scroll_wheel = sorted([m for m in state.get('all_moves',[]) if any(letter == state.get('current_power_tile') for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), state.get('number_checks',[]))], key=lambda m: m.get('score',0), reverse=True)
                elif state['practice_mode'] == "bingo_bango_bongo" and state.get('paused_for_bingo_practice'): moves_for_scroll_wheel = sorted([m for m in state.get('all_moves',[]) if m.get('is_bingo', False)], key=lambda m: m.get('score',0), reverse=True)
                else: moves_for_scroll_wheel = state.get('all_moves', [])
                
                content_height_wheel = len(moves_for_scroll_wheel) * 30
                header_height_wheel = 40
                button_area_height_wheel = BUTTON_HEIGHT + 30
                visible_content_height_wheel = ALL_WORDS_DIALOG_HEIGHT - header_height_wheel - button_area_height_wheel
                if content_height_wheel > visible_content_height_wheel:
                    max_scroll_wheel = content_height_wheel - visible_content_height_wheel
                    state['all_words_scroll_offset'] -= event.y * SCROLL_SPEED
                    state['all_words_scroll_offset'] = max(0, min(state['all_words_scroll_offset'], max_scroll_wheel))
            elif state.get('showing_stats'):
                stats_dialog_rect_wheel = pygame.Rect(state['stats_dialog_x'], state['stats_dialog_y'], 480, 600) # Use fixed stats dialog dims
                if stats_dialog_rect_wheel.collidepoint(pygame.mouse.get_pos()):
                    padding_wheel = 10
                    button_area_height_stats_wheel = BUTTON_HEIGHT + padding_wheel * 2
                    visible_content_height_stats_wheel = 600 - padding_wheel * 2 - button_area_height_stats_wheel
                    stats_total_content_height_wheel = drawn_rects.get('stats_total_content_height', 0)
                    if stats_total_content_height_wheel > visible_content_height_stats_wheel:
                        max_scroll_stats_wheel = stats_total_content_height_wheel - visible_content_height_stats_wheel
                        state['stats_scroll_offset'] -= event.y * SCROLL_SPEED
                        state['stats_scroll_offset'] = max(0, min(state['stats_scroll_offset'], max_scroll_stats_wheel))
            else: # Scoreboard scroll
                sb_x_wheel = BOARD_SIZE + 275; sb_y_wheel = 40
                sb_w_wheel = max(200, WINDOW_WIDTH - sb_x_wheel - 20)
                sb_h_wheel = WINDOW_HEIGHT - 80
                if sb_x_wheel + sb_w_wheel > WINDOW_WIDTH - 10: sb_w_wheel = WINDOW_WIDTH - sb_x_wheel - 10
                if sb_w_wheel < 150: sb_x_wheel = WINDOW_WIDTH - 160; sb_w_wheel = 150
                scoreboard_rect_wheel = pygame.Rect(sb_x_wheel, sb_y_wheel, sb_w_wheel, sb_h_wheel)
                if scoreboard_rect_wheel.collidepoint(pygame.mouse.get_pos()):
                    history_to_draw_wheel = state['move_history'][:state['current_replay_turn']] if state['replay_mode'] else state['move_history']
                    history_len_wheel = len(history_to_draw_wheel)
                    total_content_height_sb_wheel = history_len_wheel * 20 # Approx line height
                    is_final_turn_in_replay_wheel = state['replay_mode'] and state['current_replay_turn'] == len(state['move_history'])
                    scoreboard_height_disp_wheel = sb_h_wheel
                    if (state['game_over_state'] or is_final_turn_in_replay_wheel) and state.get('final_scores'):
                         scoreboard_height_disp_wheel -= (ui_font.get_linesize() + 10) # Space for final scores line

                    if total_content_height_sb_wheel > scoreboard_height_disp_wheel:
                        max_scroll_sb_wheel = total_content_height_sb_wheel - scoreboard_height_disp_wheel
                        state['scroll_offset'] -= event.y * SCROLL_SPEED
                        state['scroll_offset'] = max(0, min(state['scroll_offset'], max_scroll_sb_wheel))

        elif event.type == pygame.KEYDOWN:
            if state.get('specifying_rack') and state.get('specify_rack_active_input') is not None:
                idx_sr_key = state['specify_rack_active_input']
                if event.key == pygame.K_RETURN:
                    confirm_rect_sr_key = drawn_rects.get('confirm_rect_sr')
                    if confirm_rect_sr_key: # Simulate click on confirm
                        mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': confirm_rect_sr_key.center})
                        state.update(handle_mouse_down_event(mock_event, state, drawn_rects))
                elif event.key == pygame.K_BACKSPACE:
                    state['specify_rack_inputs'][idx_sr_key] = state['specify_rack_inputs'][idx_sr_key][:-1]
                elif event.key == pygame.K_TAB:
                    state['specify_rack_active_input'] = 1 - idx_sr_key # Toggle active input
                elif event.key == pygame.K_ESCAPE: # Allow Esc to cancel specify rack
                    state['specifying_rack'] = False; state['specify_rack_inputs'] = ["", ""]; state['specify_rack_active_input'] = None; state['specify_rack_original_racks'] = [[], []]
                else:
                    char_sr_key = event.unicode.upper()
                    if (char_sr_key.isalpha() or char_sr_key == '?' or char_sr_key == ' ') and len(state['specify_rack_inputs'][idx_sr_key]) < 7:
                        state['specify_rack_inputs'][idx_sr_key] += char_sr_key
            elif state.get('showing_simulation_config') and state.get('simulation_config_active_input') is not None:
                idx_sim_key = state['simulation_config_active_input']
                if event.key == pygame.K_RETURN:
                    sim_simulate_rect_key = drawn_rects.get('sim_simulate_rect')
                    if sim_simulate_rect_key:
                        mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': sim_simulate_rect_key.center})
                        state.update(handle_mouse_down_event(mock_event, state, drawn_rects))
                elif event.key == pygame.K_BACKSPACE:
                    state['simulation_config_inputs'][idx_sim_key] = state['simulation_config_inputs'][idx_sim_key][:-1]
                elif event.key == pygame.K_TAB:
                    state['simulation_config_active_input'] = (idx_sim_key + 1) % len(state['simulation_config_inputs'])
                elif event.key == pygame.K_ESCAPE: # Allow Esc to cancel simulation config
                    state['showing_simulation_config'] = False; state['simulation_config_active_input'] = None
                elif event.unicode.isdigit() and len(state['simulation_config_inputs'][idx_sim_key]) < 3: # Limit input length
                    state['simulation_config_inputs'][idx_sim_key] += event.unicode
            elif state.get('typing'):
                current_player_idx_key = state['turn'] - 1
                is_human_turn_or_paused_key = (0 <= current_player_idx_key < len(state['is_ai'])) and \
                                          (not state['is_ai'][current_player_idx_key] or state.get('paused_for_power_tile') or state.get('paused_for_bingo_practice'))
                if not is_human_turn_or_paused_key: continue # Ignore typing if not human's turn / not paused AI

                mods_key = pygame.key.get_mods()
                if event.key == pygame.K_v and (mods_key & pygame.KMOD_META): # CMD+V or KMOD_CTRL for Ctrl+V
                    if pyperclip_available and pyperclip:
                        try:
                            pasted_text_key = pyperclip.paste()
                            if pasted_text_key:
                                pasted_text_key = pasted_text_key.strip().upper()
                                # Ensure current_r, current_c, typing_direction are valid before pasting
                                local_current_r_key, local_current_c_key, local_typing_direction_key = state.get('current_r'), state.get('current_c'), state.get('typing_direction')
                                if state.get('typing_start'): # If typing started, ensure direction is from there
                                    local_current_r_key, local_current_c_key = state['typing_start']; local_typing_direction_key = state['typing_direction']

                                if local_current_r_key is None or local_current_c_key is None or local_typing_direction_key is None:
                                    print("  Error: Cannot paste, current cursor state (r,c,direction) is invalid."); pasted_text_key = ""
                                
                                for char_paste in pasted_text_key:
                                    if not ('A' <= char_paste <= 'Z'): continue # Paste only letters
                                    use_blank_key_paste = False
                                    if char_paste not in state['racks'][state['turn']-1]:
                                        if ' ' in state['racks'][state['turn']-1]: use_blank_key_paste = True
                                        else: continue # Cannot play this char

                                    if local_typing_direction_key == "right" and local_current_c_key < GRID_SIZE:
                                        if not state['tiles'][local_current_r_key][local_current_c_key]: # If square is empty
                                            state['tiles'][local_current_r_key][local_current_c_key] = char_paste
                                            state['word_positions'].append((local_current_r_key, local_current_c_key, char_paste))
                                            if use_blank_key_paste: state['blanks'].add((local_current_r_key, local_current_c_key)); state['racks'][state['turn']-1].remove(' ')
                                            else: state['racks'][state['turn']-1].remove(char_paste)
                                            local_current_c_key += 1
                                    elif local_typing_direction_key == "down" and local_current_r_key < GRID_SIZE:
                                        if not state['tiles'][local_current_r_key][local_current_c_key]:
                                            state['tiles'][local_current_r_key][local_current_c_key] = char_paste
                                            state['word_positions'].append((local_current_r_key, local_current_c_key, char_paste))
                                            if use_blank_key_paste: state['blanks'].add((local_current_r_key, local_current_c_key)); state['racks'][state['turn']-1].remove(' ')
                                            else: state['racks'][state['turn']-1].remove(char_paste)
                                            local_current_r_key += 1
                                    # Update state's main current_r, current_c if they were modified
                                    state['current_r'], state['current_c'] = local_current_r_key, local_current_c_key
                        except pyperclip.PyperclipException as e_pyperclip:
                            print(f"Paste error: {e_pyperclip}")
                            show_message_dialog("Could not paste from clipboard.", "Paste Error")

                elif event.key == pygame.K_BACKSPACE:
                    if state['word_positions']:
                        last_r_key, last_c_key, last_letter_key = state['word_positions'].pop()
                        state['tiles'][last_r_key][last_c_key] = ''
                        tile_to_return_key = ' ' if (last_r_key, last_c_key) in state['blanks'] else last_letter_key
                        state['racks'][state['turn']-1].append(tile_to_return_key)
                        if (last_r_key, last_c_key) in state['blanks']: state['blanks'].remove((last_r_key, last_c_key))
                        state['current_r'], state['current_c'] = last_r_key, last_c_key # Move cursor back
                        if not state['word_positions']: # If last typed letter removed
                            state['typing'] = False; state['original_tiles'] = None; state['original_rack'] = None; state['selected_square'] = None; state['typing_start'] = None; state['typing_direction'] = None; current_r = None; current_c = None; state['current_r'] = None; state['current_c'] = None; state['typing_direction'] = None
                elif event.key == pygame.K_RETURN:
                    print("\n--- DEBUG: Finalizing Typed Play ---")
                    if state['word_positions']:
                        newly_placed_details = [(r_wp, c_wp, l_wp) for r_wp, c_wp, l_wp in state['word_positions']]
                        print(f"  Newly Placed: {newly_placed_details}")
                        initial_rack_size_for_play = len(state.get('original_rack', []))
                        print(f"  First Play? {state['first_play']}")
                        # Use DAWG directly for validation
                        print(f"  DEBUG: Calling is_valid_play_cython with state['first_play'] = {state['first_play']}")
                        is_valid, is_bingo = is_valid_play_cython(newly_placed_details, state['tiles'], state['first_play'], initial_rack_size_for_play, state.get('original_tiles'), state['racks'][state['turn']-1], DAWG)
                        print(f"  is_valid_play_cython returned: is_valid={is_valid}, is_bingo={is_bingo}")

                        if is_valid:
                            score = calculate_score_cython(newly_placed_details, state['board'], state['tiles'], state['blanks'])
                            proceed_with_finalization = True
                            # Practice Mode Validation
                            if state['practice_mode'] == "only_fives":
                                if not does_move_form_five_letter_word( {'newly_placed': newly_placed_details} , state['tiles'], state['blanks']): # Pass a move-like dict
                                    show_message_dialog("At least one 5-letter word must be formed.", "Invalid Play")
                                    proceed_with_finalization = False
                            elif state['practice_mode'] == "eight_letter" and state.get('practice_best_move'):
                                # Simple check: if any tile of newly_placed is part of practice_best_move newly_placed
                                # This is a loose check, assumes player is trying to make the target word.
                                best_move_newly_placed_coords = set((bm[0], bm[1]) for bm in state['practice_best_move'].get('newly_placed', []))
                                current_play_coords = set((npd[0], npd[1]) for npd in newly_placed_details)
                                if not best_move_newly_placed_coords.issubset(current_play_coords): # Or some other check
                                    show_message_dialog("This is not the target 8-letter bingo solution.", "8-Letter Bingo")
                                    proceed_with_finalization = False
                                elif score < state['practice_best_move'].get('score', float('inf')):
                                     show_message_dialog(f"Try again. The highest score is {state['practice_best_move'].get('score',0)}.", "8-Letter Bingo")
                                     proceed_with_finalization = False
                                else: # Correct play for 8-letter
                                    state['practice_solved'] = True; state['showing_practice_end_dialog'] = True;
                                    state['practice_end_message'] = f"Correct! You found the target bingo:\n{state['practice_best_move'].get('word_with_blanks','')} ({score} pts)"
                            elif state['practice_mode'] == "power_tiles" and state.get('paused_for_power_tile'):
                                # Check if move uses current_power_tile and meets length criteria
                                uses_power = any(letter == state['current_power_tile'] for _, _, letter in newly_placed_details)
                                temp_move_dict = {'word': "".join(l for _,_,l in newly_placed_details), 'newly_placed':newly_placed_details, 'score':score} # mock for check
                                word_len_check = is_word_length_allowed(len(temp_move_dict['word']), state['number_checks'])
                                if not (uses_power and word_len_check):
                                    show_message_dialog(f"Move must use {state['current_power_tile']} and match length criteria.", "Invalid Play")
                                    proceed_with_finalization = False
                                # No "max score" check here, just validity for the power tile.
                            elif state['practice_mode'] == "bingo_bango_bongo" and state.get('paused_for_bingo_practice'):
                                if not is_bingo: # Must be a bingo
                                    show_message_dialog("This is not a bingo!", "Bingo, Bango, Bongo")
                                    proceed_with_finalization = False
                                # No "max score" check, just that it IS a bingo.

                            if proceed_with_finalization:
                                blanks_just_played_this_move = 0
                                tiles_placed_from_rack_sgs = []
                                blanks_played_info_sgs = []
                                temp_rack_for_sgs_logging = Counter(state.get('original_rack', []))
                                
                                for r_placed, c_placed, letter_placed in newly_placed_details:
                                    is_this_placement_a_blank = (r_placed, c_placed) in state['blanks']
                                    if is_this_placement_a_blank:
                                        blanks_just_played_this_move += 1
                                        blanks_played_info_sgs.append({'coord': (r_placed, c_placed), 'assigned_letter': letter_placed})
                                    
                                    tile_to_log_removed = ' ' if is_this_placement_a_blank else letter_placed
                                    if temp_rack_for_sgs_logging[tile_to_log_removed] > 0:
                                        tiles_placed_from_rack_sgs.append(tile_to_log_removed)
                                        temp_rack_for_sgs_logging[tile_to_log_removed] -=1

                                blanks_played_count += blanks_just_played_this_move # Update game total
                                
                                # Determine primary word, start_pos, direction for SGS
                                # This is complex if multiple words; GADDAG usually returns this.
                                # For typed play, we derive from newly_placed_details & typing_direction.
                                all_words_formed_details = find_all_words_formed_cython(newly_placed_details, state['tiles'])
                                primary_word_tiles = []; primary_word_str = ""
                                start_pos = state.get('typing_start') # This is where typing started
                                orientation_str_from_typing = state.get('typing_direction') # 'right' or 'down'
                                orientation_for_gaddag = '?'
                                if orientation_str_from_typing == 'right': orientation_for_gaddag = 'H'
                                elif orientation_str_from_typing == 'down': orientation_for_gaddag = 'V'

                                word_with_blanks = ""
                                newly_placed_coords_set = set((r_npc,c_npc) for r_npc,c_npc,_ in newly_placed_details)
                                current_move_blanks_coords = set((r_cmb,c_cmb) for r_cmb,c_cmb in newly_placed_coords_set if (r_cmb,c_cmb) in state['blanks'])

                                if all_words_formed_details:
                                    found_primary = False
                                    # Try to find the word along the typing axis containing a newly placed tile
                                    for word_detail in all_words_formed_details:
                                        if not any((t[0],t[1]) in newly_placed_coords_set for t in word_detail): continue # Must use a new tile
                                        is_along_axis = False
                                        current_word_rows = set(r_wd for r_wd,c_wd,l_wd in word_detail)
                                        current_word_cols = set(c_wd for r_wd,c_wd,l_wd in word_detail)
                                        if orientation_for_gaddag == 'H' and len(current_word_rows) == 1: is_along_axis = True
                                        elif orientation_for_gaddag == 'V' and len(current_word_cols) == 1: is_along_axis = True
                                        # If not along typing axis but still a main word (e.g. single letter play forming two words)
                                        if not is_along_axis and len(newly_placed_details) == 1:
                                            if len(current_word_rows) == 1: is_along_axis = True; # Treat as horizontal
                                            elif len(current_word_cols) == 1: is_along_axis = True; # Treat as vertical

                                        if is_along_axis:
                                            primary_word_tiles = word_detail; found_primary = True; break
                                    if not found_primary: # Fallback: longest word using a new tile
                                        longest_len = 0
                                        for word_detail in all_words_formed_details:
                                            if any((t[0],t[1]) in newly_placed_coords_set for t in word_detail):
                                                if len(word_detail) > longest_len:
                                                    longest_len = len(word_detail); primary_word_tiles = word_detail
                                        if primary_word_tiles: # Determine its orientation
                                            if len(set(r_wd for r_wd,c_wd,l_wd in primary_word_tiles)) == 1: orientation_for_gaddag = 'H'
                                            elif len(set(c_wd for r_wd,c_wd,l_wd in primary_word_tiles)) == 1: orientation_for_gaddag = 'V'
                                    if not primary_word_tiles and all_words_formed_details: # Further fallback
                                        primary_word_tiles = all_words_formed_details[0] # Just take the first one
                                        if len(set(r_wd for r_wd,c_wd,l_wd in primary_word_tiles)) == 1: orientation_for_gaddag = 'H'
                                        elif len(set(c_wd for r_wd,c_wd,l_wd in primary_word_tiles)) == 1: orientation_for_gaddag = 'V'

                                orientation_str_final = 'right' # Default
                                if primary_word_tiles:
                                    primary_word_str = "".join(t[2] for t in primary_word_tiles)
                                    if orientation_for_gaddag == 'H':
                                        start_pos = min(primary_word_tiles, key=lambda x_coord: x_coord[1])[:2]
                                        orientation_str_final = 'right'
                                    elif orientation_for_gaddag == 'V':
                                        start_pos = min(primary_word_tiles, key=lambda x_coord: x_coord[0])[:2]
                                        orientation_str_final = 'down'
                                    else: # Fallback if orientation couldn't be determined from primary word
                                        start_pos = primary_word_tiles[0][:2]
                                        orientation_str_final = 'right' if len(set(r_pwt for r_pwt,c_pwt,l_pwt in primary_word_tiles)) == 1 else 'down'
                                    
                                    word_with_blanks_list_temp = []
                                    for wr_pwt, wc_pwt, w_letter_pwt in primary_word_tiles:
                                        is_blank_in_word_pwt = (wr_pwt, wc_pwt) in newly_placed_coords_set and (wr_pwt, wc_pwt) in current_move_blanks_coords
                                        word_with_blanks_list_temp.append(w_letter_pwt.lower() if is_blank_in_word_pwt else w_letter_pwt.upper())
                                    word_with_blanks = "".join(word_with_blanks_list_temp)
                                elif newly_placed_details: # Fallback to just the typed letters if no primary word found (e.g. disconnected play, though invalid)
                                    primary_word_str = "".join(l_npd for r_npd,c_npd,l_npd in newly_placed_details)
                                    word_with_blanks = primary_word_str # Crude, assumes no blanks if primary_word_tiles failed
                                    start_pos = newly_placed_details[0][:2]
                                    rows_npd = set(r_npd for r_npd,c_npd,l_npd in newly_placed_details)
                                    cols_npd = set(c_npd for r_npd,c_npd,l_npd in newly_placed_details)
                                    if len(rows_npd) == 1: orientation_str_final = 'right'
                                    elif len(cols_npd) == 1: orientation_str_final = 'down'
                                else: # Should not happen if newly_placed_details exists
                                    primary_word_str = ""; word_with_blanks = ""; start_pos = (0,0)

                                state['scores'][state['turn']-1] += score
                                if is_bingo: state['scores'][state['turn']-1] += 50
                                num_to_draw = len(newly_placed_details)
                                drawn_tiles = [state['bag'].pop(0) for _ in range(num_to_draw) if state['bag']]
                                state['racks'][state['turn']-1].extend(drawn_tiles)
                                state['last_played_highlight_coords'] = newly_placed_coords_set
                                
                                luck_factor = 0.0
                                if drawn_tiles:
                                    try:
                                        tiles_grid_before_play = state.get('original_tiles')
                                        blanks_set_before_play = state.get('original_blanks_set_at_typing_start')
                                        if tiles_grid_before_play is None: tiles_grid_before_play = state['tiles'] # Fallback
                                        if blanks_set_before_play is None: blanks_set_before_play = state['blanks'].difference(current_move_blanks_coords)
                                        
                                        blanks_total_before_this_move = blanks_played_count - blanks_just_played_this_move

                                        luck_factor = calculate_luck_factor_cython(
                                            drawn_tiles, state.get('original_rack',[]),
                                            tiles_grid_before_play, blanks_set_before_play,
                                            blanks_total_before_this_move, get_remaining_tiles
                                        )
                                        if not state.get('is_batch_running'):
                                            drawn_tiles_str = "".join(sorted(t if t != ' ' else '?' for t in drawn_tiles))
                                            print(f"  Drew: {drawn_tiles_str}, Luck (Cython): {luck_factor:+.2f}")
                                    except Exception as e_luck_human:
                                        print(f"Error in human play calling calculate_luck_factor_cython: {e_luck_human}")
                                        luck_factor = 0.0

                                final_start_pos = start_pos if start_pos is not None else (0,0)
                                final_orientation = orientation_str_final if orientation_str_final is not None else 'right'

                                move_data_sgs = {
                                    'player': state['turn'], 'move_type': 'place', 'score': score + (50 if is_bingo else 0),
                                    'word': primary_word_str, 'positions': primary_word_tiles, # Store the main word tiles
                                    'blanks_coords_on_board_this_play': list(current_move_blanks_coords),
                                    'coord': get_coord(final_start_pos, final_orientation),
                                    'leave': state['racks'][state['turn']-1][:], 'is_bingo': is_bingo,
                                    'turn_duration': 0.0, # Placeholder for human, can be calculated
                                    'word_with_blanks': word_with_blanks,
                                    'newly_placed_details': newly_placed_details,
                                    'tiles_placed_from_rack': tiles_placed_from_rack_sgs,
                                    'blanks_played_info': blanks_played_info_sgs,
                                    'rack_before_move': state.get('original_rack',[]),
                                    'tiles_drawn_after_move': drawn_tiles, 'exchanged_tiles':[],
                                    'luck_factor': luck_factor,
                                    'next_turn_after_move': 3 - state['turn'],
                                    'pass_count_after_move': 0, 'exchange_count_after_move': 0,
                                    'consecutive_zero_after_move': 0,
                                    'is_first_play_after_move': False,
                                    'board_tile_counts_after_move': state['board_tile_counts'].copy(), # This should be updated by play_hint_move or equivalent
                                    'blanks_played_count_after_move': blanks_played_count # This is the updated game total
                                }
                                # Update board_tile_counts for the played tiles
                                for r_p,c_p,l_p in newly_placed_details:
                                    state['board_tile_counts'][l_p] +=1
                                move_data_sgs['board_tile_counts_after_move'] = state['board_tile_counts'].copy()

                                state['move_history'].append(move_data_sgs)
                                state['turn'] = 3 - state['turn']
                                state['first_play'] = False; state['pass_count'] = 0; state['exchange_count'] = 0; state['consecutive_zero_point_turns'] = 0
                                state['human_played'] = True # Signal that human made a move
                                state['paused_for_power_tile'] = False; state['paused_for_bingo_practice'] = False # Reset practice pauses
                            else: # proceed_with_finalization is False (e.g. practice mode invalid play)
                                # Revert tiles on board and rack from word_positions
                                if state.get('original_tiles') and state.get('original_rack'):
                                    state['tiles'] = [r[:] for r in state['original_tiles']]
                                    state['racks'][state['turn']-1] = state['original_rack'][:]
                                    blanks_to_remove_revert = set((r_br, c_br) for r_br, c_br, _ in state['word_positions'] if (r_br, c_br) in state['blanks'])
                                    state['blanks'].difference_update(blanks_to_remove_revert)
                        else: # Play was invalid
                            show_message_dialog("Invalid play.", "Error")
                            if state.get('original_tiles') and state.get('original_rack'): # Revert if original state was stored
                                state['tiles'] = [r[:] for r in state['original_tiles']]
                                state['racks'][state['turn']-1] = state['original_rack'][:]
                                blanks_to_remove_invalid = set((r_bi, c_bi) for r_bi, c_bi, _ in state['word_positions'] if (r_bi, c_bi) in state['blanks'])
                                state['blanks'].difference_update(blanks_to_remove_invalid)
                        # Reset typing state
                        state['typing'] = False; state['word_positions'] = []; state['original_tiles'] = None; state['original_rack'] = None; state['original_blanks_set_at_typing_start'] = None; state['selected_square'] = None; current_r = None; current_c = None; typing_direction = None; typing_start = None; state['current_r'] = None; state['current_c'] = None; state['typing_direction'] = None; state['typing_start'] = None;
                    else: # Enter pressed but no letters typed
                        state['typing'] = False; state['selected_square'] = None; current_r = None; current_c = None; state['current_r'] = None; state['current_c'] = None; state['typing_direction'] = None; state['typing_start'] = None;
                elif event.key == pygame.K_ESCAPE:
                    if state.get('typing'): # If typing, cancel typing
                        if state.get('original_tiles') and state.get('original_rack'):
                            state['tiles'] = [r[:] for r in state['original_tiles']]
                            state['racks'][state['turn']-1] = state['original_rack'][:] # Restore rack
                            blanks_to_remove_esc = set((r_be, c_be) for r_be, c_be, _ in state.get('word_positions',[]) if (r_be, c_be) in state.get('blanks',set())); state.get('blanks',set()).difference_update(blanks_to_remove_esc)
                        state['typing'] = False; state['word_positions'] = []; state['original_tiles'] = None; state['original_rack'] = None; state['original_blanks_set_at_typing_start'] = None; state['selected_square'] = None; current_r = None; current_c = None; typing_direction = None; typing_start = None; state['current_r'] = None; state['current_c'] = None; state['typing_direction'] = None; state['typing_start'] = None;
                    elif state.get('selected_square'): # If a square is selected but not typing, deselect it
                        state['selected_square'] = None; current_r = None; current_c = None; state['current_r'] = None; state['current_c'] = None;
                    elif state.get('hinting'): state['hinting'] = False
                    elif state.get('showing_all_words'): state['showing_all_words'] = False
                    elif state.get('exchanging'): state['exchanging'] = False; state['selected_tiles'].clear()
                    elif state.get('specifying_rack'): state['specifying_rack'] = False; state['specifying_rack_active_input'] = None
                    elif state.get('showing_simulation_config'): state['showing_simulation_config'] = False; state['simulation_config_active_input'] = None
                    elif state.get('showing_stats'): state['showing_stats'] = False
                    elif state.get('dropdown_open'): state['dropdown_open'] = False
                    # No general quit on ESC here, handled by specific dialogs or MOUSEBUTTONDOWN on Quit option
                elif event.unicode.isalpha() and state.get('typing_direction') is not None and \
                     state.get('current_r') is not None and state.get('current_c') is not None:
                    letter_key_type = event.unicode.upper()
                    current_rack_debug_key = state['racks'][state['turn']-1]; has_letter_key = letter_key_type in current_rack_debug_key; has_blank_key = ' ' in current_rack_debug_key

                    if has_letter_key or has_blank_key:
                        r_type, c_type, dir_type = state['current_r'], state['current_c'], state['typing_direction']
                        if state.get('typing_start') is None: # Should be set if typing_direction is not None
                            state['typing_start'] = (r_type, c_type)

                        if not state['tiles'][r_type][c_type]: # If square is empty
                            use_blank_key_type = False
                            if not has_letter_key and has_blank_key: use_blank_key_type = True

                            state['tiles'][r_type][c_type] = letter_key_type
                            state['word_positions'].append((r_type, c_type, letter_key_type))
                            if use_blank_key_type: state['blanks'].add((r_type, c_type)); state['racks'][state['turn']-1].remove(' ')
                            else: state['racks'][state['turn']-1].remove(letter_key_type)

                            if dir_type == "right": state['current_c'] += 1
                            else: state['current_r'] += 1
                            if state['current_c'] >= GRID_SIZE or state['current_r'] >= GRID_SIZE : # Reached edge
                                state['typing'] = False # End typing, user must press Enter or Esc
                    else:
                        print(f"Cannot play '{letter_key_type}': Not in rack {''.join(sorted(current_rack_debug_key))}")
            # Handle other KEYDOWN events (dialogs, global shortcuts)
            elif state.get('game_over_state'): # Game over dialog key shortcuts
                if event.key == pygame.K_s: # Save
                    save_rect_key = drawn_rects.get('save_rect')
                    if save_rect_key: # Simulate click
                        mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': save_rect_key.center})
                        state.update(handle_mouse_down_event(mock_event, state, drawn_rects))
                elif event.key == pygame.K_q: # Quit
                    quit_rect_key = drawn_rects.get('quit_rect')
                    if quit_rect_key:
                         mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': quit_rect_key.center})
                         state.update(handle_mouse_down_event(mock_event, state, drawn_rects))
                         running_inner = state['running_inner'] # Update from result
                elif event.key == pygame.K_r: # Replay
                    replay_rect_key = drawn_rects.get('replay_rect')
                    if replay_rect_key:
                         mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': replay_rect_key.center})
                         state.update(handle_mouse_down_event(mock_event, state, drawn_rects))
                elif event.key == pygame.K_p: # Play Again
                    play_again_rect_key = drawn_rects.get('play_again_rect')
                    if play_again_rect_key:
                        mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': play_again_rect_key.center})
                        state.update(handle_mouse_down_event(mock_event, state, drawn_rects))
                        running_inner = state['running_inner']; return_to_mode_selection = state['return_to_mode_selection']
            elif state.get('showing_stats') and event.key == pygame.K_RETURN: # OK for stats dialog
                stats_ok_button_rect_key = drawn_rects.get('stats_ok_button_rect')
                if stats_ok_button_rect_key:
                    mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': stats_ok_button_rect_key.center})
                    state.update(handle_mouse_down_event(mock_event, state, drawn_rects))
            elif state.get('hinting') and event.key == pygame.K_RETURN: # Play/Exchange for hint dialog
                play_button_rect_key = drawn_rects.get('play_button_rect') # This is "Play/Exchange"
                if play_button_rect_key:
                    mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': play_button_rect_key.center})
                    state.update(handle_mouse_down_event(mock_event, state, drawn_rects))
            elif state.get('showing_all_words') and event.key == pygame.K_RETURN: # Play for all_words
                all_words_play_rect_key = drawn_rects.get('all_words_play_rect')
                if all_words_play_rect_key:
                    mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': all_words_play_rect_key.center})
                    state.update(handle_mouse_down_event(mock_event, state, drawn_rects))
            elif state.get('exchanging') and event.key == pygame.K_RETURN: # Exchange for exchange_dialog
                exchange_button_rect_key = drawn_rects.get('exchange_button_rect')
                if exchange_button_rect_key:
                    mock_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'button': 1, 'pos': exchange_button_rect_key.center})
                    state.update(handle_mouse_down_event(mock_event, state, drawn_rects))

    # Update local control flow flags from state if they were modified by handle_mouse_down_event
    running_inner = state['running_inner']
    return_to_mode_selection = state['return_to_mode_selection']
    # Update other state variables that might have been changed by event handlers
    # This pattern of state.update() in MOUSEBUTTONDOWN and then re-assigning locals
    # helps keep the main loop variables synchronized with the state dict.
    current_r = state.get('current_r'); current_c = state.get('current_c')
    typing_direction = state.get('typing_direction'); typing_start = state.get('typing_start')
    blanks_played_count = state.get('blanks_played_count', 0) # Update from state

    # Update state dictionary that will be returned (or used in next iteration)
    # This is somewhat redundant if state.update was used effectively, but ensures all local vars are packed back if modified
    state['running_inner'] = running_inner
    state['return_to_mode_selection'] = return_to_mode_selection
    state['current_r'] = current_r; state['current_c'] = current_c
    state['typing_direction'] = typing_direction; state['typing_start'] = typing_start
    state['blanks_played_count'] = blanks_played_count
    # All other state variables modified directly in the `state` dict are already there.

    return state # Return the modified state dictionary










    

def handle_mouse_down_event(event, state, drawn_rects): # Added drawn_rects parameter
    # --- Access global directly for GADDAG status check ---
    global gaddag_loading_status, gaddag_loaded_event, GADDAG_STRUCTURE
    # --- Access global DAWG object ---
    global DAWG

    x, y = event.pos

    # Unpack necessary state variables from the dictionary
    turn = state['turn']; dropdown_open = state['dropdown_open']; 
    is_batch_running = state['is_batch_running']; replay_mode = state['replay_mode']; game_over_state = state['game_over_state']; practice_mode = state['practice_mode']; exchanging = state['exchanging']; hinting = state['hinting']; showing_all_words = state['showing_all_words']; specifying_rack = state['specifying_rack']; showing_simulation_config = state['showing_simulation_config']; showing_practice_end_dialog = state['showing_practice_end_dialog']; confirming_override = state['confirming_override']; final_scores = state['final_scores']; player_names = state['player_names']; move_history = state['move_history']; initial_racks = state.get('initial_racks', None); showing_stats = state['showing_stats']; stats_dialog_x = state['stats_dialog_x']; stats_dialog_y = state['stats_dialog_y']; dialog_x = state['dialog_x']; dialog_y = state['dialog_y']; current_replay_turn = state['current_replay_turn']; selected_tiles = state['selected_tiles']; is_ai = state['is_ai']; specify_rack_original_racks = state['specify_rack_original_racks']; specify_rack_inputs = state['specify_rack_inputs']; specify_rack_active_input = state['specify_rack_active_input']; specify_rack_proposed_racks = state['specify_rack_proposed_racks']; racks = state['racks'];
    scores = state['scores']; paused_for_power_tile = state['paused_for_power_tile']; paused_for_bingo_practice = state['paused_for_bingo_practice']; practice_best_move = state['practice_best_move']; all_moves = state.get('all_moves', []); current_power_tile = state['current_power_tile']; number_checks = state['number_checks'];
    first_play = state['first_play']; pass_count = state['pass_count']; exchange_count = state['exchange_count']; consecutive_zero_point_turns = state['consecutive_zero_point_turns']; last_played_highlight_coords = state.get('last_played_highlight_coords', set());
    practice_solved = state['practice_solved'];
    practice_end_message = state['practice_end_message']; simulation_config_inputs = state['simulation_config_inputs']; simulation_config_active_input = state['simulation_config_active_input']; hint_moves = state.get('hint_moves', []); selected_hint_index = state.get('selected_hint_index'); preview_score_enabled = state['preview_score_enabled']; dragged_tile = state['dragged_tile']; drag_pos = state['drag_pos']; drag_offset = state['drag_offset']; typing = state['typing']; word_positions = state['word_positions']; original_tiles = state['original_tiles']; original_rack = state['original_rack']; original_blanks_set_at_typing_start = state.get('original_blanks_set_at_typing_start'); selected_square = state['selected_square']; last_left_click_time = state['last_left_click_time']; last_left_click_pos = state['last_left_click_pos']; stats_dialog_dragging = state['stats_dialog_dragging']; dragging = state['dragging']; letter_checks = state['letter_checks']
    stats_scroll_offset = state['stats_scroll_offset']
    stats_dialog_drag_offset = state['stats_dialog_drag_offset']
    all_words_scroll_offset = state['all_words_scroll_offset']
    restart_practice_mode = state['restart_practice_mode']
    stats_total_content_height = state.get('stats_total_content_height', 0)
    board_tile_counts = state['board_tile_counts']
    blanks_played_count = state.get('blanks_played_count', 0) 
    practice_target_moves = state.get('practice_target_moves', [])
    current_r = state.get('current_r'); current_c = state.get('current_c'); typing_direction = state.get('typing_direction'); typing_start = state.get('typing_start')
    best_exchange_for_hint = state.get('best_exchange_for_hint'); best_exchange_score_for_hint = state.get('best_exchange_score_for_hint', -float('inf'))
    
    tiles = state['tiles'] 
    board = state['board'] 
    blanks = state['blanks'] 
    bag = state['bag']     
    bag_count = len(bag) 

    current_gaddag_loading_status_in_state = state.get('gaddag_loading_status', 'idle')
    current_move_blanks_coords = state.get('current_move_blanks_coords', set())
    pyperclip_available = state.get('pyperclip_available', False)
    pyperclip = state.get('pyperclip', None)

    practice_play_again_rect = drawn_rects.get('practice_play_again_rect'); practice_main_menu_rect = drawn_rects.get('practice_main_menu_rect'); practice_quit_rect = drawn_rects.get('practice_quit_rect')
    sim_input_rects = drawn_rects.get('sim_input_rects', []); sim_simulate_rect = drawn_rects.get('sim_simulate_rect'); sim_cancel_rect = drawn_rects.get('sim_cancel_rect')
    go_back_rect_ov = drawn_rects.get('go_back_rect_ov'); override_rect_ov = drawn_rects.get('override_rect_ov')
    p1_input_rect_sr = drawn_rects.get('p1_input_rect_sr'); p2_input_rect_sr = drawn_rects.get('p2_input_rect_sr'); p1_reset_rect_sr = drawn_rects.get('p1_reset_rect_sr'); p2_reset_rect_sr = drawn_rects.get('p2_reset_rect_sr'); confirm_rect_sr = drawn_rects.get('confirm_rect_sr'); cancel_rect_sr = drawn_rects.get('cancel_rect_sr')
    options_rect_base = drawn_rects.get('options_rect_base'); dropdown_rects_base = drawn_rects.get('dropdown_rects_base', [])
    save_rect = drawn_rects.get('save_rect'); quit_rect = drawn_rects.get('quit_rect'); replay_rect = drawn_rects.get('replay_rect'); play_again_rect = drawn_rects.get('play_again_rect'); stats_rect = drawn_rects.get('stats_rect'); stats_ok_button_rect = drawn_rects.get('stats_ok_button_rect')
    replay_start_rect = state.get('replay_start_rect'); replay_prev_rect = state.get('replay_prev_rect'); replay_next_rect = state.get('replay_next_rect'); replay_end_rect = state.get('replay_end_rect')
    suggest_rect_base = drawn_rects.get('suggest_rect_base'); simulate_button_rect = drawn_rects.get('simulate_button_rect'); preview_checkbox_rect = drawn_rects.get('preview_checkbox_rect')
    p1_alpha_rect = drawn_rects.get('p1_alpha_rect'); p1_rand_rect = drawn_rects.get('p1_rand_rect'); p2_alpha_rect = drawn_rects.get('p2_alpha_rect'); p2_rand_rect = drawn_rects.get('p2_rand_rect')
    tile_rects_exch_dlg = drawn_rects.get('tile_rects', [])
    exchange_button_exch_dlg = drawn_rects.get('exchange_button_rect')
    cancel_button_exch_dlg = drawn_rects.get('cancel_button_rect')
    hint_rects_list_hint_dlg = drawn_rects.get('hint_rects', [])
    play_exch_button_hint_dlg = drawn_rects.get('play_button_rect')
    all_words_button_hint_dlg = drawn_rects.get('all_words_button_rect')
    ok_button_hint_dlg = drawn_rects.get('ok_button_rect')
    all_words_rects_list_aw_dlg = drawn_rects.get('all_words_rects', [])
    all_words_play_rect_aw_dlg = drawn_rects.get('all_words_play_rect')
    all_words_ok_rect_aw_dlg = drawn_rects.get('all_words_ok_rect')
    
    # This print was for initial check of drawn_rects content, can be removed or kept for general debug
    # print(f"DEBUG HMD_EVENT: all_words_button_hint_dlg = {all_words_button_hint_dlg}, play_exch_button_hint_dlg = {play_exch_button_hint_dlg}, ok_button_hint_dlg = {ok_button_hint_dlg}")

    running_inner = True; return_to_mode_selection = False; batch_stop_requested = False; human_played = False

    if showing_practice_end_dialog:
        if practice_play_again_rect and practice_play_again_rect.collidepoint(x,y):
            restart_practice_mode = True
            showing_practice_end_dialog = False
        elif practice_main_menu_rect and practice_main_menu_rect.collidepoint(x,y):
            running_inner = False
            return_to_mode_selection = True
            showing_practice_end_dialog = False
        elif practice_quit_rect and practice_quit_rect.collidepoint(x,y):
            running_inner = False
            showing_practice_end_dialog = False
        state.update({
            'running_inner': running_inner,
            'return_to_mode_selection': return_to_mode_selection,
            'restart_practice_mode': restart_practice_mode,
            'showing_practice_end_dialog': showing_practice_end_dialog
        })
        return state

    if showing_simulation_config:
        clicked_input_sim_cfg = False
        for i_sim_cfg, rect_sim_cfg in enumerate(sim_input_rects):
            if rect_sim_cfg.collidepoint(x, y):
                simulation_config_active_input = i_sim_cfg
                clicked_input_sim_cfg = True
                break
        if not clicked_input_sim_cfg and not (sim_simulate_rect and sim_simulate_rect.collidepoint(x,y)) and not (sim_cancel_rect and sim_cancel_rect.collidepoint(x,y)):
            simulation_config_active_input = None

        if sim_cancel_rect and sim_cancel_rect.collidepoint(x, y):
            showing_simulation_config = False
            simulation_config_active_input = None
        elif sim_simulate_rect and sim_simulate_rect.collidepoint(x, y):
            try:
                num_ai_cand_cfg_val = int(simulation_config_inputs[0])
                num_opp_sim_cfg_val = int(simulation_config_inputs[1])
                num_post_sim_cfg_val = int(simulation_config_inputs[2])
                player_idx_cfg_val = turn - 1
                opponent_idx_cfg_val = 1 - player_idx_cfg_val
                opponent_rack_len_cfg_val = len(racks[opponent_idx_cfg_val]) if opponent_idx_cfg_val < len(racks) else 7

                if gaddag_loading_status != 'loaded' or GADDAG_STRUCTURE is None or DAWG is None:
                    show_message_dialog("AI data not loaded. Cannot run simulation.", "Error")
                else:
                    simulation_results_cfg_val = run_ai_simulation(
                        racks[player_idx_cfg_val], tiles, blanks, board, bag,
                        num_ai_cand_cfg_val, num_opp_sim_cfg_val, num_post_sim_cfg_val,
                        opponent_rack_len_cfg_val,
                        board_tile_counts,
                        blanks_played_count
                    )
                    hint_moves = simulation_results_cfg_val
                    current_player_rack_sim_cfg_local_val = racks[player_idx_cfg_val]
                    if len(bag) > 0:
                        best_exchange_for_hint, best_exchange_score_for_hint = find_best_exchange_option(
                            current_player_rack_sim_cfg_local_val, tiles, blanks, blanks_played_count, len(bag)
                        )
                    else:
                        best_exchange_for_hint = None
                        best_exchange_score_for_hint = -float('inf')
                    hinting = True
                    selected_hint_index = 0 if hint_moves or best_exchange_for_hint else None
                showing_simulation_config = False
                simulation_config_active_input = None
            except ValueError:
                show_message_dialog("Invalid input. Please enter numbers.", "Error")
            except Exception as e:
                show_message_dialog(f"Error during simulation: {e}", "Error")
                showing_simulation_config = False
                simulation_config_active_input = None
        state.update({
            'showing_simulation_config': showing_simulation_config,
            'simulation_config_inputs': simulation_config_inputs,
            'simulation_config_active_input': simulation_config_active_input,
            'hint_moves': hint_moves,
            'hinting': hinting,
            'selected_hint_index': selected_hint_index,
            'best_exchange_for_hint': best_exchange_for_hint,
            'best_exchange_score_for_hint': best_exchange_score_for_hint
        })
        return state

    if confirming_override:
        if go_back_rect_ov and go_back_rect_ov.collidepoint(x,y):
            confirming_override = False
            specifying_rack = True
        elif override_rect_ov and override_rect_ov.collidepoint(x,y):
            confirming_override = False
            specifying_rack = False
            racks[0] = list(specify_rack_proposed_racks[0])
            racks[1] = list(specify_rack_proposed_racks[1])
            show_message_dialog("Racks have been overridden.", "Racks Set")
            typing = False; word_positions = []; original_tiles = None; original_rack = None; original_blanks_set_at_typing_start = None; selected_square = None; current_r = None; current_c = None; typing_direction = None; typing_start = None; current_move_blanks_coords = set()
        state.update({
            'confirming_override': confirming_override,
            'specifying_rack': specifying_rack,
            'racks': racks,
            'typing': typing, 'word_positions': word_positions, 'original_tiles': original_tiles,
            'original_rack': original_rack, 'original_blanks_set_at_typing_start': original_blanks_set_at_typing_start,
            'selected_square': selected_square, 'current_r': current_r, 'current_c': current_c,
            'typing_direction': typing_direction, 'typing_start': typing_start,
            'current_move_blanks_coords': current_move_blanks_coords
        })
        return state

    if specifying_rack:
        clicked_input_sr = False
        if p1_input_rect_sr and p1_input_rect_sr.collidepoint(x,y):
            specify_rack_active_input = 0; clicked_input_sr = True
        elif p2_input_rect_sr and p2_input_rect_sr.collidepoint(x,y):
            specify_rack_active_input = 1; clicked_input_sr = True

        if p1_reset_rect_sr and p1_reset_rect_sr.collidepoint(x,y):
            specify_rack_inputs[0] = "".join(sorted(specify_rack_original_racks[0])).upper()
        elif p2_reset_rect_sr and p2_reset_rect_sr.collidepoint(x,y):
            specify_rack_inputs[1] = "".join(sorted(specify_rack_original_racks[1])).upper()
        elif cancel_rect_sr and cancel_rect_sr.collidepoint(x,y):
            specifying_rack = False
            specify_rack_active_input = None
        elif confirm_rect_sr and confirm_rect_sr.collidepoint(x,y):
            proposed_p1_rack_str = specify_rack_inputs[0].upper().replace("?", " ")
            proposed_p2_rack_str = specify_rack_inputs[1].upper().replace("?", " ")
            if len(proposed_p1_rack_str) > 7 or len(proposed_p2_rack_str) > 7:
                show_message_dialog("Racks cannot exceed 7 tiles.", "Input Error")
            else:
                specify_rack_proposed_racks[0] = list(proposed_p1_rack_str)
                specify_rack_proposed_racks[1] = list(proposed_p2_rack_str)
                valid_chars = set(TILE_DISTRIBUTION.keys()) | {' '}
                p1_valid = all(char in valid_chars for char in specify_rack_proposed_racks[0])
                p2_valid = all(char in valid_chars for char in specify_rack_proposed_racks[1])
                if not p1_valid or not p2_valid:
                    show_message_dialog("Invalid characters in rack input.", "Input Error")
                else:
                    current_combined_tiles = Counter(racks[0]) + Counter(racks[1]) + Counter(bag)
                    proposed_combined_tiles = Counter(specify_rack_proposed_racks[0]) + Counter(specify_rack_proposed_racks[1])
                    can_form_proposed_from_current_pool = True
                    diff = proposed_combined_tiles - current_combined_tiles
                    if any(count > 0 for count in diff.values()):
                        can_form_proposed_from_current_pool = False
                    if not can_form_proposed_from_current_pool:
                         confirming_override = True
                         specifying_rack = False
                    else:
                        racks[0] = specify_rack_proposed_racks[0][:]
                        racks[1] = specify_rack_proposed_racks[1][:]
                        show_message_dialog("Racks specified. Bag may need manual adjustment if tiles were taken.", "Racks Set")
                        specifying_rack = False
                        typing = False; word_positions = []; original_tiles = None; original_rack = None; original_blanks_set_at_typing_start = None; selected_square = None; current_r = None; current_c = None; typing_direction = None; typing_start = None; current_move_blanks_coords = set()
        elif not clicked_input_sr and not (p1_reset_rect_sr and p1_reset_rect_sr.collidepoint(x,y)) \
             and not (p2_reset_rect_sr and p2_reset_rect_sr.collidepoint(x,y)) \
             and not (confirm_rect_sr and confirm_rect_sr.collidepoint(x,y)) \
             and not (cancel_rect_sr and cancel_rect_sr.collidepoint(x,y)):
            specify_rack_active_input = None
        state.update({
            'specifying_rack': specifying_rack,
            'specify_rack_inputs': specify_rack_inputs,
            'specify_rack_active_input': specify_rack_active_input,
            'specify_rack_proposed_racks': specify_rack_proposed_racks,
            'confirming_override': confirming_override,
            'racks': racks,
            'typing': typing, 'word_positions': word_positions, 'original_tiles': original_tiles,
            'original_rack': original_rack, 'original_blanks_set_at_typing_start': original_blanks_set_at_typing_start,
            'selected_square': selected_square, 'current_r': current_r, 'current_c': current_c,
            'typing_direction': typing_direction, 'typing_start': typing_start,
            'current_move_blanks_coords': current_move_blanks_coords
        })
        return state

    if exchanging:
        clicked_tile_in_exch_dlg_local_val = False
        for i_ex_dlg_val, rect_ex_dlg_val in enumerate(tile_rects_exch_dlg):
            if rect_ex_dlg_val.collidepoint(x, y):
                if i_ex_dlg_val in selected_tiles:
                    selected_tiles.remove(i_ex_dlg_val)
                else:
                    selected_tiles.add(i_ex_dlg_val)
                clicked_tile_in_exch_dlg_local_val = True
                break
        if cancel_button_exch_dlg and cancel_button_exch_dlg.collidepoint(x, y):
            exchanging = False
            selected_tiles.clear()
        elif exchange_button_exch_dlg and exchange_button_exch_dlg.collidepoint(x, y):
            if not selected_tiles:
                show_message_dialog("No tiles selected for exchange.", "Exchange Error")
            elif len(selected_tiles) > len(bag):
                show_message_dialog("Not enough tiles in bag to exchange.", "Exchange Error")
            else:
                tiles_to_exchange_val_dlg_val = [racks[turn-1][i_sel_dlg_val] for i_sel_dlg_val in selected_tiles]
                move_rack_exch_val_dlg_val = racks[turn-1][:]
                new_rack_after_exchange_dlg_val = [tile_val_dlg_val for i_r_dlg_val, tile_val_dlg_val in enumerate(racks[turn-1]) if i_r_dlg_val not in selected_tiles]
                num_to_draw_exch_val_dlg_val = len(tiles_to_exchange_val_dlg_val)
                drawn_tiles_exch_val_dlg_val = [bag.pop(0) for _ in range(num_to_draw_exch_val_dlg_val) if bag]
                new_rack_after_exchange_dlg_val.extend(drawn_tiles_exch_val_dlg_val)
                racks[turn-1] = new_rack_after_exchange_dlg_val
                bag.extend(tiles_to_exchange_val_dlg_val)
                random.shuffle(bag)
                human_played = True
                paused_for_power_tile = False
                paused_for_bingo_practice = False
                exchanging = False
                selected_tiles.clear()
                scores[turn-1] += 0
                consecutive_zero_point_turns += 1
                exchange_count += 1
                pass_count = 0
                luck_factor_exch_val_dlg_val = 0.0
                if drawn_tiles_exch_val_dlg_val:
                    try:
                        luck_factor_exch_val_dlg_val = calculate_luck_factor_cython(
                            drawn_tiles_exch_val_dlg_val,
                            move_rack_exch_val_dlg_val,
                            tiles,
                            blanks,
                            blanks_played_count,
                            get_remaining_tiles
                        )
                    except Exception as e_luck_exch_val_err_dlg_val:
                        print(f"Error calling calculate_luck_factor_cython for exchange: {e_luck_exch_val_err_dlg_val}")
                        luck_factor_exch_val_dlg_val = 0.0
                turn_duration_exch_val = 0.0
                if 'start_turn_time' in state:
                    turn_duration_exch_val = time.time() - state['start_turn_time']
                move_data_sgs_exch_val_dlg_val = {
                    'player': turn, 'move_type': 'exchange', 'score': 0, 'word': '', 'coord': '',
                    'positions': [], 'blanks': set(), 'leave': racks[turn-1][:], 'is_bingo': False, 'word_with_blanks': '',
                    'exchanged_tiles': tiles_to_exchange_val_dlg_val, 'turn_duration': turn_duration_exch_val, 'luck_factor': luck_factor_exch_val_dlg_val,
                    'newly_placed_details': [], 'tiles_placed_from_rack': [], 'blanks_played_info': [],
                    'rack_before_move': move_rack_exch_val_dlg_val, 'tiles_drawn_after_move': drawn_tiles_exch_val_dlg_val,
                    'is_first_play_after_move': first_play,
                    'board_tile_counts_after_move': board_tile_counts.copy(),
                    'blanks_played_count_after_move': blanks_played_count
                }
                move_history.append(move_data_sgs_exch_val_dlg_val)
                turn = 3 - turn
                last_played_highlight_coords = set()
        elif not clicked_tile_in_exch_dlg_local_val:
            dialog_width_exch, dialog_height_exch = 400, 200
            dialog_rect_exch = pygame.Rect((WINDOW_WIDTH - dialog_width_exch) // 2, (WINDOW_HEIGHT - dialog_height_exch) // 2, dialog_width_exch, dialog_height_exch)
            if not dialog_rect_exch.collidepoint(x,y):
                exchanging = False
                selected_tiles.clear()
        state.update({
            'exchanging': exchanging, 'selected_tiles': selected_tiles, 'racks': racks, 'bag': bag,
            'scores': scores, 'turn': turn, 'pass_count': pass_count, 'exchange_count': exchange_count,
            'consecutive_zero_point_turns': consecutive_zero_point_turns,
            'last_played_highlight_coords': last_played_highlight_coords,
            'move_history': move_history, 'human_played': human_played,
            'paused_for_power_tile': paused_for_power_tile,
            'paused_for_bingo_practice': paused_for_bingo_practice,
            'blanks_played_count': blanks_played_count
        })
        return state

    if hinting:
        print(f"DEBUG HINTING BLOCK: Click at ({x},{y}). all_words_btn: {all_words_button_hint_dlg}, ok_btn: {ok_button_hint_dlg}, play_btn: {play_exch_button_hint_dlg}")
        clicked_in_dialog_hint_local_logic_val_h_val = False 
        exchange_option_present_hint_local_logic_val_h_val = bool(best_exchange_for_hint)
        num_plays_shown_hint_local_logic_val_h_val = 0
        if isinstance(hint_moves, list):
            for i_h_chk_local_logic_val_h_val, move_item_h_chk_local_logic_val_h_val in enumerate(hint_moves):
                if i_h_chk_local_logic_val_h_val < 5:
                    num_plays_shown_hint_local_logic_val_h_val +=1
                else:
                    break
        exchange_display_index_hint_local_logic_val_h_val = num_plays_shown_hint_local_logic_val_h_val if exchange_option_present_hint_local_logic_val_h_val else -1

        if play_exch_button_hint_dlg and play_exch_button_hint_dlg.collidepoint(x, y):
            clicked_in_dialog_hint_local_logic_val_h_val = True
            if selected_hint_index is not None:
                if exchange_option_present_hint_local_logic_val_h_val and selected_hint_index == exchange_display_index_hint_local_logic_val_h_val:
                    if best_exchange_for_hint and len(bag) >= len(best_exchange_for_hint):
                        tiles_to_exchange_hint_logic_val_h_val = best_exchange_for_hint[:]; player_idx_hint_exch_logic_val_h_val = turn - 1; move_rack_hint_exch_logic_val_h_val = racks[player_idx_hint_exch_logic_val_h_val][:]; new_rack_hint_exch_logic_val_h_val = []; exchange_counts_temp_hint_logic_val_h_val = Counter(tiles_to_exchange_hint_logic_val_h_val)
                        for tile_in_rack_h_ex_logic_val_h_val in racks[player_idx_hint_exch_logic_val_h_val]:
                            if exchange_counts_temp_hint_logic_val_h_val[tile_in_rack_h_ex_logic_val_h_val] > 0:
                                exchange_counts_temp_hint_logic_val_h_val[tile_in_rack_h_ex_logic_val_h_val] -= 1
                            else:
                                new_rack_hint_exch_logic_val_h_val.append(tile_in_rack_h_ex_logic_val_h_val)
                        num_to_draw_h_ex_logic_val_h_val = len(tiles_to_exchange_hint_logic_val_h_val); drawn_tiles_h_ex_logic_val_h_val = [bag.pop(0) for _ in range(num_to_draw_h_ex_logic_val_h_val) if bag]; new_rack_hint_exch_logic_val_h_val.extend(drawn_tiles_h_ex_logic_val_h_val); racks[player_idx_hint_exch_logic_val_h_val] = new_rack_hint_exch_logic_val_h_val
                        bag.extend(tiles_to_exchange_hint_logic_val_h_val); random.shuffle(bag); luck_factor_h_ex_logic_val_h_val = 0.0
                        try:
                            luck_factor_h_ex_logic_val_h_val = calculate_luck_factor_cython(drawn_tiles_h_ex_logic_val_h_val, move_rack_hint_exch_logic_val_h_val, tiles, blanks, blanks_played_count, get_remaining_tiles)
                        except Exception as e_luck_h_ex_logic_val_err_h_val:
                            luck_factor_h_ex_logic_val_h_val = 0.0
                        scores[player_idx_hint_exch_logic_val_h_val] += 0; consecutive_zero_point_turns += 1; exchange_count += 1; pass_count = 0;
                        turn_duration_hint_exch_val = 0.0
                        if 'start_turn_time' in state:
                            turn_duration_hint_exch_val = time.time() - state['start_turn_time']
                        move_data_sgs_h_ex_logic_val_h_val = {
                            'player': turn, 'move_type': 'exchange', 'score': 0, 'word': '', 'coord': '',
                            'positions': [], 'blanks': set(), 'leave': racks[player_idx_hint_exch_logic_val_h_val][:],
                            'is_bingo': False, 'word_with_blanks': '',
                            'exchanged_tiles': tiles_to_exchange_hint_logic_val_h_val,
                            'turn_duration': turn_duration_hint_exch_val,
                            'luck_factor': luck_factor_h_ex_logic_val_h_val,
                            'newly_placed_details': [], 'tiles_placed_from_rack': [], 'blanks_played_info': [],
                            'rack_before_move': move_rack_hint_exch_logic_val_h_val,
                            'tiles_drawn_after_move': drawn_tiles_h_ex_logic_val_h_val,
                            'is_first_play_after_move': first_play,
                            'board_tile_counts_after_move': board_tile_counts.copy(),
                            'blanks_played_count_after_move': blanks_played_count
                        }
                        move_history.append(move_data_sgs_h_ex_logic_val_h_val)
                        turn = 3 - turn; last_played_highlight_coords = set()
                        human_played = True; hinting = False
                        state.update({
                            'hinting': hinting, 'showing_all_words': showing_all_words,
                            'human_played': human_played, 'turn': turn, 'first_play': first_play,
                            'scores': scores, 'racks': racks, 'bag': bag, 
                            'pass_count': pass_count, 'exchange_count': exchange_count,
                            'consecutive_zero_point_turns': consecutive_zero_point_turns,
                            'last_played_highlight_coords': last_played_highlight_coords,
                            'move_history': move_history,
                            'blanks_played_count': blanks_played_count
                        })
                        return state
                    else:
                        show_message_dialog("Cannot perform this exchange (not enough tiles in bag or no exchange option).", "Exchange Error")
                        hinting = False 
                        state.update({'hinting': hinting})
                        return state
                else: # Play a move from hint
                    selected_move_hint_local_logic_val_h_val = None
                    is_sim_res_hint_local_logic_val_h_val = bool(hint_moves and isinstance(hint_moves[0], dict) and 'final_score' in hint_moves[0])
                    if 0 <= selected_hint_index < len(hint_moves):
                        if is_sim_res_hint_local_logic_val_h_val:
                            selected_item_hint_local_logic_val_h_val = hint_moves[selected_hint_index]
                            selected_move_hint_local_logic_val_h_val = selected_item_hint_local_logic_val_h_val.get('move')
                        else:
                            selected_move_hint_local_logic_val_h_val = hint_moves[selected_hint_index]
                    
                    if selected_move_hint_local_logic_val_h_val:
                        player_idx_hint_play_local_logic_val_h_val = turn - 1
                        
                        (ret_scores, ret_player_rack, ret_opponent_rack, 
                         ret_tiles, ret_blanks, ret_bag, 
                         ret_first_play, ret_turn, 
                         ret_board_counts, ret_blanks_played_count_val, 
                         sgs_data_for_this_move_val_h_val, ret_highlight_coords) = play_hint_move(
                            selected_move_hint_local_logic_val_h_val,
                            scores, racks, 
                            player_idx_hint_play_local_logic_val_h_val, 
                            tiles, blanks, board, bag,
                            first_play, turn,
                            board_tile_counts, 
                            blanks_played_count 
                        )
                        
                        scores = list(ret_scores) 
                        racks[player_idx_hint_play_local_logic_val_h_val] = ret_player_rack
                        racks[1 - player_idx_hint_play_local_logic_val_h_val] = ret_opponent_rack
                        tiles = ret_tiles
                        blanks = ret_blanks
                        bag = ret_bag
                        first_play = ret_first_play
                        turn = ret_turn
                        board_tile_counts = ret_board_counts 
                        blanks_played_count = ret_blanks_played_count_val 
                        last_played_highlight_coords = ret_highlight_coords
                        
                        move_history.append(sgs_data_for_this_move_val_h_val)
                        human_played = True; hinting = False
                        
                        pass_count = 0
                        exchange_count = 0
                        consecutive_zero_point_turns = 0
                        state.update({
                            'hinting': hinting, 'showing_all_words': showing_all_words,
                            'human_played': human_played, 'turn': turn, 'first_play': first_play,
                            'scores': scores, 'racks': racks, 'bag': bag, 'tiles': tiles, 'blanks': blanks,
                            'pass_count': pass_count, 'exchange_count': exchange_count,
                            'consecutive_zero_point_turns': consecutive_zero_point_turns,
                            'last_played_highlight_coords': last_played_highlight_coords,
                            'move_history': move_history,
                            'board_tile_counts': board_tile_counts,
                            'blanks_played_count': blanks_played_count
                        })
                        return state
                    else:
                        show_message_dialog("No move selected or invalid selection.", "Hint Error")
                        hinting = False 
                        state.update({'hinting': hinting})
                        return state
            else: 
                show_message_dialog("Please select a hint to play or exchange.", "Hint Action")
                # No state change that requires immediate return if no hint was selected, so allow fall-through
                # to check other hint dialog interactions (like clicking a hint item or outside)
        
        elif all_words_button_hint_dlg and all_words_button_hint_dlg.collidepoint(x, y):
            # This is the critical block for transitioning to "All Words"
            print("DEBUG: 'All Words' button in Hint Dialog CLICKED.") 
            print(f"DEBUG: Flags before change: hinting={hinting}, showing_all_words={showing_all_words}, all_moves length: {len(all_moves if all_moves is not None else [])}")
            clicked_in_dialog_hint_local_logic_val_h_val = True # Mark that this hint dialog interaction was primary
            
            hinting = False 
            showing_all_words = True 
            
            current_all_moves_dlg_h_val = all_moves 
            moves_for_all_dlg_local_h_val = [] 
            
            if practice_mode == "eight_letter":
                moves_for_all_dlg_local_h_val = practice_target_moves if practice_target_moves else []
            elif practice_mode == "power_tiles" and paused_for_power_tile:
                moves_for_all_dlg_local_h_val = sorted(
                    [m for m in current_all_moves_dlg_h_val if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)],
                    key=lambda m: m.get('score',0), reverse=True
                )
            elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice:
                moves_for_all_dlg_local_h_val = sorted(
                    [m for m in current_all_moves_dlg_h_val if m.get('is_bingo', False)],
                    key=lambda m: m.get('score',0), reverse=True
                )
            else: 
                moves_for_all_dlg_local_h_val = current_all_moves_dlg_h_val

            new_selected_hint_index = 0 if moves_for_all_dlg_local_h_val else None 
            new_all_words_scroll_offset = 0
            print(f"DEBUG: Flags SET by All Words button: hinting={hinting}, showing_all_words={showing_all_words}")
            
            state.update({
                'hinting': hinting, 
                'showing_all_words': showing_all_words,
                'selected_hint_index': new_selected_hint_index, 
                'all_words_scroll_offset': new_all_words_scroll_offset, 
                'all_moves': all_moves, # Ensure all_moves is passed for the new dialog
                # clicked_in_dialog_hint is not needed in the global state from here
            })
            return state # <<<< IMMEDIATE RETURN AFTER HANDLING THIS BUTTON
        
        elif ok_button_hint_dlg and ok_button_hint_dlg.collidepoint(x, y):
            clicked_in_dialog_hint_local_logic_val_h_val = True
            hinting = False
            # No other dialog is opened, just closing hint.
            state.update({
                'hinting': hinting,
                'showing_all_words': showing_all_words, # Preserve this flag, should be False if only OK was hit
            })
            return state
        else: # Clicked inside hint dialog but not on a button that changes primary dialog state
              # This section handles clicks on individual hint items or outside the dialog.
            for i_h_local_logic_val_h_val, rect_h_local_logic_val_h_val in enumerate(hint_rects_list_hint_dlg):
                if rect_h_local_logic_val_h_val.collidepoint(x, y):
                    selected_hint_index = i_h_local_logic_val_h_val
                    clicked_in_dialog_hint_local_logic_val_h_val = True
                    break 

            if not clicked_in_dialog_hint_local_logic_val_h_val: # Click was not on any button or hint item
                dialog_width_h_evt_local_h_val_close, dialog_height_h_evt_local_h_val_close = 400, 280 
                dialog_rect_h_evt_local_h_val_close = pygame.Rect(
                    (WINDOW_WIDTH - dialog_width_h_evt_local_h_val_close) // 2,
                    (WINDOW_HEIGHT - dialog_height_h_evt_local_h_val_close) // 2,
                    dialog_width_h_evt_local_h_val_close, dialog_height_h_evt_local_h_val_close
                )
                if not dialog_rect_h_evt_local_h_val_close.collidepoint(x,y): # Clicked outside hint dialog
                    hinting = False # Close hint dialog
            
            # This update is for clicks on hint items or clicks outside the dialog that closes it.
            # showing_all_words should not be changed here unless a button explicitly did it.
            state.update({
                'hinting': hinting, 
                'selected_hint_index': selected_hint_index,
                'showing_all_words': showing_all_words 
            })
            return state

    elif showing_all_words: # All Words Dialog Logic
        clicked_in_dialog_aw_evt_val = False
        current_all_moves_aw_evt_val = all_moves
        moves_for_all_aw_evt_val = []
        if practice_mode == "eight_letter":
            moves_for_all_aw_evt_val = practice_target_moves if practice_target_moves else []
        elif practice_mode == "power_tiles" and paused_for_power_tile:
            moves_for_all_aw_evt_val = sorted(
                [m for m in current_all_moves_aw_evt_val if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)],
                key=lambda m: m.get('score',0), reverse=True
            )
        elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice:
            moves_for_all_aw_evt_val = sorted(
                [m for m in current_all_moves_aw_evt_val if m.get('is_bingo', False)],
                key=lambda m: m.get('score',0), reverse=True
            )
        else:
            moves_for_all_aw_evt_val = current_all_moves_aw_evt_val

        if all_words_play_rect_aw_dlg and all_words_play_rect_aw_dlg.collidepoint(x, y):
            clicked_in_dialog_aw_evt_val = True
            if selected_hint_index is not None and 0 <= selected_hint_index < len(moves_for_all_aw_evt_val):
                selected_move_aw_evt_val = moves_for_all_aw_evt_val[selected_hint_index]
                player_idx_aw_play_evt_val = turn - 1
                valid_practice_play_aw = True
                if practice_mode == "eight_letter":
                    if not practice_target_moves or selected_move_aw_evt_val not in practice_target_moves: pass
                    else:
                        practice_solved = True
                        practice_end_message = f"Correct! You found: {selected_move_aw_evt_val.get('word_with_blanks', selected_move_aw_evt_val.get('word',''))}"
                        showing_practice_end_dialog = True
                elif practice_mode == "power_tiles" and paused_for_power_tile:
                    uses_power_aw = any(letter == current_power_tile for _, _, letter in selected_move_aw_evt_val.get('newly_placed',[]))
                    word_len_ok_aw = is_word_length_allowed(len(selected_move_aw_evt_val.get('word','')), number_checks)
                    if not (uses_power_aw and word_len_ok_aw):
                        valid_practice_play_aw = False
                        show_message_dialog(f"Move must use '{current_power_tile}' and meet length criteria.", "Practice Rule")
                    else:
                        practice_solved = True
                        practice_end_message = f"Good power tile play: {selected_move_aw_evt_val.get('word_with_blanks', selected_move_aw_evt_val.get('word',''))}"
                        showing_practice_end_dialog = True
                        paused_for_power_tile = False
                elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice:
                    if not selected_move_aw_evt_val.get('is_bingo', False):
                        valid_practice_play_aw = False
                        show_message_dialog("Move must be a bingo.", "Practice Rule")
                    else:
                        practice_solved = True
                        practice_end_message = f"Bingo! {selected_move_aw_evt_val.get('word_with_blanks', selected_move_aw_evt_val.get('word',''))}"
                        showing_practice_end_dialog = True
                        paused_for_bingo_practice = False
                
                if valid_practice_play_aw:
                    (ret_scores_aw, ret_player_rack_aw, ret_opponent_rack_aw,
                     ret_tiles_aw, ret_blanks_aw, ret_bag_aw,
                     ret_first_play_aw, ret_turn_aw,
                     ret_board_counts_aw, ret_blanks_played_count_aw_val, 
                     sgs_data_aw, ret_highlight_coords_aw) = play_hint_move(
                        selected_move_aw_evt_val, scores, racks, 
                        player_idx_aw_play_evt_val,
                        tiles, blanks, board, bag,
                        first_play, turn, board_tile_counts, blanks_played_count 
                    )
                    scores = list(ret_scores_aw)
                    racks[player_idx_aw_play_evt_val] = ret_player_rack_aw
                    racks[1 - player_idx_aw_play_evt_val] = ret_opponent_rack_aw
                    tiles = ret_tiles_aw; blanks = ret_blanks_aw; bag = ret_bag_aw;
                    first_play = ret_first_play_aw; turn = ret_turn_aw
                    board_tile_counts = ret_board_counts_aw; blanks_played_count = ret_blanks_played_count_aw_val 
                    last_played_highlight_coords = ret_highlight_coords_aw
                    move_history.append(sgs_data_aw)
                    human_played = True
                    showing_all_words = False
                    if not showing_practice_end_dialog:
                        paused_for_power_tile = False
                        paused_for_bingo_practice = False
            else:
                show_message_dialog("No move selected to play.", "Play Error")
            state.update({
                'showing_all_words': showing_all_words, 'human_played': human_played, 'turn': turn, 
                'first_play': first_play, 'scores': scores, 'racks': racks, 'bag': bag, 
                'tiles': tiles, 'blanks': blanks, 'pass_count': pass_count, 
                'exchange_count': exchange_count, 'consecutive_zero_point_turns': consecutive_zero_point_turns,
                'last_played_highlight_coords': last_played_highlight_coords, 'move_history': move_history,
                'board_tile_counts': board_tile_counts, 'blanks_played_count': blanks_played_count,
                'practice_solved': practice_solved, 'practice_end_message': practice_end_message,
                'showing_practice_end_dialog': showing_practice_end_dialog,
                'paused_for_power_tile': paused_for_power_tile, 'paused_for_bingo_practice': paused_for_bingo_practice
            })
            return state
        elif all_words_ok_rect_aw_dlg and all_words_ok_rect_aw_dlg.collidepoint(x, y):
            clicked_in_dialog_aw_evt_val = True
            showing_all_words = False
            state.update({'showing_all_words': showing_all_words})
            return state
        
        if not clicked_in_dialog_aw_evt_val: # If click wasn't on Play or OK
            for rect_info_aw_evt_val in all_words_rects_list_aw_dlg: 
                rect_aw_evt_val, idx_aw_evt_val = rect_info_aw_evt_val 
                # vvVvv ADD THIS CHECK vvVvv
                if rect_aw_evt_val is not None and rect_aw_evt_val.collidepoint(x, y):
                    clicked_in_dialog_aw_evt_val = True
                    selected_hint_index = idx_aw_evt_val
                    break
        
        if not clicked_in_dialog_aw_evt_val: 
            dialog_rect_aw_main_evt_val = pygame.Rect(
                (WINDOW_WIDTH - ALL_WORDS_DIALOG_WIDTH) // 2,
                (WINDOW_HEIGHT - ALL_WORDS_DIALOG_HEIGHT) // 2,
                ALL_WORDS_DIALOG_WIDTH, ALL_WORDS_DIALOG_HEIGHT
            )
            if not dialog_rect_aw_main_evt_val.collidepoint(x,y): 
                showing_all_words = False
        
        state.update({
            'showing_all_words': showing_all_words,
            'selected_hint_index': selected_hint_index,
        })
        return state

    else: # Not in a dialog that consumes clicks, process general game interactions
        if not is_batch_running :
            current_time_main_evt = pygame.time.get_ticks()
            if options_rect_base and options_rect_base.collidepoint(x, y):
                dropdown_open = not dropdown_open
            elif dropdown_open:
                clicked_dropdown_item = False
                for i_dd_opt_val, rect_dd_opt_val in enumerate(dropdown_rects_base):
                    if rect_dd_opt_val.collidepoint(x, y):
                        clicked_dropdown_item = True
                        options_list_dd_val = []
                        if is_batch_running: options_list_dd_val = ["Stop Batch", "Quit"]
                        elif game_over_state or practice_mode: options_list_dd_val = ["Main", "Quit"]
                        elif replay_mode: options_list_dd_val = ["Main", "Quit"]
                        else: options_list_dd_val = ["Pass", "Exchange", "Specify Rack", "Main", "Quit"]
                        selected_option_dd_val = options_list_dd_val[i_dd_opt_val]
                        if selected_option_dd_val == "Pass":
                            if not typing:
                                scores[turn-1] += 0
                                consecutive_zero_point_turns += 1
                                pass_count += 1
                                exchange_count = 0
                                human_played = True
                                paused_for_power_tile = False
                                paused_for_bingo_practice = False
                                turn_duration_pass_val = 0.0
                                if 'start_turn_time' in state:
                                    turn_duration_pass_val = time.time() - state['start_turn_time']
                                move_data_sgs_pass_val = {
                                    'player': turn, 'move_type': 'pass', 'score': 0,
                                    'word': '', 'coord': '', 'positions': [], 'blanks': set(),
                                    'leave': racks[turn-1][:], 'is_bingo': False, 'word_with_blanks': '',
                                    'exchanged_tiles': [], 'turn_duration': turn_duration_pass_val, 'luck_factor': 0.0,
                                    'newly_placed_details': [], 'tiles_placed_from_rack': [], 'blanks_played_info': [],
                                    'rack_before_move': racks[turn-1][:], 'tiles_drawn_after_move': [],
                                    'is_first_play_after_move': first_play,
                                    'board_tile_counts_after_move': board_tile_counts.copy(),
                                    'blanks_played_count_after_move': blanks_played_count
                                }
                                move_history.append(move_data_sgs_pass_val)
                                turn = 3 - turn
                                last_played_highlight_coords = set()
                            else:
                                show_message_dialog("Cannot pass while typing a word. Press ESC to cancel typing.", "Pass Error")
                        elif selected_option_dd_val == "Exchange":
                            if not typing:
                                if len(bag) > 0:
                                    exchanging = True; selected_tiles.clear()
                                else: show_message_dialog("Bag is empty, cannot exchange.", "Exchange Error")
                            else: show_message_dialog("Cannot exchange while typing. Press ESC to cancel.", "Exchange Error")
                        elif selected_option_dd_val == "Specify Rack":
                            if not typing:
                                specifying_rack = True
                                specify_rack_original_racks[0] = racks[0][:] if len(racks) > 0 and racks[0] is not None else []
                                specify_rack_original_racks[1] = racks[1][:] if len(racks) > 1 and racks[1] is not None else []
                                specify_rack_inputs[0] = "".join(sorted(specify_rack_original_racks[0])).upper().replace(" ", "?")
                                specify_rack_inputs[1] = "".join(sorted(specify_rack_original_racks[1])).upper().replace(" ", "?")
                                specify_rack_active_input = 0
                            else: show_message_dialog("Cannot specify rack while typing. Press ESC to cancel.", "Action Error")
                        elif selected_option_dd_val == "Main":
                            running_inner = False; return_to_mode_selection = True
                        elif selected_option_dd_val == "Quit":
                            running_inner = False
                        elif selected_option_dd_val == "Stop Batch":
                            batch_stop_requested = True
                            show_message_dialog("Batch will stop after the current game.", "Batch Control")
                        dropdown_open = False
                        break
                if not clicked_dropdown_item:
                    dropdown_open = False
            if game_over_state:
                dialog_rect_go_val = pygame.Rect(dialog_x, dialog_y, DIALOG_WIDTH, DIALOG_HEIGHT)
                if save_rect and save_rect.collidepoint(x,y):
                    now_go_save_val = datetime.datetime.now()
                    date_str_go_save_val = now_go_save_val.strftime("%d%b%y").upper()
                    time_str_go_save_val = now_go_save_val.strftime("%H%M")
                    p1_name_file_go_save_val = player_names[0] if player_names[0] else "P1"
                    p2_name_file_go_save_val = player_names[1] if player_names[1] else "P2"
                    p1_name_file_go_save_val = "".join(c if c.isalnum() else "_" for c in p1_name_file_go_save_val)
                    p2_name_file_go_save_val = "".join(c if c.isalnum() else "_" for c in p2_name_file_go_save_val)
                    filename_go_save_val = f"{date_str_go_save_val}-{time_str_go_save_val}-{p1_name_file_go_save_val}-vs-{p2_name_file_go_save_val}.sgs"
                    game_data_to_save_val = {
                        'version': 1.0, 'game_mode': game_mode, 'player_names': player_names,
                        'is_ai': is_ai, 'scores': scores, 'final_scores': final_scores,
                        'move_history': move_history,
                        'initial_racks_sgs': initial_racks,
                        'initial_shuffled_bag_order_sgs': state.get('initial_shuffled_bag_order_sgs', []),
                        'practice_mode': practice_mode,
                        'use_endgame_solver_setting': state.get('USE_ENDGAME_SOLVER', False),
                        'use_ai_simulation_setting': state.get('USE_AI_SIMULATION', False),
                    }
                    save_game_to_sgs(filename_go_save_val, game_data_to_save_val)
                    show_message_dialog(f"Game saved as {filename_go_save_val}", "Game Saved")
                elif quit_rect and quit_rect.collidepoint(x,y):
                    running_inner = False
                elif replay_rect and replay_rect.collidepoint(x,y):
                    game_over_state = False; replay_mode = True; current_replay_turn = 0
                elif play_again_rect and play_again_rect.collidepoint(x,y):
                    if practice_mode:
                        restart_practice_mode = True
                        running_inner = True
                        game_over_state = False
                    else:
                        running_inner = False
                        return_to_mode_selection = True
                elif stats_rect and stats_rect.collidepoint(x,y):
                    showing_stats = True
                    stats_scroll_offset = 0
                elif not dialog_rect_go_val.collidepoint(x,y) and not showing_stats:
                    pass
                elif showing_stats and stats_ok_button_rect and stats_ok_button_rect.collidepoint(x,y):
                    showing_stats = False
                elif showing_stats:
                    stats_dialog_current_rect = pygame.Rect(stats_dialog_x, stats_dialog_y, 480, 600)
                    if stats_dialog_current_rect.collidepoint(x,y):
                        if y < stats_dialog_y + 30:
                             stats_dialog_dragging = True
                             stats_dialog_drag_offset = (x - stats_dialog_x, y - stats_dialog_y)
                state.update({
                    'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection,
                    'game_over_state': game_over_state, 'replay_mode': replay_mode, 'current_replay_turn': current_replay_turn,
                    'showing_stats': showing_stats, 'stats_scroll_offset': stats_scroll_offset,
                    'stats_dialog_x': stats_dialog_x, 'stats_dialog_y': stats_dialog_y,
                    'stats_dialog_dragging': stats_dialog_dragging, 'stats_dialog_drag_offset': stats_dialog_drag_offset,
                    'restart_practice_mode': restart_practice_mode,
                    'dropdown_open': dropdown_open
                })
                return state
            elif replay_mode:
                if replay_start_rect and replay_start_rect.collidepoint(x,y): current_replay_turn = 0
                elif replay_prev_rect and replay_prev_rect.collidepoint(x,y): current_replay_turn = max(0, current_replay_turn - 1)
                elif replay_next_rect and replay_next_rect.collidepoint(x,y): current_replay_turn = min(len(move_history), current_replay_turn + 1)
                elif replay_end_rect and replay_end_rect.collidepoint(x,y): current_replay_turn = len(move_history)
                state.update({
                    'current_replay_turn': current_replay_turn,
                    'dropdown_open': dropdown_open
                })
                return state
            
            is_human_turn_or_paused_for_click_main_val = (0 <= turn-1 < len(is_ai)) and \
                (not is_ai[turn-1] or paused_for_power_tile or paused_for_bingo_practice)

            if suggest_rect_base and suggest_rect_base.collidepoint(x, y) and is_human_turn_or_paused_for_click_main_val:
                event_is_set_suggest_val = gaddag_loaded_event is not None and gaddag_loaded_event.is_set()
                status_is_loaded_from_state_suggest_val = (current_gaddag_loading_status_in_state == 'loaded')
                gaddag_structure_is_ready_suggest_val = GADDAG_STRUCTURE is not None
                dawg_is_ready_global_suggest_val = DAWG is not None
                data_truly_ready_suggest_val = event_is_set_suggest_val and status_is_loaded_from_state_suggest_val and gaddag_structure_is_ready_suggest_val and dawg_is_ready_global_suggest_val
                if not data_truly_ready_suggest_val:
                    show_message_dialog("AI data is still loading. Please wait.", "Suggest Moves")
                    hint_moves = []; all_moves = []; hinting = False; selected_hint_index = None
                else:
                    current_player_rack_hmd_sug_val_val = racks[turn-1]
                    if current_player_rack_hmd_sug_val_val is not None:
                        gen_start_time_sug_val = time.time()
                        all_moves_generated_for_hint_hmd_sug_val_val = generate_all_moves_gaddag_cython(current_player_rack_hmd_sug_val_val, tiles, board, blanks, GADDAG_STRUCTURE.root, DAWG)
                        gen_duration_sug_val = time.time() - gen_start_time_sug_val
                        if all_moves_generated_for_hint_hmd_sug_val_val is None:
                            all_moves_generated_for_hint_hmd_sug_val_val = []
                        moves_to_hint_local_hmd_sug_val_val = []
                        if practice_mode == "power_tiles" and paused_for_power_tile:
                            power_moves_hint_hmd_sug_val_val = [m for m in all_moves_generated_for_hint_hmd_sug_val_val if any(letter == current_power_tile for _, _, letter in m.get('newly_placed',[])) and is_word_length_allowed(len(m.get('word','')), number_checks)]
                            moves_to_hint_local_hmd_sug_val_val = sorted(power_moves_hint_hmd_sug_val_val, key=lambda m: m.get('score',0), reverse=True)
                        elif practice_mode == "bingo_bango_bongo" and paused_for_bingo_practice:
                            bingo_moves_hint_hmd_sug_val_val = [m for m in all_moves_generated_for_hint_hmd_sug_val_val if m.get('is_bingo', False)]
                            moves_to_hint_local_hmd_sug_val_val = sorted(bingo_moves_hint_hmd_sug_val_val, key=lambda m: m.get('score',0), reverse=True)
                        elif practice_mode == "eight_letter":
                             moves_to_hint_local_hmd_sug_val_val = practice_target_moves if practice_target_moves else []
                        else:
                            moves_to_hint_local_hmd_sug_val_val = all_moves_generated_for_hint_hmd_sug_val_val
                        all_moves = all_moves_generated_for_hint_hmd_sug_val_val
                        hint_moves = moves_to_hint_local_hmd_sug_val_val[:5]
                        hinting = True
                        selected_hint_index = 0 if hint_moves else None
                        if len(bag) > 0:
                            best_exchange_for_hint, best_exchange_score_for_hint = find_best_exchange_option(
                                current_player_rack_hmd_sug_val_val, tiles, blanks, blanks_played_count, len(bag)
                            )
                        else:
                            best_exchange_for_hint = None
                            best_exchange_score_for_hint = -float('inf')
                    else:
                        show_message_dialog("Current player's rack is not available.", "Suggest Moves Error")
                        hint_moves = []; all_moves = []; hinting = False; selected_hint_index = None; best_exchange_for_hint = None; best_exchange_score_for_hint = -float('inf')
            elif simulate_button_rect and simulate_button_rect.collidepoint(x,y) and is_human_turn_or_paused_for_click_main_val:
                showing_simulation_config = True
                simulation_config_inputs = [str(DEFAULT_AI_CANDIDATES), str(DEFAULT_OPPONENT_SIMULATIONS), str(DEFAULT_POST_SIM_CANDIDATES)]
                simulation_config_active_input = None
            elif preview_checkbox_rect and preview_checkbox_rect.collidepoint(x,y) and is_human_turn_or_paused_for_click_main_val:
                preview_score_enabled = not preview_score_enabled
            elif p1_alpha_rect and p1_alpha_rect.collidepoint(x,y) and turn == 1 and is_human_turn_or_paused_for_click_main_val:
                if racks[0] is not None: racks[0].sort()
            elif p1_rand_rect and p1_rand_rect.collidepoint(x,y) and turn == 1 and is_human_turn_or_paused_for_click_main_val:
                if racks[0] is not None: random.shuffle(racks[0])
            elif p2_alpha_rect and p2_alpha_rect.collidepoint(x,y) and turn == 2 and is_human_turn_or_paused_for_click_main_val:
                if racks[1] is not None: racks[1].sort()
            elif p2_rand_rect and p2_rand_rect.collidepoint(x,y) and turn == 2 and is_human_turn_or_paused_for_click_main_val:
                if racks[1] is not None: random.shuffle(racks[1])
            elif is_human_turn_or_paused_for_click_main_val :
                rack_y_drag_evt_main_val = BOARD_SIZE + 80 if turn == 1 else BOARD_SIZE + 150
                rack_width_calc_evt_main_val = 7 * (TILE_WIDTH + TILE_GAP) - TILE_GAP
                replay_area_end_x_drag_evt_main_val = 10 + 4 * (REPLAY_BUTTON_WIDTH + REPLAY_BUTTON_GAP)
                min_rack_start_x_drag_evt_main_val = replay_area_end_x_drag_evt_main_val + BUTTON_GAP + 20
                rack_start_x_calc_drag_evt_main_val = max(min_rack_start_x_drag_evt_main_val, (BOARD_SIZE - rack_width_calc_evt_main_val) // 2)
                rack_len_drag_evt_main_val = len(racks[turn-1]) if racks[turn-1] is not None else 0
                tile_idx_drag_evt_main_val = get_tile_under_mouse(x, y, rack_start_x_calc_drag_evt_main_val, rack_y_drag_evt_main_val, rack_len_drag_evt_main_val)
                if tile_idx_drag_evt_main_val is not None:
                    dragged_tile = (turn, tile_idx_drag_evt_main_val)
                    drag_pos = (x, y)
                    tile_abs_x_drag_evt_main_val = rack_start_x_calc_drag_evt_main_val + tile_idx_drag_evt_main_val * (TILE_WIDTH + TILE_GAP)
                    tile_center_x_drag_evt_main_val = tile_abs_x_drag_evt_main_val + TILE_WIDTH // 2
                    tile_center_y_drag_evt_main_val = rack_y_drag_evt_main_val + TILE_HEIGHT // 2
                    drag_offset = (x - tile_center_x_drag_evt_main_val, y - tile_center_y_drag_evt_main_val)
                else:
                    potential_col_board_evt_main_val = (x - 40) // SQUARE_SIZE
                    potential_row_board_evt_main_val = (y - 40) // SQUARE_SIZE
                    is_on_board_click_evt_main_val = (40 <= x < 40 + GRID_SIZE * SQUARE_SIZE and 40 <= y < 40 + GRID_SIZE * SQUARE_SIZE)
                    if is_on_board_click_evt_main_val:
                        row_click_evt_main_val, col_click_evt_main_val = potential_row_board_evt_main_val, potential_col_board_evt_main_val
                        current_selection_click_evt_main_val = selected_square
                        current_time_click_evt_main_val = pygame.time.get_ticks()
                        if typing and selected_square == (row_click_evt_main_val, col_click_evt_main_val):
                            if current_time_click_evt_main_val - last_left_click_time < DOUBLE_CLICK_TIME and last_left_click_pos == (row_click_evt_main_val, col_click_evt_main_val):
                                typing_direction = "down" if typing_direction == "right" else "right"
                            last_left_click_time = current_time_click_evt_main_val
                            last_left_click_pos = (row_click_evt_main_val,col_click_evt_main_val)
                        elif typing and selected_square != (row_click_evt_main_val, col_click_evt_main_val):
                            if original_tiles and original_rack:
                                tiles = [r_ot_evt_main_revert_val[:] for r_ot_evt_main_revert_val in original_tiles]
                                racks[turn-1] = original_rack[:]
                                if original_blanks_set_at_typing_start is not None:
                                    blanks = original_blanks_set_at_typing_start.copy()
                            typing = False; word_positions = []; original_tiles = None; original_rack = None; original_blanks_set_at_typing_start = None;
                            selected_square = None; current_r = None; current_c = None; typing_direction = None; typing_start = None; current_move_blanks_coords = set()
                            if not tiles[row_click_evt_main_val][col_click_evt_main_val]:
                                selected_square = (row_click_evt_main_val, col_click_evt_main_val)
                                current_r, current_c = row_click_evt_main_val, col_click_evt_main_val
                                typing = True; typing_direction = "right"; typing_start = (row_click_evt_main_val, col_click_evt_main_val)
                                word_positions = []; original_tiles = [r_ot_evt_main_new_val[:] for r_ot_evt_main_new_val in tiles]; original_rack = racks[turn-1][:]; original_blanks_set_at_typing_start = blanks.copy()
                            last_left_click_time = current_time_click_evt_main_val
                            last_left_click_pos = (row_click_evt_main_val,col_click_evt_main_val)
                        elif not tiles[row_click_evt_main_val][col_click_evt_main_val]:
                            selected_square = (row_click_evt_main_val, col_click_evt_main_val)
                            current_r, current_c = row_click_evt_main_val, col_click_evt_main_val
                            typing = True; typing_direction = "right"; typing_start = (row_click_evt_main_val, col_click_evt_main_val)
                            word_positions = []; original_tiles = [r_ot_evt_main_new_val[:] for r_ot_evt_main_new_val in tiles]; original_rack = racks[turn-1][:]; original_blanks_set_at_typing_start = blanks.copy()
                            last_left_click_time = current_time_click_evt_main_val
                            last_left_click_pos = (row_click_evt_main_val,col_click_evt_main_val)
                        else:
                            selected_square = None
                            current_r, current_c = None, None
                    else:
                        if typing:
                            if original_tiles and original_rack:
                                tiles = [r_ot_evt_main_cancel_val[:] for r_ot_evt_main_cancel_val in original_tiles]
                                racks[turn-1] = original_rack[:]
                                if original_blanks_set_at_typing_start is not None:
                                    blanks = original_blanks_set_at_typing_start.copy()
                            typing = False; typing_start = None; typing_direction = None; word_positions = []; original_tiles = None; original_rack = None; original_blanks_set_at_typing_start = None; current_move_blanks_coords = set()
                        selected_square = None
                        current_r, current_c = None, None
            else:
                if typing:
                    if original_tiles and original_rack:
                        tiles = [r_ot_typing_cancel_val[:] for r_ot_typing_cancel_val in original_tiles]
                        racks[turn-1] = original_rack[:]
                        if original_blanks_set_at_typing_start is not None:
                            blanks = original_blanks_set_at_typing_start.copy()
                    typing = False; typing_start = None; typing_direction = None; word_positions = []; original_tiles = None; original_rack = None; original_blanks_set_at_typing_start = None; current_move_blanks_coords = set()
                selected_square = None
                current_r, current_c = None, None

    state.update({
        'running_inner': running_inner, 'return_to_mode_selection': return_to_mode_selection,
        'batch_stop_requested': batch_stop_requested, 'human_played': human_played,
        'turn': turn, 'dropdown_open': dropdown_open, 'bag_count': len(bag),
        'game_over_state': game_over_state, 'practice_mode': practice_mode,
        'exchanging': exchanging, 'hinting': hinting, 'showing_all_words': showing_all_words,
        'specifying_rack': specifying_rack, 'showing_simulation_config': showing_simulation_config,
        'showing_practice_end_dialog': showing_practice_end_dialog,
        'confirming_override': confirming_override, 'final_scores': final_scores,
        'player_names': player_names, 'move_history': move_history, 'initial_racks': initial_racks,
        'showing_stats': showing_stats, 'stats_dialog_x': stats_dialog_x, 'stats_dialog_y': stats_dialog_y,
        'dialog_x': dialog_x, 'dialog_y': dialog_y, 'current_replay_turn': current_replay_turn,
        'selected_tiles': selected_tiles, 'is_ai': is_ai,
        'specify_rack_original_racks': specify_rack_original_racks,
        'specify_rack_inputs': specify_rack_inputs,
        'specify_rack_active_input': specify_rack_active_input,
        'specify_rack_proposed_racks': specify_rack_proposed_racks,
        'racks': racks, 'scores': scores, 'bag': bag, 'tiles': tiles, 'blanks': blanks,
        'paused_for_power_tile': paused_for_power_tile,
        'paused_for_bingo_practice': paused_for_bingo_practice,
        'practice_best_move': practice_best_move, 'all_moves': all_moves,
        'current_power_tile': current_power_tile, 'number_checks': number_checks,
        'first_play': first_play, 'pass_count': pass_count, 'exchange_count': exchange_count,
        'consecutive_zero_point_turns': consecutive_zero_point_turns,
        'last_played_highlight_coords': last_played_highlight_coords,
        'practice_solved': practice_solved, 'practice_end_message': practice_end_message,
        'simulation_config_inputs': simulation_config_inputs,
        'simulation_config_active_input': simulation_config_active_input,
        'hint_moves': hint_moves, 'selected_hint_index': selected_hint_index,
        'preview_score_enabled': preview_score_enabled,
        'dragged_tile': dragged_tile, 'drag_pos': drag_pos, 'drag_offset': drag_offset,
        'typing': typing, 'word_positions': word_positions, 'original_tiles': original_tiles,
        'original_rack': original_rack, 'original_blanks_set_at_typing_start': original_blanks_set_at_typing_start,
        'selected_square': selected_square, 'last_left_click_time': last_left_click_time,
        'last_left_click_pos': last_left_click_pos,
        'stats_dialog_dragging': stats_dialog_dragging, 'dragging': dragging,
        'letter_checks': letter_checks, 'stats_scroll_offset': stats_scroll_offset,
        'stats_dialog_drag_offset': stats_dialog_drag_offset,
        'all_words_scroll_offset': all_words_scroll_offset,
        'restart_practice_mode': restart_practice_mode,
        'stats_total_content_height': stats_total_content_height,
        'board_tile_counts': board_tile_counts,
        'blanks_played_count': blanks_played_count,
        'practice_target_moves': practice_target_moves,
        'current_r': current_r, 'current_c': current_c, 'typing_direction': typing_direction, 'typing_start': typing_start,
        'best_exchange_for_hint': best_exchange_for_hint,
        'best_exchange_score_for_hint': best_exchange_score_for_hint,
        'current_move_blanks_coords': current_move_blanks_coords
    })
    return state
   





# In Scrabble Game.py
def check_and_handle_game_over(state):
    """Checks for game over conditions and handles the end of a game, including batch processing."""
    global batch_results 

    # MODIFICATION START: Remove verbose debug prints
    # print(f"DEBUG check_and_handle_game_over: Entry. type(state) = {type(state)}")
    # if isinstance(state, dict):
    #     print(f"DEBUG check_and_handle_game_over: Keys in received state: {sorted(list(state.keys()))}")
    #     if 'batch_results' not in state: 
    #         print(f"DEBUG check_and_handle_game_over: Key 'batch_results' IS MISSING from received state dictionary!")
    #     if 'is_batch_running' not in state: 
    #         print(f"DEBUG check_and_handle_game_over: 'is_batch_running' IS MISSING!")
    # else:
    #     print(f"DEBUG check_and_handle_game_over: Received state IS NOT A DICT. Value: {state}")
    #     return state 
    # MODIFICATION END

    replay_mode = state.get('replay_mode', False) 
    game_over_state = state.get('game_over_state', False)
    practice_mode = state.get('practice_mode')
    bag = state.get('bag', [])
    racks = state.get('racks', [[],[]])
    consecutive_zero_point_turns = state.get('consecutive_zero_point_turns', 0)
    scores = state.get('scores', [0,0])
    final_scores = state.get('final_scores') 
    reason_for_ending_local = state.get('reason', '') 

    is_batch_running = state.get('is_batch_running', False)
    current_batch_game_num = state.get('current_batch_game_num', 0)
    initial_game_config = state.get('initial_game_config', {}) 
    player_names = state.get('player_names', ["P1", "P2"]) 
    move_history = state.get('move_history', []) 
    current_game_initial_racks = state.get('initial_racks_sgs', state.get('initial_racks', [[],[]]))
    practice_solved = state.get('practice_solved', False)

    # Use global batch_results directly for appending if in batch mode
    # state.get('batch_results', []) can be used if reading this state elsewhere

    if game_over_state or replay_mode: 
        return state

    game_ended = False
    
    rack0_exists = len(racks) > 0 and racks[0] is not None
    rack1_exists = len(racks) > 1 and racks[1] is not None
    rack0_empty = rack0_exists and not racks[0]
    rack1_empty = rack1_exists and not racks[1]

    if not bag and (rack0_empty or rack1_empty):
        game_ended = True
        reason_for_ending_local = "Bag empty & rack empty"
    elif consecutive_zero_point_turns >= 6:
        game_ended = True
        reason_for_ending_local = "Six Consecutive Zero-Point Turns"
    
    if game_ended:
        print(f"Game Over: {reason_for_ending_local}")
        game_over_state_ended = True # Use local var for clarity
        final_scores_calc_ended = calculate_final_scores(scores, racks, bag) # Renamed
        state['final_scores'] = final_scores_calc_ended 
        state['game_over_state'] = game_over_state_ended
        state['reason'] = reason_for_ending_local

        if is_batch_running:
            # print(f"Debug: Batch game {current_batch_game_num} ended. Reason: {reason_for_ending_local}") # Keep if helpful
            batch_prefix_chgo_ended = "" # Renamed
            if isinstance(initial_game_config, dict): 
                batch_prefix_chgo_ended = initial_game_config.get('batch_filename_prefix', 'UNKNOWN-BATCH')
            else: 
                print("Warning: initial_game_config is not a dict in check_and_handle_game_over")
                initial_game_config = {} # Ensure it's a dict for .get()
                batch_prefix_chgo_ended = 'ERROR-BATCH-CONFIG'

            individual_gcg_filename_chgo_ended = f"{batch_prefix_chgo_ended}-GAME-{current_batch_game_num}.gcg" # Renamed
            
            gcg_initial_racks_to_save_chgo_ended = current_game_initial_racks # Renamed
            if not gcg_initial_racks_to_save_chgo_ended or len(gcg_initial_racks_to_save_chgo_ended) != 2:
                 gcg_initial_racks_to_save_chgo_ended = state.get('initial_racks_sgs', [[],[]]) 

            individual_gcg_filename_chgo_ended = "GCG_SAVE_SKIPPED" # Placeholder

            game_stats = collect_game_stats(current_batch_game_num, player_names, final_scores_calc_ended, move_history, individual_gcg_filename_chgo_ended)
            
            batch_results.append(game_stats) # Append to global list
            state['batch_results'] = list(batch_results) # Update state's copy/reference
        
        elif practice_mode and not practice_solved: 
            state['showing_practice_end_dialog'] = True
            state['practice_end_message'] = f"Practice ended.\n{reason_for_ending_local}"

    return state 





# In Scrabble Game.py

def handle_turn_start_updates(state):
    """Handles logic that needs to run at the very start of each turn,
       before AI or event processing for that turn."""
    
    # MODIFICATION START: Access globals directly instead of from state dict
    global gaddag_loading_status, GADDAG_STRUCTURE, DAWG
    # MODIFICATION END

    turn = state['turn']
    previous_turn = state.get('previous_turn', 0) # Get from state with default
    # gaddag_loading_status from global will be used
    # GADDAG_STRUCTURE from global will be used
    # DAWG from global will be used
    is_ai = state['is_ai']
    racks = state['racks']
    tiles = state['tiles'] # Current board tiles
    board = state['board'] # Board premium square layout
    blanks = state['blanks'] # Set of blank coords
    is_batch_running = state.get('is_batch_running', False)
    
    # These might be modified and need to be packed back into state
    all_moves = state.get('all_moves', []) 
    human_played = state.get('human_played', False)
    power_tile_message_shown = state.get('power_tile_message_shown', False)
    bingo_practice_message_shown = state.get('bingo_practice_message_shown', False)
    practice_mode = state.get('practice_mode')
    paused_for_power_tile = state.get('paused_for_power_tile', False)
    paused_for_bingo_practice = state.get('paused_for_bingo_practice', False)


    if turn != previous_turn: # New turn has started
        player_idx = turn - 1
        is_current_player_ai = is_ai[player_idx] if 0 <= player_idx < len(is_ai) else False

        # Generate all moves if it's AI's turn and not batch visualizing, or if it's human's turn for hints
        # Only generate if GADDAG/DAWG are ready.
        if gaddag_loading_status == 'loaded' and GADDAG_STRUCTURE is not None and DAWG is not None:
            if (is_current_player_ai and not (is_batch_running and not state.get('initial_game_config',{}).get('visualize_batch'))) or \
               (not is_current_player_ai): # For human player, always generate for potential hints
                # print(f"Debug: Turn {turn} start, generating all_moves for rack: {racks[player_idx]}")
                # Ensure racks[player_idx] is valid before passing
                current_player_rack_for_gen = []
                if racks and len(racks) > player_idx and racks[player_idx] is not None:
                    current_player_rack_for_gen = racks[player_idx][:]
                
                all_moves = generate_all_moves_gaddag_cython(
                    current_player_rack_for_gen, tiles, board, blanks, GADDAG_STRUCTURE.root, DAWG
                )
                if all_moves is None: 
                    all_moves = []
                # print(f"Debug: Turn {turn} start, generated {len(all_moves)} moves.")
        else:
            all_moves = [] # Cannot generate moves if AI data not ready
            # if not is_batch_running:
                # print(f"Debug: Turn {turn} start, AI data not ready, all_moves set to []. Status: {gaddag_loading_status}")
        
        if not is_batch_running: # Only print player turn message for interactive modes
            player_name_display = state['player_names'][player_idx] if state['player_names'][player_idx] else f"Player {player_idx + 1}"
            rack_display_str = ""
            if racks and len(racks) > player_idx and racks[player_idx] is not None:
                 rack_display_str = "".join(sorted(racks[player_idx]))
            else:
                 rack_display_str = "N/A"
            print(f"\n{player_name_display} turn started. Rack: {rack_display_str}")

        previous_turn = turn
        human_played = False # Reset for the new turn
        # Reset practice message flags for the new turn
        power_tile_message_shown = False
        bingo_practice_message_shown = False
        # Reset practice pause flags IF it's not the AI's turn that was paused
        # (AI might need to resume from a paused state if it's its turn again)
        if not is_current_player_ai:
            if practice_mode == "power_tiles": paused_for_power_tile = False
            if practice_mode == "bingo_bango_bongo": paused_for_bingo_practice = False
    
    # Update state with potentially changed values
    state['previous_turn'] = previous_turn
    state['all_moves'] = all_moves
    state['human_played'] = human_played
    state['power_tile_message_shown'] = power_tile_message_shown
    state['bingo_practice_message_shown'] = bingo_practice_message_shown
    state['paused_for_power_tile'] = paused_for_power_tile
    state['paused_for_bingo_practice'] = paused_for_bingo_practice
    
    return state

    







# In Scrabble Game.py
def handle_ai_turn_trigger(state):
    """Checks if it's AI's turn and triggers ai_turn if conditions are met."""
    global GADDAG_STRUCTURE, DAWG, gaddag_loading_status, gaddag_loaded_event 

    game_over_state = state['game_over_state']
    replay_mode = state['replay_mode']
    is_batch_running = state['is_batch_running']
    paused_for_power_tile = state['paused_for_power_tile'] 
    paused_for_bingo_practice = state['paused_for_bingo_practice']
    exchanging = state['exchanging']
    hinting = state['hinting']
    showing_all_words = state['showing_all_words']
    specifying_rack = state['specifying_rack']
    showing_simulation_config = state['showing_simulation_config']
    turn = state['turn'] 
    is_ai_flags = state['is_ai'] 
    human_played = state['human_played']
    is_solving_endgame_flag = state['is_solving_endgame'] 
    live_racks = state['racks']
    live_tiles = state['tiles']
    live_board = state['board']
    live_blanks = state['blanks'] 
    live_scores = state['scores']
    live_bag = state['bag']
    live_board_tile_counts = state['board_tile_counts'] 
    live_move_history = state['move_history']
    current_first_play = state['first_play']
    current_pass_count = state['pass_count']
    current_exchange_count = state['exchange_count']
    current_consecutive_zero = state['consecutive_zero_point_turns']
    current_blanks_played_count = state.get('blanks_played_count', 0) 
    player_names_for_ai = state['player_names']
    dropdown_open_for_ai = state['dropdown_open']
    hinting_for_ai = state['hinting'] 
    showing_all_words_for_ai = state['showing_all_words'] 
    letter_checks_for_ai = state['letter_checks']
    number_checks_for_ai = state['number_checks'] 
    sgs_initial_racks = state.get('initial_racks_sgs', [[],[]])
    sgs_initial_bag_order = state.get('initial_shuffled_bag_order_sgs', [])
    current_last_played_highlight_coords = state.get('last_played_highlight_coords', set())
    use_endgame_solver = state.get('USE_ENDGAME_SOLVER', False)
    use_ai_simulation = state.get('USE_AI_SIMULATION', False)
    current_practice_mode = state.get('practice_mode')
    current_practice_best_move = state.get('practice_best_move')
    current_practice_target_moves = state.get('practice_target_moves', [])
    current_practice_prob_max_idx = state.get('practice_probability_max_index')
    # Use the gaddag_loading_status from the passed state, which main() updates from global each frame
    current_gaddag_loading_status_in_state_ai = state.get('gaddag_loading_status', 'idle')


    is_ai_player_turn = 0 <= turn - 1 < len(is_ai_flags) and is_ai_flags[turn - 1]
    not_paused_for_human_ui = not (exchanging or hinting or showing_all_words or specifying_rack or showing_simulation_config or dropdown_open_for_ai)
    game_is_active = not game_over_state and not replay_mode
    ai_not_already_solving = not is_solving_endgame_flag

    if game_is_active and is_ai_player_turn and not human_played and not_paused_for_human_ui and ai_not_already_solving:
        # MODIFICATION START: Refined check using the event, state's status, and global data structures
        event_is_set_ai = gaddag_loaded_event is not None and gaddag_loaded_event.is_set()
        # current_gaddag_loading_status_in_state_ai is from state, updated by main() from global gaddag_loading_status
        status_is_loaded_ai = (current_gaddag_loading_status_in_state_ai == 'loaded')
        gaddag_structure_is_ready_ai = GADDAG_STRUCTURE is not None # Check global directly
        dawg_is_ready_global_ai = DAWG is not None # Check global directly

        data_truly_ready_for_ai = event_is_set_ai and status_is_loaded_ai and gaddag_structure_is_ready_ai and dawg_is_ready_global_ai
        
        if not data_truly_ready_for_ai:
        # MODIFICATION END
            # if not is_batch_running: # Optional debug print
            #    print(f"AI {turn} waiting: Event set: {event_is_set_ai}, Status in State: {current_gaddag_loading_status_in_state_ai}, GADDAG_STRUCTURE: {'Set' if GADDAG_STRUCTURE else 'None'}, DAWG: {'Set' if DAWG else 'None'}")
            return state # AI waits, game loop continues
        
        ai_result_tuple = ai_turn(
            live_racks[turn-1][:], live_tiles, live_blanks, live_board, live_scores, live_racks,
            turn, live_bag, current_first_play, current_pass_count, current_exchange_count,
            current_consecutive_zero, player_names_for_ai, dropdown_open_for_ai,
            hinting_for_ai, showing_all_words_for_ai, letter_checks_for_ai, number_checks_for_ai,
            live_move_history, 
            sgs_initial_racks, 
            sgs_initial_bag_order, 
            live_board_tile_counts, 
            current_blanks_played_count, 
            current_last_played_highlight_coords, 
            is_batch_running, use_endgame_solver, use_ai_simulation,
            current_practice_mode, current_practice_best_move, current_practice_target_moves,
            paused_for_power_tile, current_power_tile, paused_for_bingo_practice,
            current_practice_prob_max_idx
        )

        action, move_details, exchange_tiles, move_data_sgs, \
        updated_first_play, updated_pass_count, updated_exchange_count, updated_consecutive_zero, \
        updated_highlight_coords, updated_board_counts, updated_blanks_played = ai_result_tuple

        state['first_play'] = updated_first_play
        state['pass_count'] = updated_pass_count
        state['exchange_count'] = updated_exchange_count
        state['consecutive_zero_point_turns'] = updated_consecutive_zero
        state['last_played_highlight_coords'] = updated_highlight_coords
        state['board_tile_counts'] = updated_board_counts 
        state['blanks_played_count'] = updated_blanks_played 

        if action == "pause_power_tile":
            state['paused_for_power_tile'] = True
            state['current_power_tile'] = move_details 
        elif action == "pause_bingo_practice":
            state['paused_for_bingo_practice'] = True
        elif move_data_sgs: 
            live_move_history.append(move_data_sgs) 
            state['turn'] = move_data_sgs['next_turn_after_move'] 
    return state 


    





# Function to Replace: handle_deferred_practice_init
# REASON: Call generate_all_moves_gaddag_cython instead of deleted Python version.

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
    # --- Access global DAWG ---
    global DAWG

    # Unpack necessary variables
    practice_mode = state['practice_mode']
    practice_target_moves = state['practice_target_moves']
    racks = state['racks']
    tiles = state['tiles']
    board = state['board']
    blanks = state['blanks']
    # Variables that might be modified
    practice_best_move = state['practice_best_move']
    all_moves = state['all_moves']
    running_inner = state['running_inner']

    if practice_mode == "eight_letter" and not practice_target_moves and gaddag_loading_status == 'loaded':
        print(f"--- Main Loop: Conditions met for 8-letter practice move gen. GADDAG_STRUCTURE is None? {GADDAG_STRUCTURE is None} ---")

        if GADDAG_STRUCTURE is not None and DAWG is not None and racks and racks[0] is not None: # Check DAWG too
             print(f"  DEBUG: Calling generate_all_moves_gaddag_cython for rack: {''.join(sorted(racks[0]))}")
             print(f"  DEBUG: *** ENTERED GENERATION BLOCK ***")
             # --- MODIFICATION: Call Cython function ---
             try:
                 generated_moves = generate_all_moves_gaddag_cython(
                     racks[0], tiles, board, blanks, GADDAG_STRUCTURE.root, DAWG
                 )
             except Exception as e_gen:
                  print(f"ERROR during deferred move generation: {e_gen}")
                  import traceback
                  traceback.print_exc()
                  generated_moves = None
             # --- END MODIFICATION ---

             if generated_moves is None:
                 print("  DEBUG: generate_all_moves_gaddag_cython returned None!")
                 generated_moves = []
             else:
                 print(f"  DEBUG: generate_all_moves_gaddag_cython returned {len(generated_moves)} moves.")

             if generated_moves:
                 state['practice_target_moves'] = generated_moves
                 state['practice_best_move'] = generated_moves[0] # Assumes sorted by score desc
                 state['all_moves'] = generated_moves # Also store in all_moves for consistency
                 print(f"  DEBUG: Successfully set practice_target_moves. Length is now: {len(state['practice_target_moves'])}. Best move: {state['practice_best_move']['word']} ({state['practice_best_move']['score']} pts)")
             else:
                 print("Error: No valid moves found for 8-letter practice setup (deferred gen)!")
                 show_message_dialog("Error: No possible moves found for this setup.", "Practice Error")
                 state['running_inner'] = False
        elif GADDAG_STRUCTURE is None or DAWG is None: # Check DAWG too
            reason = "GADDAG structure is missing" if GADDAG_STRUCTURE is None else "DAWG object is missing"
            print(f"Error: Cannot generate practice moves, {reason} (status was 'loaded').")
            show_message_dialog(f"Error: AI data ({reason}) missing.", "Error")
            state['running_inner'] = False
        else:
            print("Error: Invalid rack state for Player 1 in 8-letter practice (deferred gen).")
            state['running_inner'] = False
    elif practice_mode == "eight_letter" and not practice_target_moves and gaddag_loading_status == 'loading':
         print("--- Main Loop: Waiting for GADDAG to load for 8-letter practice... ---")


    return state # Return the modified state dictionary








def handle_practice_restart(state):
    """
    Handles the logic for restarting the 8-Letter Bingo practice mode
    using the previously selected probability setting.
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
    # --- ADDED: Get stored probability index ---
    practice_probability_max_index = state.get('practice_probability_max_index')
    # --- END ADDED ---

    # <<< --- ADDED DEBUG PRINT --- >>>
    #print(f"--- DEBUG handle_practice_restart: Entered. restart_flag={restart_practice_mode}, practice_probability_max_index from state = {practice_probability_max_index} (Type: {type(practice_probability_max_index)}) ---")
    # <<< --- END ADDED DEBUG PRINT --- >>>


    if restart_practice_mode:
        print("--- Restarting 8-Letter Bingo Practice ---")
        # --- MODIFICATION: Perform setup directly ---
        # REMOVED call to eight_letter_practice()

        # Ensure probability index is valid
        if practice_probability_max_index is None:
            print("Error: Cannot restart 8-Letter practice, probability setting not found.") # This is the error message being triggered
            state['running_inner'] = False
            state['return_to_mode_selection'] = True
            state['restart_practice_mode'] = False # Reset flag
            return state # Return early

        # Load word lists (consider loading once globally or passing if performance is critical)
        try:
            with open("7-letter-list.txt", "r") as seven_file, open("8-letter-list.txt", "r") as eight_file:
                seven_letter_words = [line.strip().upper() for line in seven_file.readlines()]
                eight_letter_words = [line.strip().upper() for line in eight_file.readlines()]
        except FileNotFoundError:
            print("Error: Could not find 7/8-letter lists for practice restart.")
            state['running_inner'] = False
            state['return_to_mode_selection'] = True
            state['restart_practice_mode'] = False # Reset flag
            return state # Return early
        if not seven_letter_words or not eight_letter_words:
            print("Error: Word list files are empty for practice restart.")
            state['running_inner'] = False
            state['return_to_mode_selection'] = True
            state['restart_practice_mode'] = False # Reset flag
            return state # Return early

        # Select new words using stored max_index
        max_index = practice_probability_max_index # Use stored value
        if not (1 <= max_index <= len(eight_letter_words)): # Validate stored index
             print(f"Warning: Invalid stored max_index ({max_index}). Using full list.")
             max_index = len(eight_letter_words)

        selected_eight = random.choice(eight_letter_words[:max_index])
        print("Selected 8-letter word:", selected_eight)
        remove_idx = random.randint(0, 7)
        removed_letter = selected_eight[remove_idx]
        removed_eight = selected_eight[:remove_idx] + selected_eight[remove_idx + 1:]
        print("Player 1 rack (7 letters):", removed_eight)
        print("Removed letter:", removed_letter)
        selected_seven = select_seven_letter_word(removed_letter, seven_letter_words)
        if selected_seven is None:
            print("Error: Could not find a suitable 7-letter word for practice restart.")
            state['running_inner'] = False
            state['return_to_mode_selection'] = True
            state['restart_practice_mode'] = False # Reset flag
            return state # Return early
        print("Selected 7-letter word for board:", selected_seven)

        # Create new board and place 7-letter word
        p_board, _, p_tiles = create_board()
        center_r, center_c = CENTER_SQUARE
        word_len = len(selected_seven)
        start_offset = word_len // 2
        place_horizontally = random.choice([True, False])
        placement_successful = False
        if place_horizontally:
            start_c_place = center_c - start_offset
            if 0 <= start_c_place and start_c_place + word_len <= GRID_SIZE:
                for i, letter in enumerate(selected_seven): p_tiles[center_r][start_c_place + i] = letter
                placement_successful = True
                print(f"Placed '{selected_seven}' horizontally at ({center_r},{start_c_place})")
        if not placement_successful: # Try vertically
            start_r_place = center_r - start_offset
            if 0 <= start_r_place and start_r_place + word_len <= GRID_SIZE:
                for i, letter in enumerate(selected_seven): p_tiles[start_r_place + i][center_c] = letter
                placement_successful = True
                print(f"Placed '{selected_seven}' vertically at ({start_r_place},{center_c})")
        if not placement_successful:
            print("Error: Could not place 7-letter word for practice restart.")
            state['running_inner'] = False
            state['return_to_mode_selection'] = True
            state['restart_practice_mode'] = False # Reset flag
            return state # Return early

        # Set up racks and other state
        p_racks = [[], []]
        p_racks[0] = sorted(list(removed_eight))
        p_racks[1] = []
        p_blanks = set()
        p_bag = [] # No bag in this mode

        # Update game state variables directly
        state['board'] = p_board
        state['tiles'] = p_tiles
        state['racks'] = p_racks
        state['blanks'] = p_blanks
        state['bag'] = p_bag
        state['scores'] = [0, 0]
        state['turn'] = 1
        state['first_play'] = False # 8-letter practice never starts on first play
        state['board_tile_counts'] = Counter(c for row in p_tiles for c in row if c) # Recalculate board counts

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

        print("--- 8-Letter Bingo Practice Restarted Successfully (using stored probability) ---")
        # --- END MODIFICATION ---

        # Reset the flag after handling
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






# In Scrabble Game.py
def update_preview_score(state):
    """Calculates and updates the preview score in the state if conditions are met."""
    typing = state.get('typing', False)
    preview_score_enabled = state.get('preview_score_enabled', False)
    word_positions = state.get('word_positions', []) 
    board = state.get('board') 
    tiles = state.get('tiles') 
    blanks = state.get('blanks') 

    preview_score = 0 
    if typing and preview_score_enabled and word_positions:
        blanks_in_current_typed_word = set()
        if blanks: 
            for r_wp, c_wp, _ in word_positions:
                if (r_wp, c_wp) in blanks:
                    blanks_in_current_typed_word.add((r_wp, c_wp))
        
        try:
            if board is not None and tiles is not None and blanks_in_current_typed_word is not None:
                preview_score = calculate_score_cython(word_positions, board, tiles, blanks_in_current_typed_word)
            else:
                # print("DEBUG update_preview_score: board, tiles, or blanks_in_current_typed_word is None. Cannot calculate preview.") # Optional
                preview_score = 0 
        except Exception as e_calc_score:
            print(f"Error in update_preview_score calling calculate_score_cython: {e_calc_score}")
            preview_score = 0 
    
    state['current_preview_score'] = preview_score
    
    # MODIFICATION START: Remove the verbose debug print
    # print(f"DEBUG update_preview_score: Returning, type(state) = {type(state)}")
    # if not isinstance(state, dict):
    #     print(f"DEBUG update_preview_score: state IS NOT A DICT before return. Value: {state}")
    # MODIFICATION END
    
    return state






def reset_per_game_variables():
    """Returns a dictionary with default values for UI/temporary state variables."""
    # Access constants needed for initialization
    # (Ensure these are accessible, e.g., defined globally)
    # WINDOW_WIDTH, WINDOW_HEIGHT, DIALOG_WIDTH, DIALOG_HEIGHT,
    # DEFAULT_AI_CANDIDATES, DEFAULT_OPPONENT_SIMULATIONS, DEFAULT_POST_SIM_CANDIDATES
    reset_values = {
        'running_inner': True, # Start the inner loop
        'game_over_state': False, # Reset game over flag
        # 'return_to_mode_selection': False, # Keep this for explicit user actions in main
        'dragged_tile': None,
        'drag_pos': None,
        'drag_offset': (0, 0),
        'dragging': False, # For dialog dragging
        'selected_square': None,
        'typing': False,
        'word_positions': [],
        'original_tiles': None, # Store board state when typing starts
        'original_rack': None,  # Store rack state when typing starts
        'typing_start': None,   # Store (r, c) where typing initiated
        'typing_direction': None, # Store 'right' or 'down'
        'exchanging': False,
        'selected_tiles': set(),
        'hinting': False,
        'hint_moves': [],
        'selected_hint_index': None,
        'showing_all_words': False,
        'all_words_scroll_offset': 0,
        'dropdown_open': False,
        'last_left_click_time': 0,
        'last_left_click_pos': None,
        'last_played_highlight_coords': set(),
        # 'action': None, # Consider removing if truly redundant later
        'scroll_offset': 0, # For scoreboard
        # 'scoreboard_height': WINDOW_HEIGHT - 80, # Recalculate based on constant
        'showing_stats': False,
        'stats_dialog_x': (WINDOW_WIDTH - 480) // 2, # Use stats dialog width constant
        'stats_dialog_y': (WINDOW_HEIGHT - 600) // 2, # Use stats dialog height constant
        'stats_scroll_offset': 0,
        'stats_dialog_dragging': False,
        'stats_dialog_drag_offset': (0, 0),
        'dialog_x': (WINDOW_WIDTH - DIALOG_WIDTH) // 2, # For game over dialog
        'dialog_y': (WINDOW_HEIGHT - DIALOG_HEIGHT) // 2,
        'reason': '', # Game over reason
        'showing_practice_end_dialog': False,
        'practice_end_message': "",
        'practice_solved': False,
        'paused_for_power_tile': False,
        'current_power_tile': None,
        'bingo_practice_message_shown': False, # Should be reset per game
        'power_tile_message_shown': False,    # Should be reset per game
        'paused_for_bingo_practice': False,
        'showing_simulation_config': False,
        'simulation_config_inputs': [str(DEFAULT_AI_CANDIDATES), str(DEFAULT_OPPONENT_SIMULATIONS), str(DEFAULT_POST_SIM_CANDIDATES)],
        'simulation_config_active_input': None,
        'specifying_rack': False,
        'specify_rack_inputs': ["", ""],
        'specify_rack_active_input': None,
        'specify_rack_original_racks': [[], []],
        'specify_rack_proposed_racks': [[], []],
        'confirming_override': False,
        'preview_score_enabled': False,
        'current_preview_score': 0,
        'drawn_rects': {}, # Store rects drawn by draw_game_screen
        'best_exchange_for_hint': None,
        'best_exchange_score_for_hint': -float('inf'),
        'restart_practice_mode': False, # <<< ENSURE THIS IS PRESENT
        'running_main_loop': True, # Flag possibly set by process_events to signal quit
        'stats_total_content_height': 0 # For stats dialog scrolling calculation
    }
    return reset_values






def reset_for_play_again(is_ai, practice_mode):
        """
        Resets the core game state variables for starting a new single game ("Play Again").
        Also returns the current global GADDAG loading status.
        Initializes board_tile_counts.
        Removes current_turn_pool_quality_score.

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
        blanks_played_count = 0
        bag = create_standard_bag()
        random.shuffle(bag)
        racks = [[], []]
        board_tile_counts = Counter() # Initialize counter
        # REMOVED current_turn_pool_quality_score initialization

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
        # REMOVED current_turn_pool_quality_score from return tuple
        return (board, tiles, scores, blanks, bag, racks, initial_racks,
                current_game_initial_racks, first_play, turn, replay_mode,
                move_history, pass_count, exchange_count,
                consecutive_zero_point_turns, last_played_highlight_coords,
                is_solving_endgame, practice_target_moves,
                practice_best_move, all_moves, gaddag_loading_status,
                board_tile_counts) # Added counter






# Function to Replace: run_game_loop
# REASON: Simplify to manage outer loop and handle final exit.

def run_game_loop():
    """Runs the main game loop, handling restarts and final exit."""
    global main_called # Need to access the global flag

    print("--- Script execution started (run_game_loop) ---")
    main_called = False # Initialize ONCE globally before the loop
    running = True
    while running: # Loop for "Play Again" / Return to Main Menu
        # Pass the current initialization status to main
        # main() now handles initialization, game loop, and profiling internally
        # It returns True if the user wants to go back to the menu (restart loop)
        # It returns False if the user wants to quit the application
        should_restart = main(main_called)

        if should_restart:
            # If main returns True, it means "Play Again" or "Main Menu" was selected.
            # Reset main_called to False ONLY if returning to menu/restarting.
            # This forces the mode selection screen to show again.
            main_called = False
            print("--- Restarting main loop for new game/mode selection ---")
        else:
            # If main returns False, it means Quit was selected.
            running = False # Exit the outer loop

    # --- Final Exit Logic ---
    print("--- Script exiting (run_game_loop) ---")
    pygame.quit()
    sys.exit()







# In Scrabble Game.py
def main(is_initialized): 
    global pygame, screen, clock, font, ui_font, button_font, tile_count_font, dialog_font
    global board, tiles, racks, blanks, scores, turn, first_play, game_mode, is_ai, player_names, bag, move_history
    global replay_mode, current_replay_turn, pass_count, exchange_count, consecutive_zero_point_turns
    global gaddag_loading_status, GADDAG_STRUCTURE, DAWG, gaddag_load_thread, gaddag_loaded_event
    global last_word, last_score, last_start, last_direction, is_loaded_game, final_scores, game_over_state
    global DEVELOPER_PROFILE_ENABLED, profiler, USE_ENDGAME_SOLVER, USE_AI_SIMULATION, practice_mode
    global is_batch_running, total_batch_games, current_batch_game_num, batch_results, initial_game_config
    global practice_target_moves, practice_best_move, practice_solved, practice_end_message
    global all_moves, letter_checks, number_checks, board_tile_counts, blanks_played_count
    global initial_shuffled_bag_order_sgs, initial_racks_sgs, replay_initial_shuffled_bag
    global is_solving_endgame, endgame_start_time
    global human_player, practice_probability_max_index 
    global restart_practice_mode, return_to_mode_selection 
    
    main_called_flag = is_initialized 

    if not pygame.get_init(): 
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Scrabble Game")
        font = pygame.font.SysFont("Arial", FONT_SIZE, bold=True)
        ui_font = pygame.font.SysFont("Arial", 20)
        button_font = pygame.font.SysFont("Arial", 16)
        tile_count_font = pygame.font.SysFont("Arial", 14)
        dialog_font = pygame.font.SysFont("Arial", 24)

    if clock is None: 
         clock = pygame.time.Clock()

    if not main_called_flag:
        # print("--- DEBUG main: Calling mode_selection_screen ---") # Keep this one if helpful
        selected_mode_result, return_data = mode_selection_screen()
        if selected_mode_result is None: 
            if DEVELOPER_PROFILE_ENABLED and profiler:
                profiler.disable()
                profile_filename = f"scrabble_profile_MODE_QUIT_{datetime.datetime.now():%Y%m%d%H%M%S}.prof"
                profiler.dump_stats(profile_filename)
                print(f"Profiling data saved to {profile_filename}")
            return False 

        # print(f"--- DEBUG main: Calling initialize_game with mode: {selected_mode_result} ---") # Keep if helpful
        init_success = initialize_game(selected_mode_result, return_data, main_called_flag)
        if not init_success:
            print("ERROR: Game initialization failed. Exiting or returning to menu.")
            if DEVELOPER_PROFILE_ENABLED and profiler:
                profiler.disable()
                profile_filename = f"scrabble_profile_INIT_FAIL_{datetime.datetime.now():%Y%m%d%H%M%S}.prof"
                profiler.dump_stats(profile_filename)
                print(f"Profiling data saved to {profile_filename}")
            return True 
        main_called_flag = True 
    
    if DEVELOPER_PROFILE_ENABLED and profiler is None: 
        profiler = cProfile.Profile()
        profiler.enable()
    elif not DEVELOPER_PROFILE_ENABLED and profiler is not None: 
        profiler.disable()
        profiler = None 

    batch_stop_requested_main_loc = False 
    num_loops_main_loc = total_batch_games if is_batch_running else 1 
    current_game_initial_racks_sgs_ref_main_loc = initial_racks_sgs 
    was_batch_running_at_start_main_loc = is_batch_running 

    for game_num_main_loc in range(1, num_loops_main_loc + 1): 
        if is_batch_running:
            current_batch_game_num = game_num_main_loc
            if game_num_main_loc > 1: 
                # print(f"\n--- Starting Batch Game {game_num_main_loc} / {total_batch_games} ---") # Keep if helpful
                reset_result = reset_game_state(initial_game_config)
                if reset_result is None:
                    print(f"ERROR: Failed to reset game state for batch game {game_num_main_loc}. Stopping batch.")
                    batch_stop_requested_main_loc = True; break
                board, tiles, racks, blanks, scores, turn, first_play, game_mode, is_ai, \
                player_names, bag, move_history, replay_mode, current_replay_turn, \
                pass_count, exchange_count, consecutive_zero_point_turns, \
                board_tile_counts, blanks_played_count, \
                initial_shuffled_bag_order_sgs, initial_racks_sgs_local_batch_loop = reset_result 
                initial_racks_sgs = initial_racks_sgs_local_batch_loop 
                current_game_initial_racks_sgs_ref_main_loc = initial_racks_sgs 
            # else: 
                # print(f"--- Starting Batch Game 1 / {total_batch_games} ---") # Keep if helpful
            
            if not initial_game_config.get('visualize_batch', False): 
                pass 

        current_state = {
            'board': board, 'tiles': tiles, 'racks': racks, 'blanks': blanks, 'scores': scores, 'turn': turn,
            'first_play': first_play, 'game_mode': game_mode, 'is_ai': is_ai, 'player_names': player_names,
            'bag': bag, 'bag_count': len(bag), 'move_history': move_history, 'replay_mode': replay_mode,
            'current_replay_turn': current_replay_turn, 'pass_count': pass_count,
            'exchange_count': exchange_count, 'consecutive_zero_point_turns': consecutive_zero_point_turns,
            'gaddag_loading_status': gaddag_loading_status, 
            'last_word': last_word, 'last_score': last_score, 'last_start': last_start,
            'last_direction': last_direction, 'is_loaded_game': is_loaded_game,
            'final_scores': final_scores, 'game_over_state': game_over_state,
            'practice_mode': practice_mode, 'is_batch_running': is_batch_running,
            'total_batch_games': total_batch_games, 'current_batch_game_num': current_batch_game_num,
            'initial_game_config': initial_game_config, 'batch_results': batch_results, 
            'practice_target_moves': practice_target_moves, 'practice_best_move': practice_best_move,
            'practice_solved': practice_solved, 'practice_end_message': practice_end_message,
            'all_moves': all_moves, 'letter_checks': letter_checks, 'number_checks': number_checks,
            'board_tile_counts': board_tile_counts, 'blanks_played_count': blanks_played_count,
            'initial_shuffled_bag_order_sgs': initial_shuffled_bag_order_sgs,
            'initial_racks_sgs': current_game_initial_racks_sgs_ref_main_loc, 
            'replay_initial_shuffled_bag': replay_initial_shuffled_bag,
            'is_solving_endgame': is_solving_endgame, 'endgame_start_time': endgame_start_time,
            'human_player': human_player, 
            'USE_ENDGAME_SOLVER': USE_ENDGAME_SOLVER, 'USE_AI_SIMULATION': USE_AI_SIMULATION,
            'dragged_tile': None, 'drag_pos': None, 'drag_offset': (0,0),
            'selected_square': None, 'typing': False, 'word_positions': [],
            'original_tiles': None, 'original_rack': None, 'original_blanks_set_at_typing_start': None,
            'current_r': None, 'current_c': None, 'typing_start': None, 'typing_direction': None,
            'current_move_blanks_coords': set(),
            'dropdown_open': False, 'exchanging': False, 'selected_tiles': set(),
            'hinting': False, 'hint_moves': [], 'selected_hint_index': None,
            'showing_all_words': False, 'all_words_scroll_offset': 0,
            'specifying_rack': False, 'specify_rack_inputs': ["",""], 'specify_rack_active_input': None,
            'specify_rack_original_racks': [[],[]], 'specify_rack_proposed_racks': [[],[]], 'confirming_override': False,
            'showing_simulation_config': False, 'simulation_config_inputs': ["","",""], 'simulation_config_active_input': None,
            'showing_practice_end_dialog': False,
            'dialog_x': (WINDOW_WIDTH - DIALOG_WIDTH) // 2, 'dialog_y': (WINDOW_HEIGHT - DIALOG_HEIGHT) // 2, 'reason': '',
            'showing_stats': False, 'stats_dialog_x': (WINDOW_WIDTH - 480) // 2, 'stats_dialog_y': (WINDOW_HEIGHT - 600) // 2,
            'stats_scroll_offset': 0, 'stats_dialog_dragging': False, 'stats_dialog_drag_offset': (0,0),
            'dragging': False, 'scroll_offset': 0,
            'last_left_click_time': 0, 'last_left_click_pos': None,
            'preview_score_enabled': False, 'current_preview_score': 0,
            'paused_for_power_tile': False, 'current_power_tile': None,
            'paused_for_bingo_practice': False,
            'power_tile_message_shown': False, 'bingo_practice_message_shown': False,
            'previous_turn': 0, 'human_played': False,
            'drawn_rects': {}, 'running_inner': True,
            'restart_practice_mode': restart_practice_mode, 
            'return_to_mode_selection': return_to_mode_selection, 
            'batch_stop_requested': batch_stop_requested_main_loc, 
            'practice_probability_max_index': practice_probability_max_index,
            'pyperclip_available': state_vars.pyperclip_available if 'state_vars' in locals() and hasattr(state_vars, 'pyperclip_available') else True, 
            'pyperclip': state_vars.pyperclip if 'state_vars' in locals() and hasattr(state_vars, 'pyperclip') else None,
            'stats_total_content_height': 0,
            'last_played_highlight_coords': last_played_highlight_coords 
        }
        
        current_state = handle_deferred_practice_init(current_state) 

        while current_state['running_inner']:
            current_state['gaddag_loading_status'] = gaddag_loading_status 
            
            current_state = handle_turn_start_updates(current_state)
            current_state = handle_ai_turn_trigger(current_state)

            should_process_events_and_draw_main_loop_inner_val_no_debug = True # Renamed
            if is_batch_running and not initial_game_config.get('visualize_batch', False):
                 should_process_events_and_draw_main_loop_inner_val_no_debug = False
            
            if should_process_events_and_draw_main_loop_inner_val_no_debug:
                current_state = process_game_events(current_state, current_state.get('drawn_rects', {}))
                if not current_state.get('running_inner', True): break  
                current_state = update_preview_score(current_state)
                current_state['drawn_rects'] = draw_game_screen(screen, current_state)
                current_state = handle_practice_messages(current_state)
                clock.tick(30) 

            current_state = check_and_handle_game_over(current_state)
            game_over_state_main_inner_loop_val_loc_no_debug = current_state['game_over_state'] # Renamed
            batch_stop_requested_main_inner_loop_val_loc_no_debug = current_state['batch_stop_requested'] # Renamed

            if game_over_state_main_inner_loop_val_loc_no_debug or batch_stop_requested_main_inner_loop_val_loc_no_debug :
                if is_batch_running and not batch_stop_requested_main_inner_loop_val_loc_no_debug: 
                    print(f"--- Batch Game {current_batch_game_num} Finished ---")
                break 
            
            current_state = handle_practice_restart(current_state)
            if current_state['restart_practice_mode']: 
                current_state = handle_deferred_practice_init(current_state)
                current_state['restart_practice_mode'] = False 
                continue 

        if not current_state.get('running_inner', True) and not current_state.get('return_to_mode_selection', False) and not is_batch_running: 
            if DEVELOPER_PROFILE_ENABLED and profiler: profiler.disable(); profiler.dump_stats(f"scrabble_profile_QUIT_{datetime.datetime.now():%Y%m%d%H%M%S}.prof")
            return False 
        if current_state.get('batch_stop_requested', False) and is_batch_running: 
            print("Batch processing stopped by user.")
            break 
        if current_state.get('return_to_mode_selection', False): 
            break 

    if was_batch_running_at_start_main_loc and batch_results: 
        batch_summary_filename_main_loc_val_loc_no_debug = f"{initial_game_config.get('batch_filename_prefix', 'UNKNOWN-BATCH')}.txt" # Renamed
        save_batch_statistics(batch_results, initial_game_config.get('player_names', ["P1","P2"]), batch_summary_filename_main_loc_val_loc_no_debug)
        print(f"Batch summary saved to {batch_summary_filename_main_loc_val_loc_no_debug}")

    if DEVELOPER_PROFILE_ENABLED and profiler:
        profiler.disable()
        profile_filename_main_loc_val_loc_no_debug = 'scrabble_profile.prof' # Renamed
        if was_batch_running_at_start_main_loc:
            profile_filename_main_loc_val_loc_no_debug = f"{initial_game_config.get('batch_filename_prefix', 'UNKNOWN-BATCH')}_profile.prof"
        else:
            profile_filename_main_loc_val_loc_no_debug = f"scrabble_profile_SINGLE_{datetime.datetime.now():%Y%m%d%H%M%S}.prof"
        profiler.dump_stats(profile_filename_main_loc_val_loc_no_debug)
        print(f"Profiling data saved to {profile_filename_main_loc_val_loc_no_debug}")
        profiler = None 

    if current_state.get('return_to_mode_selection', False):
        return True 
    else: 
        return False







# MODIFIED: Program Entry Point - Simplified
if __name__ == "__main__":
    # All profiling logic is now handled within main() and run_game_loop()
    run_game_loop()
    # The run_game_loop function handles the final pygame.quit() and sys.exit()
