#Scrabble 29APR25 Cython V5


# gaddag_cython.pyx
# cython: language_level=3

import cython # To use decorators like @cython.locals
import numpy as np # Keep NumPy import as it's used in _gaddag_traverse

# Import necessary Python modules/classes
from collections import Counter

# Import helpers and constants FROM scrabble_helpers
# *** REMOVED DAWG import ***
from scrabble_helpers import (
    CENTER_SQUARE, TILE_DISTRIBUTION,
    RED, PINK, BLUE, LIGHT_BLUE, LETTERS,
    get_coord, get_anchor_points, Gaddag, GaddagNode, # Ensure GaddagNode is imported
    DAWG as DAWG_cls # Import the CLASS definition for type checking if needed elsewhere
    )

from scrabble_helpers import perform_leave_lookup


# --- Define GRID_SIZE as a C constant ---
DEF GRID_SIZE_C = 15





##########################################################################################################
##########################################################################################################
##########################################################################################################








# --- MODIFIED FUNCTION ---
cpdef float evaluate_leave_cython(list rack, bint verbose=False): # Keep verbose argument
    """
    Generates the leave key and calls a Python helper function for the lookup.

    Args:
        rack (list): A list of characters representing the tiles left (blanks as ' ').
        verbose (bool): If True, print lookup details (optional).

    Returns:
        float: The score adjustment from the lookup table, or 0.0.
    """
    # --- REMOVED: Force verbose for debugging ---
    # verbose = True

    cdef int num_tiles = len(rack)
    cdef list rack_with_question_marks
    cdef str leave_key, tile
    cdef float leave_float = 0.0 # Initialize return value
    # --- REMOVED: key_count variable ---
    # cdef int key_count = 0

    # --- REMOVED Type check (can be added back if needed) ---
    # if not isinstance(leave_lookup_table_obj, dict): ...

    # --- REMOVED Key Iteration Check ---
    # if verbose: ...

    # --- REMOVED Direct Key Check ---
    # if verbose: ...

    if num_tiles == 0:
        # --- REMOVED Verbose Print ---
        # if verbose: print("--- Evaluating Leave (Cython): Empty rack -> 0.0")
        return 0.0 # Return float
    if num_tiles > 6:
        # --- REMOVED Verbose Print ---
        # if verbose: print(f"--- Evaluating Leave (Cython): Rack length {num_tiles} > 6 -> 0.0")
        return 0.0 # Return float

    # Create the sorted key for lookup
    rack_with_question_marks = ['?' if tile == ' ' else tile for tile in rack]
    leave_key = "".join(sorted(rack_with_question_marks))

    # --- REMOVED Verbose Print ---
    # if verbose: print(f"--- Evaluating Leave (Cython): Input rack: {rack}, Generated key: '{leave_key}'")

    try:
        # Call Python helper function from scrabble_helpers
        leave_float = perform_leave_lookup(leave_key) # Call imported function
        # --- REMOVED Verbose Print ---
        # if verbose: print(f"--- Evaluating Leave (Cython): Python lookup returned: {leave_float:.2f}")
        return leave_float

    except Exception as e:
        # Catch potential errors during the callback or key generation
        # Keep error print for actual errors
        print(f"Error (Cython) generating key or calling Python lookup for rack '{rack}': {e}")
        return 0.0 # Return float









##########################################################################################################
##########################################################################################################
##########################################################################################################






# Keep as cdef (internal helper)
cdef int get_char_index(str letter):
    cdef int index
    if letter == '?': # Use '?' internally for blank index
        index = 26
    elif 'A' <= letter <= 'Z': # Added check for valid letter range
        index = ord(letter) - ord('A')
    else:
        return -1 # Indicate error
    return index





# --- Helper functions for find_all_words_formed (Defined BEFORE find_all_words_formed) ---





##########################################################################################################
##########################################################################################################
##########################################################################################################







# --- find_cross_word ---
cdef list find_cross_word(tuple tile, list tiles, str main_orientation):
    """Finds a cross word formed by a single tile perpendicular to the main word."""
    cdef int r, c, min_row, max_row, min_col, max_col, rr_cw, cc_cw
    cdef list cross_word = []
    cdef object r_obj, c_obj, _ # Intermediate Python objects for unpacking

    r_obj, c_obj, _ = tile
    r = <int>r_obj
    c = <int>c_obj

    if main_orientation == "horizontal":
        min_row = r;
        while min_row > 0 and tiles[min_row - 1][c]: min_row -= 1
        max_row = r;
        while max_row < GRID_SIZE_C - 1 and tiles[max_row + 1][c]: max_row += 1
        if max_row > min_row:
            cross_word = []
            for rr_cw in range(min_row, max_row + 1):
                if tiles[rr_cw][c]:
                    cross_word.append((rr_cw, c, tiles[rr_cw][c]))
    elif main_orientation == "vertical":
        min_col = c;
        while min_col > 0 and tiles[r][min_col - 1]: min_col -= 1
        max_col = c;
        while max_col < GRID_SIZE_C - 1 and tiles[r][max_col + 1]: max_col += 1
        if max_col > min_col:
            cross_word = []
            for cc_cw in range(min_col, max_col + 1):
                if tiles[r][cc_cw]:
                    cross_word.append((r, cc_cw, tiles[r][cc_cw]))

    return cross_word if len(cross_word) > 1 else []






##########################################################################################################
##########################################################################################################
##########################################################################################################








# --- find_main_word ---
cdef tuple find_main_word(list new_tiles, list tiles):
    """Finds the primary word formed by newly placed tiles."""
    cdef int row, col, min_row, max_row, min_col, max_col, r_nt, c_nt, c_mw, r_mw
    cdef list main_word
    cdef str orientation
    cdef object r_nt_obj, c_nt_obj, _ign1_nt
    cdef set rows_set = set()
    cdef set cols_set = set()

    if not new_tiles: return [], None

    for tile_tuple in new_tiles: # Iterate through tuples
        if isinstance(tile_tuple, tuple) and len(tile_tuple) >= 3:
            r_nt_obj, c_nt_obj, _ign1_nt = tile_tuple[:3] # Unpack first 3
            r_nt = <int>r_nt_obj
            c_nt = <int>c_nt_obj
            rows_set.add(r_nt)
            cols_set.add(c_nt)
        else:
            pass # Handle error or skip invalid entry if necessary

    if len(rows_set) == 1:
        orientation = "horizontal"; row = rows_set.pop()
        min_col = 999; max_col = -1
        for tile_tuple in new_tiles: # Iterate again
            if isinstance(tile_tuple, tuple) and len(tile_tuple) >= 3:
                r_nt_obj, c_nt_obj, _ign1_nt = tile_tuple[:3]
                r_nt = <int>r_nt_obj; c_nt = <int>c_nt_obj
                if r_nt == row:
                    if c_nt < min_col: min_col = c_nt
                    if c_nt > max_col: max_col = c_nt
        while min_col > 0 and tiles[row][min_col - 1]: min_col -= 1
        while max_col < GRID_SIZE_C - 1 and tiles[row][max_col + 1]: max_col += 1
        main_word = []
        for c_mw in range(min_col, max_col + 1):
            if tiles[row][c_mw]:
                main_word.append((row, c_mw, tiles[row][c_mw]))
        return (main_word, orientation) if len(main_word) > 1 else ([], None)
    elif len(cols_set) == 1:
        orientation = "vertical"; col = cols_set.pop()
        min_row = 999; max_row = -1
        for tile_tuple in new_tiles: # Iterate again
            if isinstance(tile_tuple, tuple) and len(tile_tuple) >= 3:
                r_nt_obj, c_nt_obj, _ign1_nt = tile_tuple[:3]
                r_nt = <int>r_nt_obj; c_nt = <int>c_nt_obj
                if c_nt == col:
                    if r_nt < min_row: min_row = r_nt
                    if r_nt > max_row: max_row = r_nt
        while min_row > 0 and tiles[min_row - 1][col]: min_row -= 1
        while max_row < GRID_SIZE_C - 1 and tiles[max_row + 1][col]: max_row += 1
        main_word = []
        for r_mw in range(min_row, max_row + 1):
            if tiles[r_mw][col]:
                main_word.append((r_mw, col, tiles[r_mw][col]))
        return (main_word, orientation) if len(main_word) > 1 else ([], None)
    else: return [], None








##########################################################################################################
##########################################################################################################
##########################################################################################################







# --- find_all_words_formed ---
# Use cpdef (called from Python) - Defined BEFORE is_valid_play
cpdef list find_all_words_formed(list new_tiles, list tiles):
    """Finds all words (main and cross) formed by a play."""
    cdef list words = []
    cdef set new_positions_set = set()
    cdef list main_word_tiles, cross_word, unique_word_tile_lists
    cdef str orientation
    cdef tuple tile, signature
    cdef set seen_signatures
    cdef int r_np, c_np, r_sig, c_sig
    cdef str l_sig
    cdef list sig_list

    if not new_tiles: return words
    for r_np, c_np, _ in new_tiles:
        new_positions_set.add((r_np, c_np))

    main_word_tiles, orientation = find_main_word(new_tiles, tiles) # Call local cdef function

    if main_word_tiles:
        words.append(main_word_tiles)
        for tile in new_tiles: # tile is a tuple (r, c, letter)
            # Check if the tile position is one of the newly placed ones
            if (tile[0], tile[1]) in new_positions_set:
                cross_word = find_cross_word(tile, tiles, orientation) # Call local cdef function
                if cross_word:
                    words.append(cross_word)
    elif len(new_tiles) == 1: # Handle single tile plays forming only cross words
        tile = new_tiles[0]
        # Check horizontal cross word (pass "vertical" as main orientation)
        cross_h = find_cross_word(tile, tiles, "vertical");
        if cross_h: words.append(cross_h)
        # Check vertical cross word (pass "horizontal" as main orientation)
        cross_v = find_cross_word(tile, tiles, "horizontal");
        if cross_v: words.append(cross_v)

    # Deduplicate based on exact tile positions and letters
    unique_word_tile_lists = []; seen_signatures = set()
    for word_tile_list in words:
        sig_list = []
        for r_sig, c_sig, l_sig in word_tile_list:
            sig_list.append((r_sig, c_sig, l_sig))
        signature = tuple(sorted(sig_list))
        if signature not in seen_signatures:
            unique_word_tile_lists.append(word_tile_list)
            seen_signatures.add(signature)

    return unique_word_tile_lists





##########################################################################################################
##########################################################################################################
##########################################################################################################










# --- calculate_score ---
# Use cpdef (called from Python) - Defined BEFORE is_valid_play
cpdef int calculate_score(list new_tiles, list board, list tiles, set blanks):
    """Calculates the score for a play based on newly placed tiles."""
    cdef int total_score = 0, word_score, word_multiplier
    cdef int r, c, letter_value, letter_multiplier, r_np, c_np
    cdef bint is_blank
    cdef set new_positions = set()
    for r_np, c_np, _ in new_tiles:
        new_positions.add((r_np, c_np))
    cdef list words_formed_details, word_tiles
    cdef str letter
    cdef object square_color

    words_formed_details = find_all_words_formed(new_tiles, tiles) # Call local cpdef function

    for word_tiles in words_formed_details:
        word_score = 0; word_multiplier = 1
        for r, c, letter in word_tiles:
            if letter not in TILE_DISTRIBUTION:
                continue
            is_blank = (r, c) in blanks
            letter_value = 0 if is_blank else TILE_DISTRIBUTION[letter][1]; letter_multiplier = 1
            if (r, c) in new_positions:
                square_color = board[r][c]
                if square_color == LIGHT_BLUE: letter_multiplier = 2
                elif square_color == BLUE: letter_multiplier = 3
                elif square_color == PINK: word_multiplier *= 2
                elif square_color == RED: word_multiplier *= 3
            word_score += letter_value * letter_multiplier
        total_score += word_score * word_multiplier

    if len(new_tiles) == 7: total_score += 50
    return total_score





##########################################################################################################
##########################################################################################################
##########################################################################################################











# --- is_valid_play ---
# Use cpdef (called from Python)
cpdef tuple is_valid_play(list word_positions, list tiles, bint is_first_play, int initial_rack_size, list original_tiles, object rack, object dawg_obj):
    """Validate a potential play against game rules and dictionary."""
    # --- Start of indented block (4 spaces) ---
    cdef set newly_placed_positions_coords
    cdef list rows_list, cols_list, all_words_details, formed_word_strings
    cdef bint is_horizontal, is_vertical, connects, is_bingo
    cdef int r, c, dr, dc, nr, nc, r_np, c_np, r_wp, c_wp
    cdef int min_col, max_col, temp_min_col, temp_max_col
    cdef int min_row, max_row, temp_min_row, temp_max_row
    cdef int tiles_played_from_rack
    cdef str word, word_str
    cdef bint dawg_search_result
    cdef set rows_set_local, cols_set_local
    cdef list word_chars, word_detail
    # --- Declare tile_detail explicitly as object initially ---
    cdef object tile_detail

    # --- Added for loop replacement ---
    cdef bint center_square_played = False
    cdef int center_r, center_c

    if not word_positions:
        return False, False

    newly_placed_positions_coords = set()
    for r_np, c_np, _ in word_positions:
        newly_placed_positions_coords.add((r_np, c_np))
    if not newly_placed_positions_coords:
        return False, False

    rows_set_local = set()
    cols_set_local = set()
    for r_wp, c_wp, _ in word_positions:
        rows_set_local.add(r_wp)
        cols_set_local.add(c_wp)
    rows_list = sorted(list(rows_set_local))
    cols_list = sorted(list(cols_set_local))

    is_horizontal = len(rows_list) == 1; is_vertical = len(cols_list) == 1
    if not (is_horizontal or is_vertical):
        return False, False

    if is_horizontal:
        r = rows_list[0]; min_col = min(cols_list); max_col = max(cols_list)
        temp_min_col = min_col; temp_max_col = max_col
        while temp_min_col > 0 and tiles[r][temp_min_col - 1]: temp_min_col -= 1
        while temp_max_col < GRID_SIZE_C - 1 and tiles[r][temp_max_col + 1]: temp_max_col += 1
        for c in range(temp_min_col, temp_max_col + 1):
            if not tiles[r][c]:
                return False, False
    elif is_vertical:
        c = cols_list[0]; min_row = min(rows_list); max_row = max(rows_list)
        temp_min_row = min_row; temp_max_row = max_row
        while temp_min_row > 0 and tiles[temp_min_row - 1][c]: temp_min_row -= 1
        while temp_max_row < GRID_SIZE_C - 1 and tiles[temp_max_row + 1][c]: temp_max_row += 1
        for r in range(temp_min_row, temp_max_row + 1):
            if not tiles[r][c]:
                return False, False

    # --- Word Validity Check ---
    all_words_details = find_all_words_formed(word_positions, tiles) # Call local cpdef function

    # --- DEBUG ---
    # print(f"DEBUG is_valid_play: Type(all_words_details)={type(all_words_details)}")
    # print(f"DEBUG is_valid_play: all_words_details = {all_words_details}")
    # --- END DEBUG ---

    if not all_words_details and len(word_positions) > 1:
         return False, False

    formed_word_strings = []
    for word_detail in all_words_details: # all_words_details is the list of lists
        # --- DEBUG ---
        # print(f"DEBUG is_valid_play: Processing word_detail: {word_detail} (Type: {type(word_detail)})")
        # --- END DEBUG ---
        if not isinstance(word_detail, list):
            print(f"WARNING: Expected list in all_words_details, got {type(word_detail)}: {word_detail}")
            continue # Skip this entry

        word_chars = []
        # --- MODIFICATION: Explicitly handle tile_detail ---
        for item in word_detail: # Iterate through items in the list
            # Ensure item is a tuple of expected length before accessing index
            if isinstance(item, tuple) and len(item) >= 3:
                tile_detail = item # Assign to typed variable (optional here, but good practice)
                word_chars.append(tile_detail[2]) # Access element 2
            else:
                print(f"WARNING: Expected tuple of len>=3 in word_detail list, got {type(item)}: {item}")
                continue # Skip this malformed item
        # --- END MODIFICATION ---
        word_str = "".join(word_chars)
        if word_str: # Only add non-empty strings
            formed_word_strings.append(word_str)

    if not formed_word_strings and len(word_positions) > 1:
         print("WARNING: No valid word strings formed after processing details.")
         return False, False

    if dawg_obj is None: # Safety check
         print("ERROR: DAWG object not passed to is_valid_play!")
         return False, False
    for word in formed_word_strings:
        dawg_search_result = dawg_obj.search(word) # Use argument
        if not dawg_search_result:
            # print(f"DEBUG is_valid_play: Invalid word found: {word}") # Optional debug
            return False, False

    # --- Connection Rules Check ---
    if is_first_play:
        center_r, center_c = CENTER_SQUARE # Unpack tuple
        center_square_played = False
        for r, c in newly_placed_positions_coords:
            if r == center_r and c == center_c:
                center_square_played = True
                break
        if not center_square_played:
             return False, False
    else:
        connects = False
        if original_tiles is None:
             print("Warning: original_tiles is None in is_valid_play connection check.")
             return False, False

        for r, c in newly_placed_positions_coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE_C and 0 <= nc < GRID_SIZE_C and original_tiles[nr][nc]:
                    connects = True; break
            if connects: break
        if not connects:
            return False, False

    # --- Bingo Check ---
    tiles_played_from_rack = len(newly_placed_positions_coords)
    is_bingo = (initial_rack_size == 7 and tiles_played_from_rack == 7)

    return True, is_bingo
    # --- End of indented block ---






#################################################################
#################################################################
#################################################################




# --- _gaddag_traverse ---
# (Defined AFTER is_valid_play, calculate_score, find_all_words_formed)
def _gaddag_traverse(
    anchor_pos,
    int[:] rack_counts_c, # Memory view of incoming C int array
    tiles,
    board,
    blanks,
    cross_check_sets,
    gaddag_node, # Current node in GADDAG traversal (implicitly object)
    object gaddag_root_node, # Root of the GADDAG structure
    current_word_tiles,
    bint is_reversed, # Already typed as bint
    current_axis,
    all_found_moves,
    unique_move_signatures,
    original_tiles_state,
    bint is_first_play, # Already typed as bint
    int full_rack_size,
    object dawg_obj, # Added DAWG object argument
    int max_len=GRID_SIZE_C,
    int depth=0
):
    """ Recursive helper using C array for rack counts. Creates new lists for recursion. """
    # ... (cdef declarations remain the same) ...
    cdef object newly_placed_list_details, new_tiles_sig, temp_tiles, temp_blanks
    cdef object move_blanks_coords, newly_placed_coords, all_words_formed_details
    cdef object primary_word_tiles, primary_word_str, start_pos, orientation
    cdef str orientation_coord
    cdef object word_with_blanks_list, word_with_blanks, leave, move_details_dict
    cdef str letter
    cdef object next_node # Reverted type back to object
    cdef object next_pos, existing_tile, cross_axis
    cdef object allowed_letters, tile_on_anchor
    cdef int[:] current_rack_counts_c # Memory view for current level
    cdef int[27] temp_rack_counts_c_arr # Actual C array for modification
    cdef int i # Loop variable
    cdef object r_last_py, c_last_py, _ign1, _ign2, _ign3 # Intermediate Python objects
    cdef object last_elem # Variable for explicit check
    cdef object next_rack_np_arr # To hold the new numpy array
    cdef int r_last, c_last, next_r, next_c, anchor_r, anchor_c, ref_r, ref_c, score
    cdef bint is_valid, is_bingo, just_crossed_separator
    cdef int letter_idx, blank_idx
    cdef object anchor_r_obj, anchor_c_obj

    if depth > 20: return
    if not current_word_tiles: return
    if len(current_word_tiles) > max_len: return

    last_elem = current_word_tiles[-1]
    if isinstance(last_elem, (list, tuple)) and len(last_elem) == 5:
        r_last_py, c_last_py, _ign1, _ign2, _ign3 = last_elem
        r_last = <int>r_last_py
        c_last = <int>c_last_py
    else:
        raise TypeError(f"Expected 5-tuple at end of current_word_tiles, got {type(last_elem)}: {last_elem}")


    # --- Check if current path forms a valid move ---
    if gaddag_node.is_terminal and not is_reversed:
        newly_placed_list_details = []
        for r_lp, c_lp, l_lp, _, is_new_lp in current_word_tiles:
            if is_new_lp:
                newly_placed_list_details.append((r_lp, c_lp, l_lp))

        if newly_placed_list_details:
            new_tiles_sig = tuple(sorted(newly_placed_list_details))
            if new_tiles_sig not in unique_move_signatures:
                temp_tiles = [row[:] for row in original_tiles_state]
                temp_blanks = set(blanks); move_blanks_coords = set(); newly_placed_coords = set()
                for r, c, letter_obj, is_blank_obj, is_new_obj in current_word_tiles:
                    if is_new_obj:
                        if 0 <= r < GRID_SIZE_C and 0 <= c < GRID_SIZE_C:
                            temp_tiles[r][c] = letter_obj; newly_placed_coords.add((r, c))
                            if is_blank_obj: temp_blanks.add((r, c)); move_blanks_coords.add((r, c))

                # --- MODIFIED CALL: Pass dawg_obj ---
                is_valid, is_bingo = is_valid_play(newly_placed_list_details, temp_tiles, is_first_play, full_rack_size, original_tiles_state, None, dawg_obj)
                # --- END MODIFICATION ---

                if is_valid:
                    unique_move_signatures.add(new_tiles_sig)
                    score = calculate_score(newly_placed_list_details, board, temp_tiles, temp_blanks)
                    all_words_formed_details = find_all_words_formed(newly_placed_list_details, temp_tiles)

                    # ... (rest of primary word finding, formatting, leave calc - unchanged) ...
                    primary_word_tiles = []; primary_word_str = ""; start_pos = (0, 0); orientation = current_axis
                    orientation_coord = "right" if current_axis == 'H' else "down"
                    if all_words_formed_details:
                         found_primary = False
                         for word_detail in all_words_formed_details:
                             is_along_axis = False
                             if orientation == 'H' and len(set(r for r,c,l in word_detail)) == 1: is_along_axis = True
                             elif orientation == 'V' and len(set(c for r,c,l in word_detail)) == 1: is_along_axis = True
                             if is_along_axis and any((t[0], t[1]) in newly_placed_coords for t in word_detail):
                                 primary_word_tiles = word_detail; found_primary = True; break
                         if not found_primary:
                              for word_detail in all_words_formed_details:
                                  if any((t[0], t[1]) in newly_placed_coords for t in word_detail):
                                       primary_word_tiles = word_detail
                                       if len(set(r for r,c,l in primary_word_tiles)) == 1: orientation = 'H'
                                       elif len(set(c for r,c,l in primary_word_tiles)) == 1: orientation = 'V'
                                       break
                         if not primary_word_tiles and all_words_formed_details:
                              primary_word_tiles = all_words_formed_details[0]
                              if len(set(r for r,c,l in primary_word_tiles)) == 1: orientation = 'H'
                              elif len(set(c for r,c,l in primary_word_tiles)) == 1: orientation = 'V'

                         if primary_word_tiles:
                             primary_word_str = "".join(t[2] for t in primary_word_tiles)
                             # Determine start_pos based on orientation and min row/col
                             if orientation == 'H':
                                 start_pos = min(primary_word_tiles, key=lambda x: x[1])[:2]
                             elif orientation == 'V':
                                 start_pos = min(primary_word_tiles, key=lambda x: x[0])[:2]
                             else: # Fallback if orientation unclear
                                 start_pos = primary_word_tiles[0][:2]


                    word_with_blanks_list = []
                    for wr, wc, w_letter in primary_word_tiles:
                        is_blank_in_word = (wr, wc) in newly_placed_coords and (wr, wc) in move_blanks_coords
                        word_with_blanks_list.append(w_letter.lower() if is_blank_in_word else w_letter.upper())
                    word_with_blanks = "".join(word_with_blanks_list)
                    leave = []
                    for i in range(26):
                        leave.extend([chr(ord('A') + i)] * rack_counts_c[i])
                    leave.extend([' '] * rack_counts_c[26])

                    move_details_dict = {
                        'positions': [(t[0], t[1], t[2]) for t in primary_word_tiles], 'blanks': move_blanks_coords,
                        'word': primary_word_str, 'score': score, 'start': start_pos,
                        'direction': orientation_coord,
                        'leave': leave, 'is_bingo': is_bingo, 'word_with_blanks': word_with_blanks,
                        'newly_placed': newly_placed_list_details
                    }
                    all_found_moves.append(move_details_dict)

    # --- Explore Next Steps ---
    current_rack_counts_c = rack_counts_c
    blank_idx = 26

    for letter, next_node in gaddag_node.children.items():
        if letter == Gaddag.SEPARATOR:
            if is_reversed:
                # --- MODIFIED CALL: Pass dawg_obj ---
                _gaddag_traverse(
                    anchor_pos, current_rack_counts_c, tiles, board, blanks, cross_check_sets,
                    next_node,
                    gaddag_root_node, # Pass the root node through
                    current_word_tiles, False, current_axis, # Pass same list
                    all_found_moves, unique_move_signatures, original_tiles_state,
                    is_first_play, full_rack_size, dawg_obj, max_len, depth + 1 # Pass dawg_obj
                )
                # --- END MODIFICATION ---
            continue

        # ... (coordinate calculation logic - unchanged) ...
        next_r, next_c = -1, -1
        if is_reversed:
            if current_axis == 'H': next_r, next_c = r_last, c_last - 1
            else:                   next_r, next_c = r_last - 1, c_last
        else:
            anchor_r_obj, anchor_c_obj = anchor_pos
            anchor_r = <int>anchor_r_obj
            anchor_c = <int>anchor_c_obj
            tile_on_anchor = None
            for t_r, t_c, t_l, t_b, t_n in current_word_tiles:
                 if t_r == anchor_r and t_c == anchor_c:
                      tile_on_anchor = (t_r, t_c); break
            just_crossed_separator = False
            if len(current_word_tiles) > 0:
                if current_axis == 'H':
                    if c_last <= anchor_c: just_crossed_separator = True
                else:
                    if r_last <= anchor_r: just_crossed_separator = True
            ref_r = anchor_r if just_crossed_separator else r_last
            ref_c = anchor_c if just_crossed_separator else c_last
            if current_axis == 'H': next_r, next_c = ref_r, ref_c + 1
            else:                   next_r, next_c = ref_r + 1, ref_c

        if not (0 <= next_r < GRID_SIZE_C and 0 <= next_c < GRID_SIZE_C): continue

        next_pos = (next_r, next_c)
        existing_tile = tiles[next_r][next_c]

        if not existing_tile:
            cross_axis = 'V' if current_axis == 'H' else 'H'
            allowed_letters = cross_check_sets.get(next_pos, {}).get(cross_axis, set())

            letter_idx = get_char_index(letter)
            if letter_idx == -1: continue

            # Option 1a: Use regular tile
            if current_rack_counts_c[letter_idx] > 0 and letter in allowed_letters:
                for i in range(27): temp_rack_counts_c_arr[i] = current_rack_counts_c[i]
                temp_rack_counts_c_arr[letter_idx] -= 1
                next_rack_np_arr = np.array(temp_rack_counts_c_arr, dtype=np.intc)
                # --- MODIFIED CALL: Pass dawg_obj ---
                _gaddag_traverse(
                    anchor_pos, next_rack_np_arr, tiles, board, blanks, cross_check_sets,
                    next_node,
                    gaddag_root_node, # Pass the root node through
                    current_word_tiles + [(next_r, next_c, letter, <bint>False, <bint>True)],
                    is_reversed, current_axis, all_found_moves, unique_move_signatures,
                    original_tiles_state, is_first_play, full_rack_size, dawg_obj, max_len, depth + 1 # Pass dawg_obj
                )
                # --- END MODIFICATION ---

            # Option 1b: Use blank tile
            if current_rack_counts_c[blank_idx] > 0 and ' ' in allowed_letters:
                for i in range(27): temp_rack_counts_c_arr[i] = current_rack_counts_c[i]
                temp_rack_counts_c_arr[blank_idx] -= 1
                next_rack_np_arr = np.array(temp_rack_counts_c_arr, dtype=np.intc)
                # --- MODIFIED CALL: Pass dawg_obj ---
                _gaddag_traverse(
                    anchor_pos, next_rack_np_arr, tiles, board, blanks, cross_check_sets,
                    next_node,
                    gaddag_root_node, # Pass the root node through
                    current_word_tiles + [(next_r, next_c, letter, <bint>True, <bint>True)],
                    is_reversed, current_axis, all_found_moves, unique_move_signatures,
                    original_tiles_state, is_first_play, full_rack_size, dawg_obj, max_len, depth + 1 # Pass dawg_obj
                )
                # --- END MODIFICATION ---
        # Case 2: Square has matching existing tile
        elif existing_tile == letter:
            # --- MODIFIED CALL: Pass dawg_obj ---
            _gaddag_traverse(
                anchor_pos, current_rack_counts_c, tiles, board, blanks, cross_check_sets,
                next_node,
                gaddag_root_node, # Pass the root node through
                current_word_tiles + [(next_r, next_c, letter, <bint>False, <bint>False)],
                is_reversed, current_axis, all_found_moves, unique_move_signatures,
                original_tiles_state, is_first_play, full_rack_size, dawg_obj, max_len, depth + 1 # Pass dawg_obj
            )
            # --- END MODIFICATION ---







##########################################################################################################
##########################################################################################################
##########################################################################################################







# --- compute_cross_checks_cython ---
# (Defined AFTER _gaddag_traverse)
cpdef dict compute_cross_checks_cython(list tiles, object dawg_obj):
    """
    Computes the cross-check sets for empty squares on the board using Cython.

    Args:
        tiles (list): The current board state (list of lists).
        dawg_obj (object): The DAWG object for dictionary lookups.

    Returns:
        dict: The cross_check_sets dictionary.
    """
    # --- Declare C types for performance ---
    cdef int r, c, rr, cc
    cdef str up_word, down_word, left_word, right_word, letter, full_word_v, full_word_h
    cdef set allowed_letters_v, allowed_letters_h
    cdef dict cross_check_sets = {}
    cdef bint search_result # For DAWG search result

    # --- Use GRID_SIZE_C for loops ---
    for r in range(GRID_SIZE_C):
        for c in range(GRID_SIZE_C):
            if not tiles[r][c]: # Only process empty squares
                # Vertical check
                up_word = ""
                rr = r - 1
                while rr >= 0 and tiles[rr][c]:
                    up_word = tiles[rr][c] + up_word
                    rr -= 1
                down_word = ""
                rr = r + 1
                while rr < GRID_SIZE_C and tiles[rr][c]:
                    down_word += tiles[rr][c]
                    rr += 1

                allowed_letters_v = set()
                if not up_word and not down_word:
                    allowed_letters_v = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ ') # Include blank
                else:
                    for letter_ord in range(ord('A'), ord('Z') + 1):
                        letter = chr(letter_ord)
                        full_word_v = up_word + letter + down_word
                        search_result = dawg_obj.search(full_word_v)
                        if search_result:
                            allowed_letters_v.add(letter)
                    if allowed_letters_v:
                        allowed_letters_v.add(' ')

                # Horizontal check
                left_word = ""
                cc = c - 1
                while cc >= 0 and tiles[r][cc]:
                    left_word = tiles[r][cc] + left_word
                    cc -= 1
                right_word = ""
                cc = c + 1
                while cc < GRID_SIZE_C and tiles[r][cc]:
                    right_word += tiles[r][cc]
                    cc += 1

                allowed_letters_h = set()
                if not left_word and not right_word:
                    allowed_letters_h = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ ') # Include blank
                else:
                    for letter_ord in range(ord('A'), ord('Z') + 1):
                        letter = chr(letter_ord)
                        full_word_h = left_word + letter + right_word
                        search_result = dawg_obj.search(full_word_h)
                        if search_result:
                            allowed_letters_h.add(letter)
                    if allowed_letters_h:
                        allowed_letters_h.add(' ')

                cross_check_sets[(r, c)] = {'V': allowed_letters_v, 'H': allowed_letters_h}

    return cross_check_sets
