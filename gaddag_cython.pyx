
# 30APR25 V5

# gaddag_cython.pyx
# cython: language_level=3

import cython # To use decorators like @cython.locals
import numpy as np
cimport numpy as np
np.import_array()

from collections import Counter
# --- MODIFICATION: Move cimport here ---
#from libc.stddef cimport Py_ssize_t
# --- END MODIFICATION ---

from scrabble_helpers import (
    CENTER_SQUARE, TILE_DISTRIBUTION,
    RED, PINK, BLUE, LIGHT_BLUE, LETTERS,
    get_coord, get_anchor_points, Gaddag, GaddagNode,
    DAWG as DAWG_cls
)
from scrabble_helpers import perform_leave_lookup

DEF GRID_SIZE_C = 15

# --- get_char_index ---
cdef int get_char_index(str letter):
    cdef int index
    if letter == '?':
        index = 26
    elif 'A' <= letter <= 'Z':
        index = ord(letter) - ord('A')
    else:
        return -1
    return index

# --- find_cross_word ---
cdef list find_cross_word(tuple tile, list tiles, str main_orientation):
    cdef int r, c, min_row, max_row, min_col, max_col, rr_cw, cc_cw
    cdef list cross_word = []
    cdef object r_obj, c_obj, _

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



cpdef float evaluate_single_move_cython(dict move_dict):
    """
    Cython version: Combines immediate score and leave value.
    Calls evaluate_leave_cython directly.
    """
    cdef float immediate_score = 0.0
    cdef list leave = []
    cdef float leave_score_adjustment = 0.0
    cdef float combined_score = 0.0
    cdef object leave_obj # Use object to handle potential non-list type from dict.get

    # Use .get with default values for safety
    # Ensure score is treated as float
    immediate_score = float(move_dict.get('score', 0.0))

    leave_obj = move_dict.get('leave', []) # Get the 'leave' item
    # Check if the retrieved item is actually a list
    if isinstance(leave_obj, list):
        leave = <list>leave_obj # Cast to list if it is one
    else:
        # Handle cases where 'leave' might not be a list (e.g., None or other type)
        # You could print a warning here if needed:
        # print(f"Warning: 'leave' in move_dict is not a list, type: {type(leave_obj)}")
        leave = [] # Default to an empty list

    # Call the existing Cython leave evaluation function (defined in this file)
    leave_score_adjustment = evaluate_leave_cython(leave)

    combined_score = immediate_score + leave_score_adjustment
    return combined_score







# --- find_main_word ---
cdef tuple find_main_word(list new_tiles, list tiles):
    cdef int row, col, min_row, max_row, min_col, max_col, r_nt, c_nt, c_mw, r_mw
    cdef list main_word
    cdef str orientation
    cdef object r_nt_obj, c_nt_obj, _ign1_nt
    cdef set rows_set = set()
    cdef set cols_set = set()

    if not new_tiles: return [], None

    for tile_tuple in new_tiles:
        if isinstance(tile_tuple, tuple) and len(tile_tuple) >= 3:
            r_nt_obj, c_nt_obj, _ign1_nt = tile_tuple[:3]
            r_nt = <int>r_nt_obj
            c_nt = <int>c_nt_obj
            rows_set.add(r_nt)
            cols_set.add(c_nt)
        else:
            pass

    if len(rows_set) == 1:
        orientation = "horizontal"; row = rows_set.pop()
        min_col = 999; max_col = -1
        for tile_tuple in new_tiles:
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
        for tile_tuple in new_tiles:
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








# --- find_all_words_formed ---
cpdef list find_all_words_formed(list new_tiles, list tiles):
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

    main_word_tiles, orientation = find_main_word(new_tiles, tiles)

    if main_word_tiles:
        words.append(main_word_tiles)
        for tile in new_tiles:
            if (tile[0], tile[1]) in new_positions_set:
                cross_word = find_cross_word(tile, tiles, orientation)
                if cross_word:
                    words.append(cross_word)
    elif len(new_tiles) == 1:
        tile = new_tiles[0]
        cross_h = find_cross_word(tile, tiles, "vertical");
        if cross_h: words.append(cross_h)
        cross_v = find_cross_word(tile, tiles, "horizontal");
        if cross_v: words.append(cross_v)

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









# --- calculate_score ---
cpdef int calculate_score(list new_tiles, list board, list tiles, set blanks):
    cdef int total_score = 0, word_score, word_multiplier
    cdef int r, c, letter_value, letter_multiplier, r_np, c_np
    cdef bint is_blank
    cdef set new_positions = set()
    for r_np, c_np, _ in new_tiles:
        new_positions.add((r_np, c_np))
    cdef list words_formed_details, word_tiles
    cdef str letter
    cdef object square_color

    words_formed_details = find_all_words_formed(new_tiles, tiles)

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





def get_final_score_for_sort(eval_dict):
    """Helper function to extract 'final_score' for sorting."""
    # Add error handling in case the key is missing or value is not numeric
    try:
        # Attempt to get the score, default to negative infinity if missing
        score = eval_dict.get('final_score', -float('inf'))
        # Ensure it's a float before returning
        return float(score)
    except (TypeError, ValueError):
        # Handle cases where 'final_score' exists but isn't a number
        return -float('inf') # Return a value that sorts to the end









@cython.locals(move=dict, evaluated_score=float) # Optional type hints for performance
cpdef list standard_evaluation_cython(list all_moves):
    """
    Performs standard evaluation (raw score + leave) on a list of moves.
    Calls evaluate_single_move_cython.
    Returns a list of dictionaries [{'move': move_dict, 'final_score': float}],
    sorted by final_score descending.
    """
    cdef list temp_evaluated_plays = []
    cdef dict eval_dict
    cdef float best_play_evaluation = -float('inf') # Not strictly needed for return, but useful if optimizing further

    if all_moves is None: # Handle None input
        return []

    for move in all_moves:
        # Ensure move is a dictionary before processing
        if not isinstance(move, dict):
            # print(f"Warning: Item in all_moves is not a dict: {type(move)}") # Optional warning
            continue

        # Call the Cython function to get combined score + leave
        evaluated_score = evaluate_single_move_cython(move)

        # Create the dictionary structure expected by the Python code
        eval_dict = {'move': move, 'final_score': evaluated_score}
        temp_evaluated_plays.append(eval_dict)

        # Keep track of best score (optional, might be removed if only sorting matters)
        if evaluated_score > best_play_evaluation:
            best_play_evaluation = evaluated_score

    # Sort the results based on 'final_score' descending
    # The key for sorting needs to be accessible from Python later,
    # so using a simple lambda here is fine, or define a helper if needed.
    # Python's sort is efficient even when called from Cython on Python list objects.
    temp_evaluated_plays.sort(key=get_final_score_for_sort, reverse=True)

    return temp_evaluated_plays







@cython.locals(best_play_raw_score=float, current_play_eval=float)
cpdef tuple ai_turn_logic_cython(
    list all_moves,
    list current_rack,
    object board_tile_counts_obj, # Pass Counter as object
    int blanks_played_count,
    int bag_count,
    object get_remaining_tiles_func, # Pass Python function as object
    object find_best_exchange_option_func, # Pass Python function as object
    float EXCHANGE_PREFERENCE_THRESHOLD,
    float MIN_SCORE_TO_AVOID_EXCHANGE
    ):
    """
    Performs standard evaluation and decides between play, exchange, or pass.
    Calls standard_evaluation_cython and Python helper functions passed as arguments.
    Passes board_tile_counts and blanks_played_count to find_best_exchange_option_func.

    Returns:
        tuple: (action_chosen_str, best_move_data)
               where best_move_data is either the best move dict or the list of tiles to exchange.
               Returns ('pass', None) if no action is chosen.
    """
    cdef list evaluated_play_options = []
    cdef dict best_play_move = None
    cdef float best_play_evaluation = -float('inf')
    cdef list best_exchange_tiles = []
    cdef float best_exchange_evaluation = -float('inf')
    cdef bint can_play = False
    cdef bint can_exchange_proactively = (bag_count >= 1)
    cdef str action_chosen = 'pass'
    cdef dict remaining_dict_for_exchange # No longer needed here
    cdef tuple exchange_result

    # --- Evaluate Play Options ---
    can_play = bool(all_moves)
    if can_play:
        evaluated_play_options = standard_evaluation_cython(all_moves)
        if evaluated_play_options:
            best_play_evaluation = evaluated_play_options[0]['final_score']
            best_play_move = evaluated_play_options[0]['move']
        else:
            can_play = False
            best_play_move = None
            best_play_evaluation = -float('inf')

    # --- Evaluate Exchange Option ---
    if not can_play or can_exchange_proactively:
        try:
            exchange_result = find_best_exchange_option_func(
                current_rack,
                board_tile_counts_obj, # Pass board counts object
                blanks_played_count,   # Pass blanks played count
                bag_count
            )

            if isinstance(exchange_result, tuple) and len(exchange_result) == 2:
                best_exchange_tiles_obj, best_exchange_evaluation_obj = exchange_result
                if isinstance(best_exchange_tiles_obj, list):
                    best_exchange_tiles = best_exchange_tiles_obj
                try:
                    best_exchange_evaluation = float(best_exchange_evaluation_obj)
                except (TypeError, ValueError):
                    best_exchange_evaluation = -float('inf')
            else:
                 print("Warning: find_best_exchange_option returned unexpected type.")
                 best_exchange_tiles = []
                 best_exchange_evaluation = -float('inf')

        except Exception as e:
            print(f"Error calling Python helper function find_best_exchange_option from Cython: {e}")
            best_exchange_tiles = []
            best_exchange_evaluation = -float('inf')


    # --- Final Decision Logic ---
    if can_play and best_play_move is not None:
        best_play_raw_score = float(best_play_move.get('score', 0.0))

        if best_play_raw_score >= MIN_SCORE_TO_AVOID_EXCHANGE:
            action_chosen = 'play'
        else:
            action_chosen = 'play'
            if best_exchange_tiles:
                current_play_eval = best_play_evaluation
                if best_exchange_evaluation > current_play_eval + EXCHANGE_PREFERENCE_THRESHOLD:
                    action_chosen = 'exchange'
    elif best_exchange_tiles:
         action_chosen = 'exchange'
    else:
        action_chosen = 'pass'

    # --- Return chosen action and relevant data ---
    if action_chosen == 'play':
        return (action_chosen, best_play_move)
    elif action_chosen == 'exchange':
        return (action_chosen, best_exchange_tiles)
    else: # Pass
        return (action_chosen, None)










# --- is_valid_play ---
cpdef tuple is_valid_play(list word_positions, list tiles, bint is_first_play, int initial_rack_size, list original_tiles, object rack, object dawg_obj):
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
    cdef object tile_detail
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

    all_words_details = find_all_words_formed(word_positions, tiles)

    if not all_words_details and len(word_positions) > 1:
         return False, False

    formed_word_strings = []
    for word_detail in all_words_details:
        if not isinstance(word_detail, list):
            continue
        word_chars = []
        for item in word_detail:
            if isinstance(item, tuple) and len(item) >= 3:
                tile_detail = item
                word_chars.append(tile_detail[2])
            else:
                continue
        word_str = "".join(word_chars)
        if word_str:
            formed_word_strings.append(word_str)

    if not formed_word_strings and len(word_positions) > 1:
         return False, False

    if dawg_obj is None:
         print("ERROR: DAWG object not passed to is_valid_play!")
         return False, False
    for word in formed_word_strings:
        dawg_search_result = dawg_obj.search(word)
        if not dawg_search_result:
            return False, False

    if is_first_play:
        center_r, center_c = CENTER_SQUARE
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

    tiles_played_from_rack = len(newly_placed_positions_coords)
    is_bingo = (initial_rack_size == 7 and tiles_played_from_rack == 7)

    return True, is_bingo








def _gaddag_traverse(
    anchor_pos,
    int[:] rack_counts_c,
    tiles,
    board,
    blanks,
    cross_check_sets,
    gaddag_node,
    object gaddag_root_node,
    current_word_tiles,
    bint is_reversed,
    current_axis,
    all_found_moves,
    unique_move_signatures,
    original_tiles_state,
    bint is_first_play,
    int full_rack_size,
    object dawg_obj,
    int max_len=GRID_SIZE_C,
    int depth=0
):
    # ... (function body unchanged) ...
    cdef object newly_placed_list_details, new_tiles_sig, temp_tiles, temp_blanks
    cdef object move_blanks_coords, newly_placed_coords, all_words_formed_details
    cdef object primary_word_tiles, primary_word_str, start_pos, orientation
    cdef str orientation_coord
    cdef object word_with_blanks_list, word_with_blanks, leave, move_details_dict
    cdef str letter
    cdef object next_node
    cdef object next_pos, existing_tile, cross_axis
    cdef object allowed_letters, tile_on_anchor
    cdef int[:] current_rack_counts_c
    cdef int[27] temp_rack_counts_c_arr
    cdef int i
    cdef object r_last_py, c_last_py, _ign1, _ign2, _ign3
    cdef object last_elem
    cdef object next_rack_np_arr
    cdef int r_last, c_last, next_r, next_c, anchor_r, anchor_c, ref_r, ref_c, score
    cdef bint is_valid, is_bingo, just_crossed_separator
    cdef int letter_idx, blank_idx
    cdef object anchor_r_obj, anchor_c_obj

    if depth > 20: return
    if not current_word_tiles: return
    if len(current_word_tiles) > max_len: return

    r_last = -1
    c_last = -1
    if current_word_tiles:
        last_elem = current_word_tiles[-1]
        if isinstance(last_elem, (list, tuple)) and len(last_elem) == 5:
            r_last_py, c_last_py, _ign1, _ign2, _ign3 = last_elem
            r_last = <int>r_last_py
            c_last = <int>c_last_py
        else:
            if depth == 0 and not current_word_tiles:
                 pass
            else:
                 raise TypeError(f"Expected 5-tuple at end of current_word_tiles, got {type(last_elem)}: {last_elem}")

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

                is_valid, is_bingo = is_valid_play(newly_placed_list_details, temp_tiles, is_first_play, full_rack_size, original_tiles_state, None, dawg_obj)

                if is_valid:
                    unique_move_signatures.add(new_tiles_sig)
                    score = calculate_score(newly_placed_list_details, board, temp_tiles, temp_blanks)
                    all_words_formed_details = find_all_words_formed(newly_placed_list_details, temp_tiles)

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
                             if orientation == 'H':
                                 start_pos = min(primary_word_tiles, key=lambda x: x[1])[:2]
                             elif orientation == 'V':
                                 start_pos = min(primary_word_tiles, key=lambda x: x[0])[:2]
                             else:
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

    current_rack_counts_c = rack_counts_c
    blank_idx = 26

    children_dict = getattr(gaddag_node, 'children', {})
    if not isinstance(children_dict, dict):
         children_dict = {}

    for letter, next_node in children_dict.items():
        if letter == Gaddag.SEPARATOR:
            if is_reversed:
                _gaddag_traverse(
                    anchor_pos, current_rack_counts_c, tiles, board, blanks, cross_check_sets,
                    next_node,
                    gaddag_root_node,
                    current_word_tiles, False, current_axis,
                    all_found_moves, unique_move_signatures, original_tiles_state,
                    is_first_play, full_rack_size, dawg_obj, max_len, depth + 1
                )
            continue

        next_r, next_c = -1, -1
        if is_reversed:
            if current_word_tiles:
                 if current_axis == 'H': next_r, next_c = r_last, c_last - 1
                 else:                   next_r, next_c = r_last - 1, c_last
            else: continue
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
            else:
                 just_crossed_separator = True

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

            if current_rack_counts_c[letter_idx] > 0 and letter in allowed_letters:
                for i in range(27): temp_rack_counts_c_arr[i] = current_rack_counts_c[i]
                temp_rack_counts_c_arr[letter_idx] -= 1
                next_rack_np_arr = np.array(temp_rack_counts_c_arr, dtype=np.intc)
                _gaddag_traverse(
                    anchor_pos, next_rack_np_arr, tiles, board, blanks, cross_check_sets,
                    next_node,
                    gaddag_root_node,
                    current_word_tiles + [(next_r, next_c, letter, <bint>False, <bint>True)],
                    is_reversed, current_axis, all_found_moves, unique_move_signatures,
                    original_tiles_state, is_first_play, full_rack_size, dawg_obj, max_len, depth + 1
                )

            if current_rack_counts_c[blank_idx] > 0 and ' ' in allowed_letters:
                for i in range(27): temp_rack_counts_c_arr[i] = current_rack_counts_c[i]
                temp_rack_counts_c_arr[blank_idx] -= 1
                next_rack_np_arr = np.array(temp_rack_counts_c_arr, dtype=np.intc)
                _gaddag_traverse(
                    anchor_pos, next_rack_np_arr, tiles, board, blanks, cross_check_sets,
                    next_node,
                    gaddag_root_node,
                    current_word_tiles + [(next_r, next_c, letter, <bint>True, <bint>True)],
                    is_reversed, current_axis, all_found_moves, unique_move_signatures,
                    original_tiles_state, is_first_play, full_rack_size, dawg_obj, max_len, depth + 1
                )
        elif existing_tile == letter:
            _gaddag_traverse(
                anchor_pos, current_rack_counts_c, tiles, board, blanks, cross_check_sets,
                next_node,
                gaddag_root_node,
                current_word_tiles + [(next_r, next_c, letter, <bint>False, <bint>False)],
                is_reversed, current_axis, all_found_moves, unique_move_signatures,
                original_tiles_state, is_first_play, full_rack_size, dawg_obj, max_len, depth + 1
            )







# --- compute_cross_checks_cython ---
cpdef dict compute_cross_checks_cython(list tiles, object dawg_obj):
    # ... (function body unchanged) ...
    cdef int r, c, rr, cc
    cdef str up_word, down_word, left_word, right_word, letter, full_word_v, full_word_h
    cdef set allowed_letters_v, allowed_letters_h
    cdef dict cross_check_sets = {}
    cdef bint search_result
    cdef int letter_ord

    for r in range(GRID_SIZE_C):
        for c in range(GRID_SIZE_C):
            if not tiles[r][c]:
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
                    allowed_letters_v = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
                else:
                    for letter_ord in range(ord('A'), ord('Z') + 1):
                        letter = chr(letter_ord)
                        full_word_v = up_word + letter + down_word
                        search_result = dawg_obj.search(full_word_v)
                        if search_result:
                            allowed_letters_v.add(letter)
                    if allowed_letters_v:
                        allowed_letters_v.add(' ')

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
                    allowed_letters_h = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
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







# --- evaluate_leave_cython ---
cpdef float evaluate_leave_cython(list rack, bint verbose=False):
    # ... (function body unchanged) ...
    cdef int num_tiles = len(rack)
    cdef list rack_with_question_marks
    cdef str leave_key, tile
    cdef float leave_float = 0.0

    if num_tiles == 0:
        return 0.0
    if num_tiles > 6:
        return 0.0

    rack_with_question_marks = ['?' if tile == ' ' else tile for tile in rack]
    leave_key = "".join(sorted(rack_with_question_marks))

    try:
        leave_float = perform_leave_lookup(leave_key)
        return leave_float
    except Exception as e:
        print(f"Error (Cython) generating key or calling Python lookup for rack '{rack}': {e}")
        return 0.0






cpdef float calculate_luck_factor_cython(
    list drawn_tiles,
    list move_rack_before,
    object board_tile_counts_obj, # Pass Counter as object
    int blanks_played_count,
    object get_remaining_tiles_func # Pass Python function as object
    ):
    """
    Calculates the luck factor for a set of drawn tiles.
    Calls evaluate_leave_cython and replicates analyze_unseen_pool logic.
    Calls the passed Python get_remaining_tiles function.
    """
    # These are declared with cdef
    cdef float drawn_leave_value = 0.0
    cdef dict remaining_dict
    cdef float expected_value_sum = 0.0
    cdef int total_unseen_tiles = 0
    cdef float probability = 0.0
    cdef float single_tile_value = 0.0
    cdef float expected_single_draw_value = 0.0
    cdef float expected_draw_value_total = 0.0
    cdef float luck_factor = 0.0
    # Loop variables declared with cdef
    cdef str tile
    cdef int count

    if not drawn_tiles:
        return 0.0 # No luck factor if no tiles were drawn

    # 1. Calculate the actual leave value of the drawn tiles
    drawn_leave_value = evaluate_leave_cython(drawn_tiles)

    # 2. Calculate the expected leave value before the draw
    try:
        # Call the passed Python function to get remaining tiles before draw
        remaining_dict = get_remaining_tiles_func(move_rack_before, board_tile_counts_obj, blanks_played_count)

        # Replicate analyze_unseen_pool logic here
        total_unseen_tiles = sum(remaining_dict.values())
        expected_value_sum = 0.0

        if total_unseen_tiles > 0:
            # Iterate through the remaining tiles dictionary
            for tile, count in remaining_dict.items(): # Use cdef'd tile and count
                if count <= 0:
                    continue
                if not isinstance(tile, str):
                    continue

                # Get single tile leave value using evaluate_leave_cython
                try:
                    single_tile_value = evaluate_leave_cython([tile])
                except Exception as e_eval:
                    # print(f"Warning (Cython luck): Could not evaluate single tile '{tile}': {e_eval}")
                    single_tile_value = 0.0

                probability = <float>count / total_unseen_tiles
                expected_value_sum += single_tile_value * probability

        expected_single_draw_value = expected_value_sum

    except Exception as e_pool:
        print(f"Error calculating expected value in Cython luck factor: {e_pool}")
        expected_single_draw_value = 0.0

    # 3. Calculate total expected value and luck factor
    expected_draw_value_total = expected_single_draw_value * len(drawn_tiles)
    luck_factor = drawn_leave_value - expected_draw_value_total

    return luck_factor






cpdef float get_expected_draw_value_cython(
    list current_rack,            # Need rack to calculate remaining
    object board_tile_counts_obj, # Pass Counter as object
    int blanks_played_count,
    object get_remaining_tiles_func # Pass Python function as object
    ):
    """
    Calculates the expected value of drawing a single tile from the unseen pool.
    Calls the passed Python get_remaining_tiles function.
    Calls evaluate_leave_cython for single tiles.
    """
    cdef dict remaining_dict
    cdef float expected_value_sum = 0.0
    cdef int total_unseen_tiles = 0
    cdef float probability = 0.0
    cdef float single_tile_value = 0.0
    cdef str tile
    cdef int count

    try:
        # Call the passed Python function to get remaining tiles
        remaining_dict = get_remaining_tiles_func(current_rack, board_tile_counts_obj, blanks_played_count)

        total_unseen_tiles = sum(remaining_dict.values())

        if total_unseen_tiles > 0:
            for tile, count in remaining_dict.items():
                if count <= 0:
                    continue
                if not isinstance(tile, str):
                    continue
                try:
                    single_tile_value = evaluate_leave_cython([tile])
                except Exception as e_eval:
                    # print(f"Warning (Cython expected val): Could not evaluate single tile '{tile}': {e_eval}")
                    single_tile_value = 0.0

                probability = <float>count / total_unseen_tiles
                expected_value_sum += single_tile_value * probability

        return expected_value_sum # Return the calculated expected value

    except Exception as e_pool:
        print(f"Error calculating expected value in Cython: {e_pool}")
        return 0.0 # Return default on error







# --- Helper function for sorting ---
def _get_move_score_for_sort(move_dict):
    """Helper to safely get score for sorting, defaulting to -1."""
    if isinstance(move_dict, dict):
        return move_dict.get('score', -1)
    return -1







# --- Consolidated Move Generation Function ---
def generate_all_moves_gaddag_cython(
    object rack, # Keep as object
    object tiles,
    object board,
    object blanks,
    object gaddag_root,
    object dawg_obj
):
    """
    Generates ALL valid Scrabble moves using GADDAG traversal.
    Consolidated version performing setup, core logic, and post-processing in Cython.
    Defined as 'def' to bypass cpdef closure error.
    """
    # --- Type Declarations ---
    # from libc.stddef cimport Py_ssize_t # Moved to top
    cdef list all_found_moves = []
    cdef set unique_move_signatures = set()
    cdef bint is_first_play
    cdef set anchors
    cdef list original_tiles_state
    cdef int full_rack_size
    cdef int i
    cdef object rack_counts_py # Use object for Python Counter
    # --- MODIFICATION: Changed back to object ---
    cdef object rack_counts_c_arr # Use object for NumPy array fallback
    # --- END MODIFICATION ---
    cdef dict cross_check_sets
    # --- MODIFICATION: Ensure declared ---
    cdef set processed_adjacent_starts = set()
    # --- END MODIFICATION ---
    cdef tuple anchor_pos, adj_pos
    cdef int r_anchor, c_anchor, dr, dc, nr, nc
    cdef object allowed_h_obj, allowed_v_obj
    cdef set allowed_h, allowed_v
    cdef str tile_letter, existing_tile_letter, assigned_letter, start_axis
    cdef int count # Count from Counter items
    cdef int letter_idx_s1, blank_idx
    cdef object next_node_obj
    cdef object next_node # Keep as object for flexibility with GaddagNode
    cdef list initial_tiles
    cdef object next_rack_counts_c_arr_s1, next_rack_counts_c_arr_blank # Use object
    cdef object initial_rack_counts_c_copy # Use object
    cdef object anchor_r_obj, anchor_c_obj
    cdef int assigned_letter_ord
    cdef list final_unique_moves = []
    cdef set seen_final_signatures = set()
    cdef dict move # Keep as dict (Python object)
    cdef list sig_details, sig_tuple_list
    cdef tuple sig_tuple, item
    cdef bint valid_sig

    # --- Setup Phase ---
    if gaddag_root is None or dawg_obj is None:
         print("ERROR (Cython): GADDAG root or DAWG object is None in generate_all_moves.")
         return []

    # Cast Python objects if needed for clarity
    cdef list py_tiles = <list>tiles
    cdef list py_rack = <list>rack
    cdef list py_board = <list>board
    cdef set py_blanks = <set>blanks

    is_first_play = sum(1 for row in py_tiles for t in row if t) == 0
    anchors = get_anchor_points(py_tiles, is_first_play) # Call Python helper
    original_tiles_state = [row[:] for row in py_tiles]
    full_rack_size = len(py_rack)

    rack_counts_py = Counter(py_rack)
    rack_counts_c_arr = np.zeros(27, dtype=np.intc) # Create NumPy array
    for i in range(26):
        letter = chr(ord('A') + i)
        rack_counts_c_arr[i] = rack_counts_py.get(letter, 0)
    rack_counts_c_arr[26] = rack_counts_py.get(' ', 0)

    cross_check_sets = compute_cross_checks_cython(py_tiles, dawg_obj)

    # --- Core Logic Phase (Anchor Processing) ---
    initial_rack_counts_c_copy = rack_counts_c_arr.copy()
    blank_idx = 26

    for anchor_pos in anchors:
        if isinstance(anchor_pos, tuple) and len(anchor_pos) == 2:
            anchor_r_obj, anchor_c_obj = anchor_pos
            r_anchor = <int>anchor_r_obj
            c_anchor = <int>anchor_c_obj
        else:
            continue

        allowed_h_obj = cross_check_sets.get(anchor_pos, {}).get('H', set())
        allowed_v_obj = cross_check_sets.get(anchor_pos, {}).get('V', set())
        allowed_h = <set>allowed_h_obj if isinstance(allowed_h_obj, set) else set()
        allowed_v = <set>allowed_v_obj if isinstance(allowed_v_obj, set) else set()

        # Strategy 1
        for tile_letter, count in (<dict>rack_counts_py).items():
            if count > 0 and tile_letter != ' ':
                next_node_obj = getattr(gaddag_root, 'children', {}).get(tile_letter)
                if next_node_obj is not None:
                    next_node = next_node_obj
                    next_rack_counts_c_arr_s1 = rack_counts_c_arr.copy()
                    letter_idx_s1 = ord(tile_letter) - ord('A')
                    if 0 <= letter_idx_s1 < 26:
                        next_rack_counts_c_arr_s1[letter_idx_s1] -= 1
                    else:
                        continue
                    initial_tiles = [(r_anchor, c_anchor, tile_letter, <bint>False, <bint>True)]
                    if tile_letter in allowed_v:
                         _gaddag_traverse(anchor_pos, next_rack_counts_c_arr_s1, py_tiles, py_board, py_blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, 'H', all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size, dawg_obj)
                    if tile_letter in allowed_h:
                         _gaddag_traverse(anchor_pos, next_rack_counts_c_arr_s1, py_tiles, py_board, py_blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, 'V', all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size, dawg_obj)

        # Handle blank for Strategy 1
        if (<dict>rack_counts_py).get(' ', 0) > 0:
            if ' ' in allowed_h or ' ' in allowed_v:
                next_rack_counts_c_arr_blank = rack_counts_c_arr.copy()
                next_rack_counts_c_arr_blank[blank_idx] -= 1
                for assigned_letter_ord in range(ord('A'), ord('Z') + 1):
                    assigned_letter = chr(assigned_letter_ord)
                    next_node_obj = getattr(gaddag_root, 'children', {}).get(assigned_letter)
                    if next_node_obj is not None:
                        next_node = next_node_obj
                        initial_tiles = [(r_anchor, c_anchor, assigned_letter, <bint>True, <bint>True)]
                        if ' ' in allowed_v:
                            _gaddag_traverse(anchor_pos, next_rack_counts_c_arr_blank.copy(), py_tiles, py_board, py_blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, 'H', all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size, dawg_obj)
                        if ' ' in allowed_h:
                            _gaddag_traverse(anchor_pos, next_rack_counts_c_arr_blank.copy(), py_tiles, py_board, py_blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, 'V', all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size, dawg_obj)

        # Strategy 2
        if not is_first_play:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = r_anchor + dr
                nc = c_anchor + dc
                adj_pos = (nr, nc)
                if 0 <= nr < GRID_SIZE_C and 0 <= nc < GRID_SIZE_C:
                    existing_tile_letter = py_tiles[nr][nc]
                    if existing_tile_letter and adj_pos not in processed_adjacent_starts:
                         processed_adjacent_starts.add(adj_pos)
                         next_node_obj = getattr(gaddag_root, 'children', {}).get(existing_tile_letter)
                         if next_node_obj is not None:
                            next_node = next_node_obj
                            initial_tiles = [(nr, nc, existing_tile_letter, <bint>False, <bint>False)]
                            start_axis = 'V' if dr != 0 else 'H'
                            _gaddag_traverse(anchor_pos, initial_rack_counts_c_copy, py_tiles, py_board, py_blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), False, start_axis, all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size, dawg_obj)
                            _gaddag_traverse(anchor_pos, initial_rack_counts_c_copy, py_tiles, py_board, py_blanks, cross_check_sets, next_node, gaddag_root, list(initial_tiles), True, start_axis, all_found_moves, unique_move_signatures, original_tiles_state, is_first_play, full_rack_size, dawg_obj)

  
    all_found_moves.sort(key=_get_move_score_for_sort, reverse=True)

    final_unique_moves = []
    seen_final_signatures = set()
    for move in all_found_moves:
        sig_details = move.get('newly_placed')
        if sig_details is None:
             sig_details = move.get('positions', [])
        try:
            sig_tuple_list = []
            valid_sig = True
            if not isinstance(sig_details, (list, tuple)):
                 valid_sig = False
            else:
                for item in sig_details:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        sig_tuple_list.append(tuple(item[:3]))
                    else:
                        valid_sig = False
                        break
            if not valid_sig:
                continue
            sig_tuple = tuple(sorted(sig_tuple_list)) + (move.get('score', 0),)
        except TypeError as e:
             continue
        if sig_tuple not in seen_final_signatures:
            final_unique_moves.append(move)
            seen_final_signatures.add(sig_tuple)

    return final_unique_moves
