
#Scrabble 28APR25 Cython V3


# gaddag_cython.pyx
# cython: language_level=3

import cython # To use decorators like @cython.locals
import numpy as np # Keep NumPy import as it's used in _gaddag_traverse

# Import necessary Python modules/classes
from collections import Counter

# Import helpers and constants FROM scrabble_helpers
from scrabble_helpers import (
    DAWG, # Keep DAWG for is_valid_play
    # GRID_SIZE, # REMOVED import of Python GRID_SIZE
    CENTER_SQUARE, TILE_DISTRIBUTION,
    RED, PINK, BLUE, LIGHT_BLUE, LETTERS,
    get_coord, evaluate_leave, get_anchor_points, Gaddag, GaddagNode # Ensure GaddagNode is imported
)

# --- Define GRID_SIZE as a C constant ---
DEF GRID_SIZE_C = 15

# Keep as cdef (internal helper)
# REMOVED @cython.locals
cdef int get_char_index(str letter):
    cdef int index
    if letter == '?': # Use '?' internally for blank index
        index = 26
    elif 'A' <= letter <= 'Z': # Added check for valid letter range
        index = ord(letter) - ord('A')
    else:
        return -1 # Indicate error
    return index

# --- Cython version using C array for rack counts ---
# REMOVED @cython.locals decorator from _gaddag_traverse
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
    int max_len=GRID_SIZE_C,
    int depth=0
):
    """ Recursive helper using C array for rack counts. Creates new lists for recursion. """
    # --- Declare types for local variables using cdef ---
    cdef object newly_placed_list_details, new_tiles_sig, temp_tiles, temp_blanks
    cdef object move_blanks_coords, newly_placed_coords, all_words_formed_details
    cdef object primary_word_tiles, primary_word_str, start_pos, orientation
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

    # --- Explicitly check the last element before unpacking ---
    last_elem = current_word_tiles[-1]
    if isinstance(last_elem, (list, tuple)) and len(last_elem) == 5:
        r_last_py, c_last_py, _ign1, _ign2, _ign3 = last_elem
        # --- Assign to C integers ---
        r_last = <int>r_last_py # Cast Python object to C int
        c_last = <int>c_last_py # Cast Python object to C int
    else:
        raise TypeError(f"Expected 5-tuple at end of current_word_tiles, got {type(last_elem)}: {last_elem}")
    # --- End Explicit Check ---


    # --- Check if current path forms a valid move ---
    if gaddag_node.is_terminal and not is_reversed:
        # --- FIX: Replace list comprehension ---
        newly_placed_list_details = []
        for r_lp, c_lp, l_lp, _, is_new_lp in current_word_tiles:
            if is_new_lp:
                newly_placed_list_details.append((r_lp, c_lp, l_lp))
        # --- End Fix ---
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

                # --- CALL LOCAL CYTHON HELPER (now cpdef) ---
                # The is_valid, is_bingo variables assigned here are now typed as bint
                is_valid, is_bingo = is_valid_play(newly_placed_list_details, temp_tiles, is_first_play, full_rack_size, original_tiles_state, None)
                # --- END CALL ---

                if is_valid:
                    unique_move_signatures.add(new_tiles_sig)
                    # --- CALL LOCAL CYTHON HELPER (now cpdef) ---
                    score = calculate_score(newly_placed_list_details, board, temp_tiles, temp_blanks)
                    # --- END CALL ---
                    # --- CALL LOCAL CYTHON HELPER (now cpdef) ---
                    all_words_formed_details = find_all_words_formed(newly_placed_list_details, temp_tiles)
                    # --- END CALL ---

                    primary_word_tiles = []; primary_word_str = ""; start_pos = (0, 0); orientation = current_axis
                    # ... (rest of primary word finding logic - unchanged) ...
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
                             start_pos = (primary_word_tiles[0][0], primary_word_tiles[0][1])

                    word_with_blanks_list = []
                    for wr, wc, w_letter in primary_word_tiles:
                        is_blank_in_word = (wr, wc) in newly_placed_coords and (wr, wc) in move_blanks_coords
                        word_with_blanks_list.append(w_letter.lower() if is_blank_in_word else w_letter.upper())
                    word_with_blanks = "".join(word_with_blanks_list)

                    # Use C array for leave calculation
                    leave = []
                    for i in range(26):
                        leave.extend([chr(ord('A') + i)] * rack_counts_c[i])
                    leave.extend([' '] * rack_counts_c[26])

                    move_details_dict = {
                        'positions': [(t[0], t[1], t[2]) for t in primary_word_tiles], 'blanks': move_blanks_coords,
                        'word': primary_word_str, 'score': score, 'start': start_pos, 'direction': orientation,
                        'leave': leave, 'is_bingo': is_bingo, 'word_with_blanks': word_with_blanks,
                        'newly_placed': newly_placed_list_details
                    }
                    all_found_moves.append(move_details_dict)

    # --- Explore Next Steps ---
    current_rack_counts_c = rack_counts_c
    blank_idx = 26

    # The 'letter' variable in the loop will now use the 'cdef str' type
    # The 'next_node' variable will use the 'cdef object' type
    for letter, next_node in gaddag_node.children.items():
        if letter == Gaddag.SEPARATOR:
            if is_reversed:
                # Pass the existing memory view
                _gaddag_traverse(
                    anchor_pos, current_rack_counts_c, tiles, board, blanks, cross_check_sets,
                    next_node,
                    gaddag_root_node, # Pass the root node through
                    current_word_tiles, False, current_axis, # Pass same list
                    all_found_moves, unique_move_signatures, original_tiles_state,
                    is_first_play, full_rack_size, max_len, depth + 1
                )
            continue

        next_r, next_c = -1, -1
        # --- Use C integers for coordinate calculation ---
        if is_reversed:
            # Use C integers r_last, c_last directly
            if current_axis == 'H': next_r, next_c = r_last, c_last - 1
            else:                   next_r, next_c = r_last - 1, c_last
        else:
            # Unpack anchor_pos tuple into Python objects first
            anchor_r_obj, anchor_c_obj = anchor_pos
            # Cast Python objects to C integers
            anchor_r = <int>anchor_r_obj
            anchor_c = <int>anchor_c_obj

            # Check for tile on anchor (still uses Python objects for iteration)
            tile_on_anchor = None
            for t_r, t_c, t_l, t_b, t_n in current_word_tiles:
                 if t_r == anchor_r and t_c == anchor_c:
                      tile_on_anchor = (t_r, t_c); break

            # Determine if separator was just crossed using C integers
            # The just_crossed_separator variable is now typed as bint
            just_crossed_separator = False
            if len(current_word_tiles) > 0:
                if current_axis == 'H':
                    if c_last <= anchor_c: just_crossed_separator = True
                else:
                    if r_last <= anchor_r: just_crossed_separator = True

            # Determine reference point using C integers
            ref_r = anchor_r if just_crossed_separator else r_last
            ref_c = anchor_c if just_crossed_separator else c_last

            # Calculate next position using C integers
            if current_axis == 'H': next_r, next_c = ref_r, ref_c + 1
            else:                   next_r, next_c = ref_r + 1, ref_c
        # --- End coordinate calculation modification ---


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
                # Create and modify temporary C array
                for i in range(27):
                    temp_rack_counts_c_arr[i] = current_rack_counts_c[i]
                temp_rack_counts_c_arr[letter_idx] -= 1
                # Create NumPy array from C array
                next_rack_np_arr = np.array(temp_rack_counts_c_arr, dtype=np.intc)
                # Pass the NumPy array
                _gaddag_traverse(
                    anchor_pos, next_rack_np_arr, tiles, board, blanks, cross_check_sets,
                    next_node,
                    gaddag_root_node, # Pass the root node through
                    current_word_tiles + [(next_r, next_c, letter, <bint>False, <bint>True)],
                    is_reversed, current_axis, all_found_moves, unique_move_signatures,
                    original_tiles_state, is_first_play, full_rack_size, max_len, depth + 1
                )

            # Option 1b: Use blank tile
            if current_rack_counts_c[blank_idx] > 0 and ' ' in allowed_letters:
                # Create and modify temporary C array
                for i in range(27):
                    temp_rack_counts_c_arr[i] = current_rack_counts_c[i]
                temp_rack_counts_c_arr[blank_idx] -= 1
                # Create NumPy array from C array
                next_rack_np_arr = np.array(temp_rack_counts_c_arr, dtype=np.intc)
                # Pass the NumPy array
                _gaddag_traverse(
                    anchor_pos, next_rack_np_arr, tiles, board, blanks, cross_check_sets,
                    next_node,
                    gaddag_root_node, # Pass the root node through
                    current_word_tiles + [(next_r, next_c, letter, <bint>True, <bint>True)],
                    is_reversed, current_axis, all_found_moves, unique_move_signatures,
                    original_tiles_state, is_first_play, full_rack_size, max_len, depth + 1
                )
        # Case 2: Square has matching existing tile
        elif existing_tile == letter:
            # Pass the existing memory view (no change needed here)
            _gaddag_traverse(
                anchor_pos, current_rack_counts_c, tiles, board, blanks, cross_check_sets,
                next_node,
                gaddag_root_node, # Pass the root node through
                current_word_tiles + [(next_r, next_c, letter, <bint>False, <bint>False)],
                is_reversed, current_axis, all_found_moves, unique_move_signatures,
                original_tiles_state, is_first_play, full_rack_size, max_len, depth + 1
            )


# ==============================================================================
# === HELPER FUNCTIONS (Use cpdef for Python visibility) ===
# ==============================================================================

# --- find_cross_word ---
# Keep as cdef (only called by find_all_words_formed)
cdef list find_cross_word(tuple tile, list tiles, str main_orientation):
    """Finds a cross word formed by a single tile perpendicular to the main word."""
    # --- MODIFICATION: Add cdef int declarations ---
    cdef int r, c, min_row, max_row, min_col, max_col, rr_cw, cc_cw
    # --- END MODIFICATION ---
    cdef list cross_word = []
    # Unpack tuple - r, c will be implicitly converted if needed, but are typed now
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
            # rr_cw is now cdef int
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
            # cc_cw is now cdef int
            for cc_cw in range(min_col, max_col + 1):
                if tiles[r][cc_cw]:
                    cross_word.append((r, cc_cw, tiles[r][cc_cw]))

    return cross_word if len(cross_word) > 1 else []

# --- find_main_word ---
# Keep as cdef (only called by find_all_words_formed)
cdef tuple find_main_word(list new_tiles, list tiles):
    """Finds the primary word formed by newly placed tiles."""
    # --- MODIFICATION: Add cdef int declarations ---
    cdef int row, col, min_row, max_row, min_col, max_col, r_nt, c_nt, c_mw, r_mw
    # --- END MODIFICATION ---
    cdef list main_word
    cdef str orientation
    # --- ADDED: Intermediate Python objects for unpacking ---
    cdef object r_nt_obj, c_nt_obj, _ign1_nt

    if not new_tiles: return [], None
    rows_set = set()
    cols_set = set()
    # r_nt, c_nt are now cdef int
    for tile_tuple in new_tiles: # Iterate through tuples
        if isinstance(tile_tuple, tuple) and len(tile_tuple) >= 3:
            r_nt_obj, c_nt_obj, _ign1_nt = tile_tuple[:3] # Unpack first 3
            r_nt = <int>r_nt_obj
            c_nt = <int>c_nt_obj
            rows_set.add(r_nt)
            cols_set.add(c_nt)
        else:
            # Handle error or skip invalid entry if necessary
            pass

    if len(rows_set) == 1:
        orientation = "horizontal"; row = rows_set.pop()
        min_col = 999
        max_col = -1
        # r_nt, c_nt are now cdef int
        for tile_tuple in new_tiles: # Iterate again
            if isinstance(tile_tuple, tuple) and len(tile_tuple) >= 3:
                r_nt_obj, c_nt_obj, _ign1_nt = tile_tuple[:3]
                r_nt = <int>r_nt_obj
                c_nt = <int>c_nt_obj
                if r_nt == row:
                    if c_nt < min_col: min_col = c_nt
                    if c_nt > max_col: max_col = c_nt
        while min_col > 0 and tiles[row][min_col - 1]: min_col -= 1
        while max_col < GRID_SIZE_C - 1 and tiles[row][max_col + 1]: max_col += 1
        main_word = []
        # c_mw is now cdef int
        for c_mw in range(min_col, max_col + 1):
            if tiles[row][c_mw]:
                main_word.append((row, c_mw, tiles[row][c_mw]))
        return (main_word, orientation) if len(main_word) > 1 else ([], None)
    elif len(cols_set) == 1:
        orientation = "vertical"; col = cols_set.pop()
        min_row = 999
        max_row = -1
        # r_nt, c_nt are now cdef int
        for tile_tuple in new_tiles: # Iterate again
            if isinstance(tile_tuple, tuple) and len(tile_tuple) >= 3:
                r_nt_obj, c_nt_obj, _ign1_nt = tile_tuple[:3]
                r_nt = <int>r_nt_obj
                c_nt = <int>c_nt_obj
                if c_nt == col:
                    if r_nt < min_row: min_row = r_nt
                    if r_nt > max_row: max_row = r_nt
        while min_row > 0 and tiles[min_row - 1][col]: min_row -= 1
        while max_row < GRID_SIZE_C - 1 and tiles[max_row + 1][col]: max_row += 1
        main_word = []
        # r_mw is now cdef int
        for r_mw in range(min_row, max_row + 1):
            if tiles[r_mw][col]:
                main_word.append((r_mw, col, tiles[r_mw][col]))
        return (main_word, orientation) if len(main_word) > 1 else ([], None)
    else: return [], None

# --- find_all_words_formed ---
# Use cpdef (called from Python)
cpdef list find_all_words_formed(list new_tiles, list tiles):
    """Finds all words (main and cross) formed by a play."""
    cdef list words = []
    cdef set new_positions_set
    cdef list main_word_tiles, cross_word, unique_word_tile_lists
    cdef str orientation
    cdef tuple tile, signature
    cdef set seen_signatures

    if not new_tiles: return words
    new_positions_set = set()
    for r_np, c_np, _ in new_tiles:
        new_positions_set.add((r_np, c_np))

    main_word_tiles, orientation = find_main_word(new_tiles, tiles) # Call local cdef function

    if main_word_tiles:
        words.append(main_word_tiles)
        for tile in new_tiles:
            if (tile[0], tile[1]) in new_positions_set:
                cross_word = find_cross_word(tile, tiles, orientation) # Call local cdef function
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
# Use cpdef (called from Python)
cpdef int calculate_score(list new_tiles, list board, list tiles, set blanks):
    """Calculates the score for a play based on newly placed tiles."""
    cdef int total_score = 0, word_score, word_multiplier
    cdef int r, c, letter_value, letter_multiplier
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


# --- is_valid_play ---
# Use cpdef (called from Python)
cpdef tuple is_valid_play(list word_positions, list tiles, bint is_first_play, int initial_rack_size, list original_tiles, object rack):
    """Validate a potential play against game rules and dictionary."""
    cdef set newly_placed_positions_coords
    cdef list rows_list, cols_list, all_words_details, formed_word_strings
    cdef bint is_horizontal, is_vertical, connects, is_bingo
    cdef int r, c, dr, dc, nr, nc
    cdef int min_col, max_col, temp_min_col, temp_max_col
    cdef int min_row, max_row, temp_min_row, temp_max_row
    cdef int tiles_played_from_rack
    cdef str word
    cdef bint dawg_search_result
    # --- Local sets for building ---
    cdef set rows_set_local, cols_set_local

    if not word_positions:
        return False, False

    # --- FIX: Replace set comprehension ---
    newly_placed_positions_coords = set()
    for r_np, c_np, _ in word_positions:
        newly_placed_positions_coords.add((r_np, c_np))
    # --- End Fix ---
    if not newly_placed_positions_coords:
        return False, False

    # --- FIX: Replace set/list comprehensions ---
    rows_set_local = set()
    cols_set_local = set()
    for r_wp, c_wp, _ in word_positions:
        rows_set_local.add(r_wp)
        cols_set_local.add(c_wp)
    rows_list = sorted(list(rows_set_local))
    cols_list = sorted(list(cols_set_local))
    # --- End Fix ---

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

    if not all_words_details and len(word_positions) > 1:
         return False, False

    # --- FIX: Replace list comprehension ---
    formed_word_strings = []
    for word_detail in all_words_details:
        # --- FIX: Replace generator expression (inner join) ---
        word_chars = []
        for tile_detail in word_detail:
            word_chars.append(tile_detail[2])
        word_str = "".join(word_chars)
        # --- End Fix ---
        formed_word_strings.append(word_str)
    # --- End Fix ---
    if not formed_word_strings and len(word_positions) > 1:
         return False, False

    for word in formed_word_strings:
        dawg_search_result = DAWG.search(word)
        if not dawg_search_result:
            return False, False

    # --- Connection Rules Check ---
    if is_first_play:
        if CENTER_SQUARE not in newly_placed_positions_coords:
            return False, False
    else:
        connects = False
        if original_tiles is None:
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
