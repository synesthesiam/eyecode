import os, sys
import numpy as np
import pandas

from ..util import just2, window
from lxml import builder as lb

# Constants {{{

SYNTAX_CATEGORIES = {
    "Token.Keyword": "keyword",
    "Token.Literal.Number.Integer": "literal",
    "Token.Literal.String": "literal",
    "Token.Name": "identifier",
    "Token.Name.Builtin.Pseudo": "identifier",
    "Token.Name.Class": "identifier",
    "Token.Name.Function": "identifier",
    "Token.Operator": "operator",
    "Token.Operator.Word": "operator",
    "Token.Punctuation": "punctuation",
    "Token.Text": "whitespace",
    "Token.Text.Indentation": "indentation"
}

# }}}

# Utility Methods {{{

def kind_to_col(kind):
    """Converts an AOI kind to a column name.

    Parameters
    ----------
    kind : str
        AOI kind

    Returns
    -------
    str
        Column name for AOI kind
    """
    return "aoi_{0}".format(kind)

def kinds_to_cols(kinds):
    """Converts a list of AOI kinds to column names.

    Parameters
    ----------
    kind : list of str
        AOI kinds

    Returns
    -------
    list of str
        Column names for AOI kinds
    """
    return ["aoi_{0}".format(k) for k in kinds]

def get_aoi_columns(fixations):
    """Gets all columns in a dataframe that hold AOI names.

    Parameters
    ----------
    fixations : pandas DataFrame
        A dataframe with a row for each fixation

    Returns
    -------
    list of str
        Column names that correspond to AOI kinds
    """
    return [c for c in fixations.columns
            if c.startswith("aoi_")]

def get_aoi_kinds(fixations):
    """Gets all AOI kinds in a dataframe.

    Parameters
    ----------
    fixations : pandas DataFrame
        A dataframe with a row for each fixation

    Returns
    -------
    list of str
        AOI kinds in the dataframe
    """
    return [c.split("_", 1)[1] for c in get_aoi_columns(fixations)]

def col_to_kind(column):
    return column.split("_", 1)[1]

def envelope(aois, padding=0):
    """Returns a rectangle that envelopes the given AOI rectangles.
    
    Parameters
    ----------
    aois : pandas DataFrame
        A dataframe with a row for each AOI (x, y, width, height)

    Returns
    -------
    bbox : list of int
        Bounding box around all aois (x, y, width, height)
    
    """
    x1, y1 = sys.maxint, sys.maxint
    x2, y2 = 0, 0

    for x, y, w, h in aois[["x", "y", "width", "height"]].values:
        x1, y1 = min(x1, x), min(y1, y)
        x2, y2 = max(x + w, x2), max(y + h, y2)

    return [x1 - (padding/2), y1 - (padding/2),
            x2 - x1 + padding,
            y2 - y1 + padding]

def pad(aois, padding):
    """Pads the given AOIs.

    Parameters
    ----------
    aois : pandas DataFrame
        A dataframe with a row for each AOI (x, y, width, height)

    padding : int or list of int
        Uniform padding (int) or top, right, bottom, left (list of int)

    Returns
    -------
    padded_aois : pandas DataFrame
        A copy of the input aois with padding applied

    """
    top, right, bottom, left = (0, 0, 0, 0)

    if isinstance(padding, list) or isinstance(padding, tuple):
        assert len(padding) == 4, "Padding must be (top, right, bottom, left)"
        top, right, bottom, left = padding
    else:
        top, right, bottom, left = [padding] * 4

    aois = aois.copy()
    aois.x -= left
    aois.width += (left + right)
    aois.y -= top
    aois.height += (top + bottom)

    return aois

def add_bbox(aois, bbox, kind, name):
    """Adds a new AOI with the given bounding box, kind, and name.
    
    Parameters
    ----------
    aois : pandas DataFrame
        A dataframe with a row for each AOI (x, y, width, height)
    bbox : list of int
        Bounding box of new AOI (x, y, width, height)
    kind : str
        New AOI kind
    name : str
        New AOI name

    Returns
    -------
    more_aois : pandas DataFrame
        A copy of the input aois with the new AOI appended
    
    """
    return aois.append({
        "kind"   : kind,
        "name"   : name,
        "x"      : bbox[0],
        "y"      : bbox[1],
        "width"  : bbox[2],
        "height" : bbox[3]
    }, ignore_index=True)

def only_lines(aois, lines, kind="line", fmt="line {0}"):
    """Returns line-based AOIs that match the given line numbers."""
    if isinstance(lines, int):
        lines = [lines]
    names = [fmt.format(l) for l in lines]
    return aois[(aois.kind == "line") &
                aois.name.isin(names)]

def fixations_to_json(fixations):
    """Converts fixations to JSON (with hit AOI names)."""
    import json
    aoi_kinds = get_aoi_kinds(fixations)
    aoi_cols = get_aoi_columns(fixations)
    fixations[aoi_cols] = fixations[aoi_cols].fillna("")
    js_fixes = []
    fix_cols = ["fix_x", "fix_y", "start_ms", "end_ms"] + aoi_cols
    for x, y, start, end, aoi_names in just2(5, fixations[fix_cols].values):
        hit_names = { k : n for k, n in zip(aoi_kinds, aoi_names)
                      if len(n) > 0 }

        js_fixes.append({
            "start"      : int(start),
            "end"        : int(end),
            "x"          : int(x),
            "y"          : int(y),
            "hit_names"  : hit_names
        })

    return json.dumps(js_fixes)

def aois_to_json(aois):
    """Converts AOI rectangles to JSON."""
    import json
    js_aois = []
    cols = ["kind", "name", "x", "y", "width", "height", "program"]
    for kind, name, x, y, w, h, prog in aois[cols].values:
        js_aois.append({
            "kind"    : kind,
            "name"    : name,
            "x"       : x,
            "y"       : y,
            "width"   : w,
            "height"  : h,
            "program" : prog,
        })

    return json.dumps(js_aois)

# }}}

# Scanpath Methods {{{

def scanpath_from_fixations(fixations, aoi_names=None, mixed=False,
        repeats=True, name_map=None):
    """Generates one or more scanpaths (sequences of fixated AOIs) from fixations.

    Parameters
    ----------
    fixations : pandas DataFrame
        A dataframe with one row per fixation
    aoi_names : dict or None, optional
        Optional dictionary mapping AOI kinds to lists of AOI names.  If None,
        all AOI kinds and names in fixations will be included in the scanpath.
        If specified, only the given kinds (keys) and names (values) will be
        included. An empty list or None for a value will include all names for
        the AOI kind.
    mixed : bool, optional
        If True, a single scanpath with mixed AOI kinds will be generated.
        If False, separate scanpaths for each AOI kind are generated.
        Default is False (multiple scanpaths).
    repeats : bool
        If True, repeated AOI names in scanpaths will be removed (default:
        True).
    name_map : dict or None, optional
        Optional dictionary mapping AOI (kind, name) tuples to unique names
        (across all AOI kinds). This is required only if mixed is True and
        there are overlapping names between kinds.

    Returns
    -------
    pandas DataFrame or dict
        If mixed is True or there is only a single AOI kind, a dataframe is
        returned with the sequence of fixated AOI names (indexed by start
        time). Otherwise, a dictionary is returned with AOI kinds for keys and
        dataframe scanpaths for values.

    Examples
    --------
    >>> from eyecode import aoi
    >>> rects = aoi.make_grid(2, 2, "ABCD", width=50)
    >>> fixes = aoi.fixations_from_scanpath("AABACDD", rects)
    >>> print fixes
       fix_x  fix_y  start_ms  duration_ms aoi_all
    0     25     25         0          200       A
    1     25     25       220          200       A
    2     75     25       440          200       B
    3     25     25       660          200       A
    4     25     75       880          200       C
    5     75     75      1100          200       D
    6     75     75      1320          200       D

    >>> sp = aoi.scanpath_from_fixations(fixes)
    >>> print sp

    start_ms
    0           A
    220         A
    440         B
    660         A
    880         C
    1100        D
    1320        D
    Name: aoi_all, dtype: object

    >>> sp_no_repeats = aoi.scanpath_from_fixations(fixes, repeats=False)
    >>> print sp_no_repeats

    start_ms
    0           A
    440         B
    660         A
    880         C
    1100        D
    Name: aoi_all, dtype: object

    """
    kinds = get_aoi_kinds(fixations) if aoi_names is None else aoi_names.keys()
    columns = kinds_to_cols(kinds)

    # Re-index by start time so scanpaths will retain it
    fixations = fixations.set_index("start_ms", drop=False)

    # Drop missing values and separate by kind
    aoi_fixes = { k: fixations[c].dropna() for (c, k) in zip(columns, kinds) }

    if aoi_names is None:
        aoi_names = { k: None for k in aoi_fixes.keys() }

    # Create dictionary of aoi names
    for kind, names in aoi_names.iteritems():
        if names is None or len(names) == 0:
            # Fill in missing aoi names
            col = kind_to_col(kind)
            aoi_names[kind] = sorted(aoi_fixes[kind].unique())
        else:
            # Filter out unwanted names
            f = aoi_fixes[kind]
            aoi_fixes[kind] = f[f.isin(names)]

    # Generate AOI name map if not provided (use AOI name directly)
    if name_map is None:
        name_map = {}
        for kind, names in aoi_names.iteritems():
            for name in names:
                name_map[(kind, name)] = name

    # Verify that there are enough unique names
    if mixed:
        total_names = len(name_map.values())
        unique_names = len(set(name_map.values))
        assert total_names >= unique_names, "AOI names overlap between kinds."\
                " You must specify a name map or set mixed to False."

    scanpaths = {}

    # Create separate scanpath for each kind
    for col, kind in zip(columns, kinds):
        scanpaths[kind] = aoi_fixes[kind].apply(lambda name: name_map[(kind, name)])

    if mixed:
        # Create a single scanpath with mixed kinds (assumed to be disjoint in time)
        mixed_names = pandas.concat(scanpaths.values()).sort_index()
        scanpaths = { "__mixed__": mixed_names }

    # Remove repeats if requested
    if not repeats:
        for kind, sp in scanpaths.iteritems():
            keep = np.ones(len(sp)).astype(bool)
            last = None
            for i, v in enumerate(sp):
                if (last is not None) and (v == last):
                    keep[i] = False
                last = v
            scanpaths[kind] = sp[keep]

    if len(scanpaths) == 1:
        return scanpaths.values()[0]
    else:
        return scanpaths

def scanpath_edit_distance(path_1, path_2, norm=True):
    from nltk.metrics import edit_distance
    distance = edit_distance(path_1, path_2)

    if norm:
        distance = distance / float(max(len(path_1), len(path_2)))

    return distance

def fixations_from_scanpath(scanpath, aoi_rectangles, duration_ms=200,
        saccade_ms=20, aoi_kinds="all", point_fun="center"):
    """Generates a dataframe of fixations from the given scanpath.

    Parameters
    ----------
    scanpath : list
        Sequence of fixated AOI names
    aoi_rectangles : dict
        Dictionary mapping AOI names (str) to rectangle tuples (x, y, width, height)
    duration_ms : int
        Duration of every fixation generated (milliseconds)
    saccade_ms : int
        Delay between every fixation generated (milliseconds)
    aoi_kinds : str or dict
        If a string, this will be the kind of every AOI. If a dictionary, keys
        should be AOI names and values should be AOI kinds.
    point_fun : "center" or callable
        If "center", fixations will occur in the center of each rectangle.
        Otherwise, point_fun will be called with the name of the AOI and the
        rectangle tuple (x, y, width, height). An (x, y) tuple must be
        returned. 

    Returns
    -------
    pandas DataFrame
        A dataframe with a fixation for each item in the scanpath

    See Also
    --------
    make_grid : Create AOI rectangles in a grid

    Examples
    --------
    >>> from eyecode import aoi
    >>> rects = aoi.make_grid(2, 2, "ABCD", width=50)
    >>> fixes = aoi.fixations_from_scanpath("AABACDD", rects)
    >>> print fixes
       fix_x  fix_y  start_ms  duration_ms aoi_all
    0     25     25         0          200       A
    1     25     25       220          200       A
    2     75     25       440          200       B
    3     25     25       660          200       A
    4     25     75       880          200       C
    5     75     75      1100          200       D
    6     75     75      1320          200       D


    >>> from eyecode import aoi
    >>> def upper_left(s, rect):
    >>>    return rect[0], rect[1]
    >>> rects = aoi.make_grid(2, 2, "ABCD", width=50)
    >>> fixes = aoi.fixations_from_scanpath("AABACDD", rects, point_fun=upper_left)
    >>> print fixes

       fix_x  fix_y  start_ms  duration_ms aoi_all
    0      0      0         0          200       A
    1      0      0       220          200       A
    2     50      0       440          200       B
    3      0      0       660          200       A
    4      0     50       880          200       C
    5     50     50      1100          200       D
    6     50     50      1320          200       D

    """
    aoi_names = sorted(aoi_rectangles.keys())

    if isinstance(aoi_kinds, str):
        kind = aoi_kinds
        aoi_kinds = { n : kind for n in aoi_names }

    sorted_kinds = sorted(set(aoi_kinds.values()))
    aoi_cols = kinds_to_cols(sorted_kinds)
    rows = []
    time = 0

    for s in scanpath:
        kind = aoi_kinds[s]
        rect = aoi_rectangles[s]
        x, y = 0, 0

        if point_fun == "center":
            x = rect[0] + (rect[2] / 2.0)
            y = rect[1] + (rect[3] / 2.0)
        else:
            x, y = point_fun(s, rect)

        aoi_vals = [np.NaN] * len(aoi_cols)
        kind_idx = sorted_kinds.index(kind)
        aoi_vals[kind_idx] = s
        rows.append([int(x), int(y), time, duration_ms] + aoi_vals)
        time += (duration_ms + saccade_ms)

    fix_cols = ["fix_x", "fix_y", "start_ms", "duration_ms"]
    fixations = pandas.DataFrame(rows, columns=fix_cols + aoi_cols)
    return fixations

def transition_matrix(scanpath, shape=None, norm=True):
    if shape is None:
        max_aoi = scanpath.max()
        shape = (max_aoi + 1, max_aoi + 1)

    trans_counts = np.zeros(shape)
    for i, j in zip(scanpath, scanpath[1:]):
        trans_counts[i, j] += 1

    if norm:
        # Normalize by rows
        row_sums = trans_counts.sum(axis=1)
        trans_probs = trans_counts / row_sums.reshape((-1, 1))
    
        # Get rid of NaNs
        return np.nan_to_num(trans_probs)
    else:
        return trans_counts

# }}}

# AOI Creation {{{

def find_rectangles(screen_image, black_thresh=255, white_row_thresh=3,
        white_col_thresh=3, vert_kind="line", horz_kind="sub-line"):
    """Scans a black and white code image for line and sub-line rectangles.

    Parameters
    ----------
    screen_image : PIL.Image
        Image with code (will be converted to 'L' mode)
    black_thresh : int, optional
        Luminescence threshold for deciding a pixel is black (default: 255)       
    white_row_thresh : int, optional
        Number of white rows before deciding a rectangle is done (default: 3)
    vert_kind : str, optional
        AOI kind to assign to all vertical rectangles (default: line)

    horz_kind : str, optional
        AOI kind to assign to all horizontal rectangles (default: sub-line)

    Returns
    -------
    pandas DataFrame
        A dataframe with rectangle coordinates and sizes

    Examples
    --------
    >>> from eyecode import aoi, data
    >>> code_img = data.busjahn_2013.program_image("basketball")
    >>> code_aois = aoi.find_rectangles(code_img)
    >>> print code_aois[:3]
           kind           name    x   y  width  height
    0      line         line 1  335  28    212      20
    1  sub-line  line 1 part 1  335  28     53      20
    2  sub-line  line 1 part 2  392  28     47      20

    See Also
    --------
    eyecode.plot.aoi.draw_rectangles: Visualize AOI rectangles

    """
    start_y, end_y = None, 0
    left_x, right_x = 0, 0
    white_rows = 0
    rects = []
    rect_i = 1

    # Scan down each 1-pixel line of the image, looking for black pixels (lower
    # than black_thresh). If found, start a rectangle. If more than
    # white_row_thresh lines are white in a row, end the block where the first
    # white row was found.
    img_data = np.array(screen_image.convert("L"))
    for y in range(img_data.shape[0]):
        line = img_data[y, :]
        blacks = line < black_thresh
        num_blacks = sum(blacks)

        if num_blacks > 1 and start_y is None:
            # Start of block
            start_y = y
            left_x = np.argwhere(blacks).min()
            right_x = np.argwhere(blacks).max()
        elif num_blacks > 1:
            # Block continues, update left/right boundaries
            left_x = min(left_x, np.argwhere(blacks).min())
            right_x = max(right_x, np.argwhere(blacks).max())
            white_rows = 0
            end_y = 0
        elif num_blacks == 0 and (start_y is not None):
            # Block may be ending
            if white_rows == 0:
                # Record potential ending
                end_y = y

            white_rows += 1
            if white_rows > white_row_thresh:
                # Block ends; record x, y, width, height
                name = "line {0}".format(rect_i)

                rects.append([vert_kind, name, left_x, start_y,
                    right_x - left_x + 1, end_y - start_y + 1])

                # ------------------------------------------------------------
                # Scan right on each 1-pixel column of the line, looking for
                # black pixels (lower than black_thresh). If found, start a
                # rectangle. If more than white_col_thresh lines are white in a
                # row, end the block where the first white column was found.
                part_i = 1
                white_cols = 0
                start_x, end_x = None, 0
                x_range_end = min(img_data.shape[1], right_x + white_col_thresh + 2)
                for x in range(left_x, x_range_end):
                    col = img_data[start_y:end_y + 1, x]
                    col_blacks = sum(col < black_thresh)
                    if col_blacks > 1 and start_x is None:
                        start_x = x
                    elif col_blacks > 1:
                        white_cols = 0
                    elif col_blacks == 0 and (start_x is not None):
                        if white_cols == 0:
                            end_x = x
                        white_cols += 1
                        if white_cols > white_col_thresh:
                            name = "line {0} part {1}".format(rect_i, part_i)
                            part_i += 1

                            # Block ends; record x, y, width, height
                            rects.append([horz_kind, name, start_x, start_y,
                                end_x - start_x + 1, end_y - start_y + 1])

                            # Reset horizontal
                            start_x, end_x = None, 0
                            white_cols = 0

                # ------------------------------------------------------------

                # Reset vertical
                start_y, end_y = None, 0
                left_x, right_x = 0, 0
                white_rows = 0
                rect_i += 1

    # Convert to data frame
    rect_df = pandas.DataFrame(rects, columns=["kind", "name",
        "x", "y", "width", "height"])

    return rect_df


def make_grid(rows, cols, names, width=100, height=None):
    """Creates a grid of equally-sized AOIs rectangles."""
    if height is None:
        height = width

    rects = {}
    name_idx = 0
    for r in range(rows):
        y = r * height
        for c in range(cols):
            x = c * width
            name = names[name_idx]
            rects[name] = (x, y, width, height)
            name_idx += 1

    return rects

def make_code_aois(code_file, font_size=(14, 25), line_offset=5,
        syntax_categories=SYNTAX_CATEGORIES):
    """Creates block, line, and syntax AOI rectangles from Python source code."""

    aoi_df = pandas.DataFrame(columns=("kind", "name",
        "x", "y", "width", "height", "note"))

    # Needed for syntax-based AOIs
    from pygments.lexers import PythonLexer
    lexer = PythonLexer()

    # Parse the file and generate AOIs
    if isinstance(code_file, str):
        code_file = open(code_file, "r")

    code_lines = code_file.readlines()
    code_str = "".join(code_lines)

    # Add extra newline token to trigger last AOI block
    code_lines += [""]
    tokens = list(lexer.get_tokens(code_str, unfiltered=True)) + [("Token.Text", u"\n")]

    col = 0            # Current column
    line = 0           # Current line number
    block_start = 0    # Current whitespace separated block
    last_blank = False # Was last line blank?
    block_lines = []   # Lines in current block

    for t in tokens:
        kind = str(t[0])
        val = t[1]

        # Check if end of line
        if val == u"\n":
            line_str = code_lines[line].rstrip()
            if len(line_str.strip()) > 0:
                # Non-blank line: add AOI for whole line
                aoi_df = aoi_df.append({
                    "kind"     : "line",
                    "name"     : "line {0}".format(line + 1),
                    "x"        : 0,
                    "y"        : (line * font_size[1]) + (line * line_offset) - (line_offset / 2),
                    "width"    : len(line_str) * font_size[0],
                    "height"   : font_size[1] + line_offset - 1,
                    "note"     : line_str
                }, ignore_index=True)

                # Add to current block
                last_blank = False
                block_lines.append(line_str)
            else:
                # Blank line
                if not last_blank:
                    # Add AOI for whitespace separated block of lines
                    aoi_df = aoi_df.append({
                        "kind"     : "block",
                        "name"     : "lines {0}-{1}".format(block_start + 1, line + 1),
                        "x"        : 0,
                        "y"        : (block_start * font_size[1]) + (block_start * line_offset) - (line_offset / 2),
                        "width"    : max([len(l) for l in block_lines]) * font_size[0],
                        "height"   : len(block_lines) * (font_size[1] + line_offset),
                        "note"     : "\n".join(block_lines)
                    }, ignore_index=True)

                # Reset block variables
                last_blank = True
                block_lines = []
                block_start = line + 1

            # Next line
            col = 0
            line += 1
            continue

        # Add AOI for syntax token
        if (kind == "Token.Text") and (col == 0):
            kind += ".Indentation"

        aoi_df = aoi_df.append({
            "kind"     : "syntax",
            "name"     : syntax_categories[kind],
            "x"        : col * font_size[0],
            "y"        : (line * font_size[1]) + (line * line_offset) - (line_offset / 2),
            "width"    : len(val) * font_size[0],
            "height"   : font_size[1] + line_offset - 1,
            "note"     : val
        }, ignore_index=True)

        col += len(val)

    return aoi_df

def make_code_aois_from_files(code_paths, **kwargs):
    """Creates block, line, and syntax AOI rectangles from the
    given Python files."""
    aois_df = None

    for path in code_paths:
        with open(path, "r") as code_file:
            df = make_code_aois(code_file, **kwargs)
            df["file_name"] = os.path.split(path)[1]

            if aois_df is None:
                aois_df = df
            else:
                aois_df = pandas.concat([aois_df, df], ignore_index=True)

    return aois_df

# }}}

# Hit Testing {{{

def make_polygon(aoi_row):
    """Converts an AOI rectangle to a shapely box."""
    from shapely.geometry import box
    x = int(aoi_row["x"])
    y = int(aoi_row["y"])
    width = int(aoi_row["width"])
    height = int(aoi_row["height"])
    return box(x, y, x + width, y + height)

def hit_point(fix_pt, aoi_polys, **kwargs):
    """Returns the first polygon that contains the fixation Point."""
    for aoi, poly in aoi_polys.iteritems():
        if poly.intersects(fix_pt):
            return aoi
    return None

def hit_circle(fix_pt, aoi_polys, radius=1, **kwargs):
    """Returns the polygon that has the most overlap with
    the fixation circle."""
    fix_circle = fix_pt.buffer(radius)
    best_aoi = None
    best_area = 0

    for aoi, poly in aoi_polys.iteritems():
        if poly.intersects(fix_circle):
            area = poly.intersection(fix_circle).area
            if area > best_area:
                best_aoi = aoi
                best_area = area
    return best_aoi

def hit_test(fixations, aois, offsets=None, hit_fun=hit_circle,
        hit_radius=20, **kwargs):
    """Hit tests fixations against AOI rectangles.

    Parameters
    ----------
    fixations : pandas DataFrame
        A DataFrame with fixations to hit test (fix_x, fix_y)
    aois : pandas DataFrame
        A DataFrame with areas of interest (kind, name, x, y, width, height)
    offsets : pandas DataFrame or None
        A DataFrame with different fixations offsets to apply (name, x, y).
        If None, no offset is applied
    hit_fun : callable
        Hit testing function. See hit_point and hit_circle for examples
    hit_radius : int
        Fixation circle radius for hit_circle

    Returns
    -------
    aoi_fixations : pandas DataFrame
        A copy of the fixations DataFrame with additional columns for each
        offset and AOI kind

    Notes
    -----
    Requires the shapely library: http://toblerity.org/shapely

    Examples
    --------
    >>> from eyecode import aoi, data
    >>> code_img = data.busjahn_2013.program_image("basketball")
    >>> code_aois = aoi.find_rectangles(code_img)
    >>> raw_fixes = data.busjahn_2013.raw_fixations()
    >>> print raw_fixes[:5][["trial_id", "start_ms", "fix_x", "fix_y"]]
       trial_id  start_ms       fix_x       fix_y
    0         8       250  423.437500  378.083344
    1         8       567  324.711548   67.538460
    2         8       867  415.625000   -3.750000
    3         8      1284  444.852936  159.117645
    4         8      2034  366.030792  133.842896
    >>> aoi_fixes = aoi.hit_test(raw_fixes, sub_line_aois)
    >>> aoi_cols = aoi.get_aoi_columns(aoi_fixes)
    >>> print aoi_fixes[:5][["trial_id", "start_ms", "fix_x", "fix_y"] + aoi_cols]
       trial_id  start_ms       fix_x       fix_y aoi_line   aoi_sub-line
    0         8       250  423.437500  378.083344   line 9  line 9 part 1
    1         8       567  324.711548   67.538460      NaN            NaN
    2         8       867  415.625000   -3.750000      NaN            NaN
    3         8      1284  444.852936  159.117645   line 4  line 4 part 2
    4         8      2034  366.030792  133.842896      NaN            NaN
    
    """
    from shapely.geometry import Point
    output_rows = []

    # Create AOI polygons
    aoi_polys = {}
    for kind, group in aois.groupby("kind"):
        aoi_polys[kind] = { a["name"] : make_polygon(a)
                            for _, a in group.iterrows() }

    aoi_kinds = sorted(aoi_polys.keys())

    # Default offset
    if offsets is None:
        offsets = pandas.DataFrame({
            "name" : "none",
            "x"    : 0,
            "y"    : 0
        }, index=[0])

    # Hit test all fixations
    for _, fix in fixations.iterrows():
        for _, offset in offsets.iterrows():
            offset_kind = offset["name"]

            # Apply offset
            fix_x = fix["fix_x"] + offset["x"]
            fix_y = fix["fix_y"] + offset["y"]
            fix_pt = Point(fix_x, fix_y)

            row = list(fix.values) + [offset_kind]

            # Test AOIs in groups (no overlap within a group is assumed)
            for kind in aoi_kinds:
                test_polys = aoi_polys[kind]
                hit_aoi = hit_fun(fix_pt, test_polys, radius=hit_radius, **kwargs)
                if hit_aoi is None:
                    row.append(np.NaN)
                else:
                    row.append(hit_aoi)

            output_rows.append(row)

    cols = list(fixations.columns) + ["offset_kind"]

    # Add AOI hit columns
    cols += kinds_to_cols(aoi_kinds)

    return pandas.DataFrame(output_rows, columns=cols)

# }}}

# Automated Coding {{{

def combine_aoi_blocks(fixations, aoi_kind):
    aoi_col = kind_to_col(aoi_kind)

    # Create fixation blocks. Multiple instances of an AOI name are
    # collapsed into a single block.
    cols = [aoi_col, "start_ms", "end_ms"]
    block_start, block_end = None, None
    current_name = None
    blocks = []

    for name, start, end in fixations[cols].values:
        if name != current_name:
            if current_name is not None:
                blocks.append([current_name, int(block_start), int(block_end)])
            block_start, block_end = start, end
            current_name = name if not pandas.isnull(name) else None
        else:
            block_end = end

    if current_name is not None:
        blocks.append([current_name, int(block_start), int(block_end)])

    # Verify that no blocks overlap
    for b1 in blocks:
        assert b1[1] < b1[2]  # start is before end
        for b2 in blocks:
            if b1 != b2:
                # Block b1 is before or after b2
                assert (b1[2] < b2[1]) or (b1[1] > b2[2]), (b1, b2)

    return blocks

def code_fixations(fixations, lines_text_order, lines_exec_order,
        linear_order=3):
    from nltk.util import ngrams
    tags = []  # kind, name, description, start_ms, end_ms

    # Block, SubBlock, Signature, MethodCall
    # --------------------------------------
    # Codes that correspond directly to AOI rectangles.
    kind_categories = {
        "block"       : "Block",
        "sub-block"   : "SubBlock",
        "signature"   : "Signature",
        "method-call" : "MethodCall"
    }

    for kind, category in kind_categories.iteritems():
        blocks = combine_aoi_blocks(fixations, kind)
        for block in blocks:
            aoi_name, start, end = block
            code_name = ""

            if kind == "block":
                code_name = aoi_name.capitalize()
            elif kind == "sub-block":
                code_name = aoi_name.split(" ")[1].capitalize()
            elif kind == "signature":
                aoi_name = aoi_name.split(" ")[1]
                if aoi_name == "params":
                    code_name = "FormalParameterList"
                else:
                    code_name = aoi_name.capitalize()
            elif kind == "method-call":
                aoi_name = aoi_name.split(" ")[1]
                if aoi_name == "params":
                    code_name = "ActualParameterList"
                else:
                    code_name = aoi_name.capitalize()

            tags.append([category, code_name, aoi_name, start, end])

    # Linear
    # ------
    # Any 3 lines that follow text order (excluding blank lines).
    line_blocks = combine_aoi_blocks(fixations, "line")
    text_order_ngrams = set(ngrams(lines_text_order.name, linear_order))

    for line_group in window(line_blocks, linear_order):
        line_group = list(line_group)
        names = tuple(b[0] for b in line_group)
        if names in text_order_ngrams:
            start, end = line_group[0][1], line_group[-1][2]
            tags.append(["Pattern", "Linear", " -> ".join(names), start, end])

    # JumpControl
    # -----------
    # Any 2 lines that follow execution order.
    exec_order_ngrams = set(ngrams(lines_exec_order.name, 2))
    for line_group in window(line_blocks, 2):
        line_group = list(line_group)
        names = tuple(b[0] for b in line_group)
        if names in exec_order_ngrams:
            start, end = line_group[0][1], line_group[-1][2]
            tags.append(["Pattern", "JumpControl", " -> ".join(names), start, end])


    # LineScan
    # --------
    # 3 or more line chunks fixated consecutively.
    line_chunk_blocks = combine_aoi_blocks(fixations, "line chunks")
    current_line = None
    line_count = 0
    first_block = None
    for block in line_chunk_blocks:
        line = int(block[0].split(" ")[1])
        if line != current_line:
            if line_count >= 3:
                start, end = first_block[1], block[2]
                tags.append(["Pattern", "LineScan",
                    "line {0}".format(current_line), start, end])

            # Reset to new line
            first_block = block
            current_line = line
            line_count = 1
        else:
            # Continuing on current line
            line_count += 1


    # Signatures
    # ----------
    #
    aoi_cols = get_aoi_columns(fixations)
    sub_block_blocks = combine_aoi_blocks(fixations, "sub-block")
    for tag in tags:
        if tag[1] != "LineScan":
            continue

        # Make sure the scan is on a signature line
        scan_start, scan_end = tag[3], tag[4]
        fix_filter = fixations.apply(lambda f: scan_start <= f["start_ms"]
                and f["end_ms"] <= scan_end, axis=1)

        scan_fixes = fixations[fix_filter]
        assert len(scan_fixes) > 0
        scan_subblock = scan_fixes.iloc[0]["aoi_sub-block"]
        if pandas.isnull(scan_subblock):
            continue

        scan_name = scan_subblock.split(" ")
        if scan_name[1] != "signature":
            continue

        # Check that the next fixation is in the corresponding body
        later_fixes = fixations[fixations.start_ms > scan_end]\
                .dropna(subset=aoi_cols, how="all")

        next_fix_i = later_fixes.start_ms.argmin()
        next_fix = later_fixes.iloc[next_fix_i]
        next_subblock = next_fix["aoi_sub-block"]
        if pandas.isnull(next_subblock):
            continue

        next_name = next_subblock.split(" ")
        if (next_name[0] == scan_name[0]) and (next_name[1] == "body"):
            next_block = [b for b in sub_block_blocks
                    if (b[1] <= next_fix["start_ms"]) and (next_fix["end_ms"] <= b[2])][0]
            tags.append(["Pattern", "Signatures",
                "{0} -> {1}".format(" ".join(scan_name), " ".join(next_name)),
                scan_start, next_block[2]])


    # Pattern:Scan
    # ------------
    # First block of down fixations, terminated by too many up fixations or
    # dwell time on a single line.
    up_count = 3
    time_thresh = 1500
    current_line = None
    first_block = None
    prev_block = None

    for block in line_blocks:
        line = int(block[0].split(" ")[1])
        if current_line is None:
            current_line = line
            prev_block = block
            continue

        if first_block is None:
            if line > current_line:
                first_block = prev_block
            else:
                prev_block = block
        else:
            if line < current_line:
                up_count -= 1

            line_time = block[2] - block[1]
            if up_count < 1 or line_time > time_thresh:
                start, end = first_block[1], block[2]
                names = first_block[0], block[0]
                tags.append(["Pattern", "Scan", " -> ".join(names), start, end])
                break

        current_line = line

    # Create final data frame
    tags_df = pandas.DataFrame(tags, columns=["kind", "name",
        "description", "start_ms", "end_ms"])

    return tags_df

def make_xml_coding(xml_root, tags, media_url):
    from urlparse import urlunparse

    last_annotation_prop = xml_root.find("./HEADER/PROPERTY[@NAME='lastUsedAnnotationId']")
    last_annotation_id = int(last_annotation_prop.text)
    last_time_slot_id = 0

    # Find the actual last time slot id (assuming naming convention tsX)
    for time_order in xml_root.findall("./TIME_ORDER/TIME_SLOT"):
        slot_id_str = time_order.attrib["TIME_SLOT_ID"]
        if slot_id_str.startswith("ts"):
            slot_id = int(slot_id_str[2:])
            last_time_slot_id = max(last_time_slot_id, slot_id)


    # Utility methods
    def find_tier(tier_name):
        tier = xml_root.find("./TIER[@TIER_ID='{0}']".format(tier_name))
        tier_ann = tier.find("./ANNOTATION")
        if tier_ann is None:
            tier_ann = lb.E.ANNOTATION()
            tier.append(tier_ann)
        return tier_ann

    def add_time_slots(start, end):
        time_order = xml_root.find("./TIME_ORDER")
        time_1 = "ts{0}".format(last_time_slot_id + 1)
        time_2 = "ts{0}".format(last_time_slot_id + 2)
        time_order.append(lb.E.TIME_SLOT(TIME_SLOT_ID=time_1, TIME_VALUE=str(start)))
        time_order.append(lb.E.TIME_SLOT(TIME_SLOT_ID=time_2, TIME_VALUE=str(end)))
        return time_1, time_2

    def add_annotation(tier_ann, time_1, time_2, code_name):
        annotation = lb.E("ALIGNABLE_ANNOTATION",
                ANNOTATION_ID=str(last_annotation_id + 1),
                TIME_SLOT_REF1=time_1, TIME_SLOT_REF2=time_2)
        ann_value = lb.E.ANNOTATION_VALUE()
        ann_value.text = code_name
        annotation.append(ann_value)
        tier_ann.append(annotation)
        return annotation

    # -------------------------------------------

    # Add media info
    med_desc = xml_root.find("./HEADER/MEDIA_DESCRIPTOR")
    med_desc.attrib["MEDIA_URL"] = urlunparse(media_url)
    med_desc.attrib["RELATIVE_MEDIA_URL"] = "./{0}"\
            .format(media_url.path.split("/")[-1])

    # Tag kind -> tier name
    kinds_tiers = {
        "Block"      : "Block",
        "SubBlock"   : "Sub-Block",
        "Signature"  : "Signature",
        "MethodCall" : "Method-Call",
        "Pattern"    : "Pattern",
        "Strategy"   : "Strategy",
    }

    # Add time slots and annotations for each block
    cols = ["name", "start_ms", "end_ms"]
    for kind, kind_group in tags.groupby("kind"):
        tier = find_tier(kinds_tiers[kind])
        for name, start, end in kind_group[cols].values:
            time_1, time_2 = add_time_slots(start, end)
            add_annotation(tier, time_1, time_2, name)
            last_time_slot_id += 2
            last_annotation_id +=1 

    # Update annotation id
    last_annotation_prop.text = str(last_annotation_id)

    return xml_root

# }}}
