import numpy as np, functools as ft, itertools as it, pandas
import sys, cStringIO, contextlib
import re
from grading import *

def filter_trial(frame, exp_id, trial_id=None):
    if trial_id is None:
        return frame[frame.exp_id == exp_id]
    else:
        return frame[(frame.exp_id == exp_id) & (frame.trial_id == trial_id)]

def filter_program(frame, base, version=None):
    if version is None:
        return frame[frame.base == base]
    else:
        return frame[(frame.base == base) & (frame.version == version)]

def filter_aois(frame, kind, name=None):
    if name is None:
        return frame[frame.kind == kind]
    else:
        return frame[(frame.kind == kind) & (frame.name == name)]

def comma_list_contains(s, list_str):
    return list_str.split(",").contains(s)

def comma_list_contains_any(fun, list_str):
    for s in list_str.split(","):
        if fun(s):
            return True
    return False

def filter_lines(fixations, hit_kind="circle", offset_kind="manual experiment"):
    name_filter = ft.partial(comma_list_contains_any, lambda s: s.startswith("line "))

    line_fixes = fixations[(fixations.hit_kind == hit_kind) &
                           (fixations.offset_kind == offset_kind) &
                           (fixations.aoi_names.apply(name_filter))]

    line_fixes["line"] = line_fixes.aoi_name.apply(lambda n: int(n.split(" ")[1]))
    return line_fixes

def split_by_median(frame, column):
    m = frame[column].median()
    return frame[frame[column] <= m], frame[frame[column] > m]

def split_by_boolean(frame, column):
    return frame[frame[column]], frame[np.invert(frame[column])]

def contrast_color(rgba):
    a = 1 - ((0.299 * rgba[0]) + (0.587 * rgba[1]) + (0.114 * rgba[2]))
    return "black" if a < 0.5 else "white"

def transition_matrix(lines, num_lines=None):
    if num_lines is None:
        num_lines = max(lines)
        
    trans_counts = np.zeros(shape=(num_lines, num_lines))    
    for l1, l2 in zip(lines, lines[1:]):
        trans_counts[l1 - 1, l2 - 1] += 1
        
    # Normalize by rows
    row_sums = trans_counts.sum(axis=1)
    trans_probs = trans_counts / row_sums.reshape((-1, 1))
    
    # Get rid of NaNs
    return np.nan_to_num(trans_probs)

def gauss_kern(size, sigma=1.0):
   """ Returns a normalized 2D gauss kernel array for convolutions """
   h1 = size[0]
   h2 = size[1]
   x, y = np.mgrid[0:h2, 0:h1]
   x = x-h2/2
   y = y-h1/2
   g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) );
   return g / g.sum()

def make_heatmap(points, screen_size, point_size, sigma_denom=5.0):
    point_radius = point_size / 2
    screen = np.zeros((screen_size[0] + point_size, screen_size[1] + point_size))
    kernel = gauss_kern((point_size, point_size), sigma=(point_size / sigma_denom))
    for pt in points:
        x_start, y_start = pt[0], pt[1]
        x_end, y_end = x_start + point_size, y_start + point_size
        scr_slice = screen[x_start:x_end, y_start:y_end]
        width, height = scr_slice.shape[0], scr_slice.shape[1]
        screen[x_start:x_end, y_start:y_end] = scr_slice + kernel[:width, :height]

    screen = screen / screen.max()
    screen = screen[point_radius:-point_radius,
                    point_radius:-point_radius]

    return screen

def python_line_tokens(code_lines, blank_lines=False):
    from pygments.lexers import PythonLexer
    lexer = PythonLexer()
    code_str = "".join(code_lines)
    all_tokens = list(lexer.get_tokens(code_str, unfiltered=True))
    line_tokens = []
    current_line = []

    for t in all_tokens:
        if t[1] == u"\n":
            line_tokens.append(current_line)
            current_line = []
        else:
            current_line.append(t)

    rows = []
    for i, tokens in enumerate(line_tokens):
        # Check for blank line
        line_str = code_lines[i].rstrip()
        if (not blank_lines) and len(line_str.strip()) == 0:
            continue

        for t in tokens:
            kind, value = str(t[0]), t[1]
            yield line_str, i, kind, value, t

def python_line_categories(code_lines):
    from pygments.lexers import PythonLexer

    lexer = PythonLexer()
    code_str = "".join(code_lines)
    all_tokens = list(lexer.get_tokens(code_str, unfiltered=True))
    line_tokens = []
    current_line = []

    for t in all_tokens:
        if t[1] == u"\n":
            line_tokens.append(current_line)
            current_line = []
        else:
            current_line.append(t)

    line_categories = []
    for i, tokens in enumerate(line_tokens):
        # Check for blank line
        line_str = code_lines[i].rstrip()
        if len(line_str.strip()) == 0:
            line_categories.append(["blank line"])
            continue

        assert len(tokens) > 0, "No tokens for line"
        categories = []
        last_kind, last_value = None, None

        for t in tokens:
            kind, value = str(t[0]), t[1]

            if kind == u"Token.Keyword" and value == u"def":
                categories.append("function definition")
            elif kind == u"Token.Keyword" and value == u"if":
                categories.append("if statement")
            elif kind == u"Token.Keyword" and value == u"for":
                categories.append("for loop")
            elif kind == u"Token.Keyword" and value == u"return":
                categories.append("return statement")
            elif kind == u"Token.Keyword" and value == u"print":
                categories.append("print statement")
            elif kind == u"Token.Keyword" and value == u"class":
                categories.append("class definition")
            elif kind == u"Token.Operator" and value == u"=":
                categories.append("assignment")
            elif kind == u"Token.Operator" and value == u".":
                categories.append("object access")
            elif kind == u"Token.Operator" and value in [u"+", u"*"]:
                categories.append("mathematical operation")
            elif last_kind == u"Token.Operator" and last_value == u"-" and kind == "Token.Whitespace":
                categories.append("mathematical operation")
            elif kind == u"Token.Operator" and value in [u"<", u">"]:
                categories.append("comparison")
            elif last_kind == u"Token.Name" and kind == "Token.Punctuation" and value == u"(":
                categories.append("function call")
            elif kind == "Token.Punctuation" and value == u"[":
                categories.append("list creation")

            last_kind, last_value = kind, value

        if len(categories) == 0:
            categories.append("unknown")

        line_categories.append(set(categories))

    return line_categories

def python_token_metrics(code_lines, indent_size=4):
    from pygments.lexers import PythonLexer
    indent_regex = re.compile(r"^\s*")

    lexer = PythonLexer()
    code_str = "".join(code_lines)
    all_tokens = list(lexer.get_tokens(code_str, unfiltered=True))
    line_tokens = []
    current_line = []

    for t in all_tokens:
        if t[1] == u"\n":
            line_tokens.append(current_line)
            current_line = []
        else:
            current_line.append(t)

    rows = []
    for i, tokens in enumerate(line_tokens):
        # Check for blank line
        line_str = code_lines[i].rstrip()
        if len(line_str.strip()) == 0:
            continue

        assert len(tokens) > 0, "No tokens for line"

        num_keywords = 0
        num_identifiers = 0
        num_operators = 0
        line_length = len(line_str)
        whitespace_prop = line_str.count(" ") / float(line_length)
        line_indent = len(indent_regex.findall(line_str)[0]) / indent_size

        for t in tokens:
            kind, value = str(t[0]), t[1]
            if kind.startswith(u"Token.Keyword"):
                num_keywords += 1
            elif kind.startswith(u"Token.Name"):
                num_identifiers += 1
            elif kind.startswith(u"Token.Operator"):
                num_operators += 1

        line_number = i + 1
        rows.append([line_number, line_length, num_keywords,
            num_identifiers, num_operators, whitespace_prop,
            line_indent])

    columns = ["line", "line_length", "keywords",
               "identifiers", "operators", "whitespace_prop",
               "line_indent"]
    return pandas.DataFrame(rows, columns=columns)

def all_pairs(items, fun, same_value=np.NaN):
    results = np.zeros((len(items), len(items)))
    for i, item_i in enumerate(items):
        for j, item_j in enumerate(items):
            if i < j:
                results[i, j] = fun(item_i, item_j)
                results[j, i] = results[i, j]
            else:
                results[i, j] = same_value
    return results

def file_to_text_buffer(f, pad_left=0):
    if isinstance(f, str):
        f = open(f, "r")

    lines = [l.strip() for l in f.readlines()]
    rows = len(lines)
    cols = max([len(l) for l in lines]) + pad_left
    buffer = np.zeros((rows, cols), dtype=str)
    buffer[:, :] = " "  # Fill with white space

    # Fill buffer
    for r, line in enumerate(lines):
        chars = ([" "] * pad_left) + list(line)
        buffer[r, :len(chars)] = chars

    return buffer

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = cStringIO.StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

def rolling_func(fixations, fun, window_size_ms, step_ms):
    start, end = 0, window_size_ms
    return_series = False
    if not isinstance(fun, dict):
        fun = { "value" : fun }
        return_series = True

    values = { k : [] for k, v in fun.iteritems() }
    times = []
    while start < fixations.end_ms.max():
        times.append(start + (window_size_ms / 2))
        win_fixations = fixations[
                ((fixations.start_ms >= start) & (fixations.start_ms < end)) |
                ((fixations.end_ms >= start) & (fixations.end_ms < end))]

        for k, f in fun.iteritems():
            values[k].append(f(win_fixations))

        start += step_ms
        end += step_ms

    first_values = values[values.keys()[0]]

    if return_series:
        series = pandas.Series(first_values, index=times)
        return series
    else:
        df = pandas.DataFrame(values, index=times)
        return df

def window(seq, n):
    """Returns a sliding window (of width n) over data from the iterable s ->
    (s0,s1,...s[n-1]), (s1,s2,...,sn), ..."""
    seq_it = iter(seq)
    result = tuple(it.islice(seq_it, n))
    if len(result) == n:
        yield result    
    for elem in seq_it:
        result = result[1:] + (elem,)
        yield result

def just(n, seq):
    """Splits a sequence into n, rest parts."""
    it = iter(seq)
    for _ in range(n - 1):
        yield next(it, None)
    yield tuple(it)

def just2(n, seq):
    """Iterates over a sequence, splitting each item into n, rest parts."""
    for inner_seq in seq:
        yield tuple(just(n, inner_seq))

def significant(p_value):
    if p_value < 0.001:
        return "***"
    
    if p_value < 0.01:
        return "**"
    
    if p_value < 0.05:
        return "*"
    
    return ""

def significant_p(p_value):
    if p_value < 0.001:
        return "p < .001"
    
    if p_value < 0.01:
        return "p < .01"
    
    if p_value < 0.05:
        return "p < .05"
    
    return ""

def grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    
    See also
    --------
    http://www.python.org/doc//current/library/itertools.html

    """
    args = [iter(iterable)] * n
    return it.izip_longest(fillvalue=fillvalue, *args)

def pairwise(iterable, fillvalue=None):
    """s -> (s0,s1), (s1,s2), (s2, s3), ...

    See also
    --------
    http://www.python.org/doc//current/library/itertools.html

    """
    a, b = it.tee(iterable)
    next(b, fillvalue)
    return it.izip(a, b)

def split_whitespace_tokens(line):
    """Splits a line of text by whitespace"""
    in_quote = False
    token = ""
    token_start = 0
    for i, char in enumerate(line):
        if char == ' ':
            if len(token) > 0:
                yield (token_start, token)
                token = ""
        else:
            if len(token) == 0:
                token_start = i
            token += char
    if len(token) > 0:
        yield (token_start, token)
