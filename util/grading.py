from io import StringIO

FORMAT_CHARS = ['[', ']', ',', ' ', '\n', '"', '\'']
TABLE = dict.fromkeys(map(ord, FORMAT_CHARS), None)

def correct_string(s):
    return unicode(s).translate(TABLE).lower()

def matching_strings(s):
    matching = [s]

    # Convert to universal newlines, strip extraneous whitespace
    s_io = StringIO(unicode(s.strip()), newline=None)
    s_str = s_io.read()
    matching.append(s_str)
    matching.append(s_str.translate(TABLE).lower())

    s_io.seek(0)
    s_lines = [line.strip() for line in s_io.readlines()]

    # Remove blank lines
    actual_lines = [line for line in s_lines if len(line.strip()) > 0]
    matching.append("\n".join(actual_lines))

    matching.append("\n".join([l.translate(TABLE).lower() for l in actual_lines]))

    return set(matching)

def grade_string(expected, actual):
    # Convert to universal newlines, strip extraneous whitespace
    expected_io = StringIO(unicode(expected.strip()), newline=None)
    actual_io = StringIO(unicode(actual.strip()), newline=None)

    expected_str = expected_io.read()
    actual_str = actual_io.read()

    # Pefect match
    if expected_str == actual_str:
        return "exact"

    table = dict.fromkeys(map(ord, FORMAT_CHARS), None)

    expected_io.seek(0)
    expected_lines = [line.strip() for line in expected_io.readlines()]
    actual_io.seek(0)
    actual_lines = [line.strip() for line in actual_io.readlines()]

    # Remove blank lines
    removed_blanks = False

    if len(expected_lines) != len(actual_lines):
        actual_lines = [line for line in actual_lines if len(line.strip()) > 0]
        removed_blanks = True

    # Check for line by line exact/partial match
    if len(expected_lines) == len(actual_lines):
        exact_match = True
        partial_match = False

        for (e_line, a_line) in zip(expected_lines, actual_lines):
            if e_line != a_line:
                exact_match = False
                if (e_line.translate(table).lower() == a_line.translate(table).lower()):
                    partial_match = True
                else:
                    partial_match = False
                    break

        if exact_match:
            return "exact" if not removed_blanks else "line"
        elif partial_match:
            return "line"

    # Check for partial match of values only
    if expected_str.translate(table).lower() == actual_str.translate(table).lower():
        return "values"

    return None

def cluster_responses(responses):
    sets = [matching_strings(s) for s in responses]
    classes = []

    for s in sets:
        found = False
        for i, (cls, count) in enumerate(classes):
            for s_i in s:
                if s_i in cls:
                    classes[i] = (cls, count + 1)
                    found = True
                    break
            if found:
                break
        if not found:
            classes.append((s, 1))
    
    return classes
