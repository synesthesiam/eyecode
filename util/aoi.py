import pandas, os
from shapely.geometry import Point, box

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

# AOI Creation {{{

def make_code_aois(code_file, font_size=(14, 25), line_offset=5,
        syntax_categories=SYNTAX_CATEGORIES):

    aoi_df = pandas.DataFrame(columns=("aoi_kind", "name",
        "x", "y", "width", "height", "note"))

    # Needed for syntax-based AOIs
    from pygments.lexers import PythonLexer
    lexer = PythonLexer()

    # Parse each file and generate AOIs
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
                    "aoi_kind" : "line",
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
                        "aoi_kind" : "block",
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
            "aoi_kind" : "syntax",
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
    x = int(aoi_row["x"])
    y = int(aoi_row["y"])
    width = int(aoi_row["width"])
    height = int(aoi_row["height"])
    return box(x, y, x + width, y + height)

def hit_point(fix_pt, aoi_polys, **kwargs):
    for aoi, poly in aoi_polys.iteritems():
        if poly.intersects(fix_pt):
            return aoi
    return None

def hit_circle(fix_pt, aoi_polys, radius):
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

def hit_test(fixations, aois, offsets=None, hit_fun=hit_circle, hit_radius=20):
    output_rows = []

    # Create AOI polygons
    aoi_polys = {}
    for kind, group in aois.groupby("kind"):
        aoi_polys[kind] = { a["name"] : make_polygon(a)
                            for _, a in group.iterrows() }

    aoi_kinds = sorted(aoi_polys.keys())
    #aoi_names = { k : sorted(aoi_polys[k].keys()) for k in aoi_kinds }

    # Default offset
    if offsets is None:
        offsets = pandas.DataFrame({
            "name" : "none",
            "x"    : 0,
            "y"    : 0
        })

    # Hit test all fixations
    for _, fix in fixations.iterrows():
        for _, offset in offsets.iterrows():
            offset_kind = offset["name"]

            # Apply offset
            fix_x = fix["fix_x"] + offset["x"]
            fix_y = fix["fix_y"] + offset["y"]
            fix_pt = Point(fix_x, fix_y)

            row = list(fixations.values) + [offset_kind]

            # Test AOIs in groups (no overlap within a group is assumed)
            for kind in aoi_kinds:
                test_polys = aoi_polys[kind]
                hit_aoi = hit_fun(fix_pt, test_polys, radius=hit_radius)
                if hit_aoi is None:
                    row.append("")
                else:
                    row.append(hit_aoi)

            output_rows.append(row)

    cols = list(fixations.columns) + ["offset_kind", "aoi_kind"]

    # Add AOI hit columns
    for kind in aoi_kinds:
        col_name = "hit_{0}".format(kind)
        cols.append(col_name)

    return pandas.DataFrame(output_rows, columns=cols)


#def fixations_dataframe(xml_paths, ):
    #output_rows = []
    #for exp_path in xml_paths:
        #exp = etree.parse(exp_path).xpath("/experiment")[0]
        #exp_id = int(exp.attrib["id"])
        ##print "Experiment {0}".format(exp.attrib["id"])

        #for trial in exp.xpath(".//trial"):
            #trial_id = int(trial.attrib["id"])
            #offsets = trial.xpath(".//offset")

            #aois = [a for a in trial.xpath(".//aoi")]

            ## Restrict to interface and line-based AOIs
            ##aois = [a for a in trial.xpath(".//aoi")
                ##if a.attrib["kind"] in ["interface", "line"]
                ##and a.attrib["name"] != "code box"]

            ##aoi_polys = { a: make_polygon(a) for a in aois }
            #aoi_groups = defaultdict(dict)
            
            #for aoi in aois:
                #aoi_kind = aoi.attrib["kind"]
                #aoi_groups[aoi_kind][aoi] = make_polygon(aoi)

            ## Hit test fixations
            #fixes = trial.xpath(".//fixation")
            #for offset in offsets:
                #offset_kind = offset.attrib["kind"]

                ##print "Using offset method: {0}".format(offset.attrib["kind"])
                #offset_x = int(offset.attrib["x"])
                #offset_y = int(offset.attrib["y"])

                #for hit_kind, hit_fun in hit_kinds.iteritems():
                    ##fix_aois = {}
                    #for fix in fixes:
                        #fix_id = int(fix.attrib["id"])

                        ## Correct using offset
                        #fix_x = int(fix.attrib["x"]) + offset_x
                        #fix_y = int(fix.attrib["y"]) + offset_y
                        #fix_pt = Point(fix_x, fix_y)

                        #aoi_kinds = []
                        #aoi_names = []

                        ## Do hit testing by AOI group to avoid overlapping of AOIs
                        #for aoi_kind, aoi_polys in aoi_groups.iteritems():
                            #hit_aoi = hit_fun(fix_pt, aoi_polys, radius=hit_radius)
                            ##fix_aois[fix] = hit_aoi
                            #if hit_aoi is not None:
                                #aoi_kinds.append(aoi_kind)
                                #aoi_names.append(hit_aoi.attrib["name"])

                        #output_rows.append([
                            #exp_id, trial_id,
                            #trial.attrib["base"], trial.attrib["version"],
                            #fix_id, offset_kind, hit_kind,
                            #",".join(aoi_kinds), ",".join(aoi_names),
                            #int(fix.attrib["start"]), int(fix.attrib["end"]),
                            #fix_x, fix_y
                        #])

                    ## Hit %
                    ##hit_pct = len(fix_aois) / float(len(fixes))
                    ##print "Hit % ({0}): {1:.2f}".format(hit_kind, hit_pct * 100)

    #fixes_df = pandas.DataFrame(output_rows, columns=("exp_id", "trial_id",
        #"base", "version", "fix_id", "offset_kind", "hit_kind",
        #"aoi_kind", "aoi_name", "start_ms", "end_ms", "fix_x", "fix_y"))

    ## Add duration column
    #fixes_df["duration"] = fixes_df.apply(lambda r: r["end_ms"] - r["start_ms"], axis=1)

    #return fixes_df

# }}}

