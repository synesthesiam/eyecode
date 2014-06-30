import sys, os
# eyecode/data/hansen_2012
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import gzip, pandas, json
import eyecode.aoi, eyecode.util, eyecode.metrics
import numpy as np
import scipy.spatial
import nltk.metrics
from lxml import etree
from glob import glob
from collections import defaultdict
from datetime import datetime
from time import mktime

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
GRID_AOI_SIZE = (30, 30)
xml_paths = glob(os.path.join("xml", "*.xml.gz"))
code_paths = glob(os.path.join("programs", "*.py"))

def pretty_json(js_str):
    return json.dumps(json.loads(js_str), indent=4)

# -------------------------------------------------- 

def task_raw_fixations():
    def make_raw():
        output_rows = []

        # Experiments
        for path in xml_paths:
            xml_node = etree.parse(path)
            exp = xml_node.xpath("/experiment")[0]
            exp_id = int(exp.attrib["id"])

            # Trials
            for trial in exp.xpath(".//trial"):
                trial_id = int(trial.attrib["id"])

                # Fixations
                for fix in trial.xpath(".//fixation"):
                    fix_x = int(fix.attrib["x"])
                    fix_y = int(fix.attrib["y"])

                    output_rows.append([
                        exp_id, trial_id,
                        int(fix.attrib["start"]), int(fix.attrib["end"]),
                        fix_x, fix_y
                    ])

        fixes_df = pandas.DataFrame(output_rows, columns=("exp_id", "trial_id",
            "start_ms", "end_ms", "fix_x", "fix_y"))

        # Add duration column
        fixes_df["duration_ms"] = fixes_df.apply(lambda r: r["end_ms"] - r["start_ms"], axis=1)

        with gzip.open("raw_fixations.csv.gz", "w") as out_file:
            fixes_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_raw],
        "file_dep" : xml_paths,
        "targets"  : ["raw_fixations.csv.gz"]
    }

# -------------------------------------------------- 

def task_intuitive_aois():
    def get_category(t_kind):
        if t_kind == "List":
            return "list"
        elif t_kind == "Args":
            return "args"
        elif t_kind == "Index":
            return "index"
        elif t_kind == "Condition":
            return "condition"
        elif t_kind == "Tuple":
            return "Tuple"
        else:
            return eyecode.aoi.get_token_category(t_kind)

    def make_intuitive_aois():
        import pygments
        import pygments.lexers
        lexer = pygments.lexers.PythonLexer()

        programs = pandas.read_csv(gzip.open("programs.csv.gz", "r"))
        all_aois = []

        for (base, version) in programs[["base", "version"]].values:
            program_path = os.path.join("programs", "{0}_{1}.py".format(base, version))
            with open(program_path, "r") as in_file:
                code = "\n".join([line.rstrip() for line in in_file])

            # Convert to 2-d tokens and then to monospace AOIs
            tokens = lexer.get_tokens_unprocessed(code)
            tokens2d = eyecode.aoi.tokens_to_2d(tokens)
            tokens2d = eyecode.aoi.intuitive_python_tokens(tokens2d)

            aois = eyecode.aoi.tokens2d_monospace_aois(tokens2d,
                    font_size=(14, 29),
                    get_category=get_category, line_offset=1,
                    token_kind="intuitive", line_kind=None)

            aois["base"] = base
            aois["version"] = version
            all_aois.append(aois)

        # Append new AOIs
        aois_df = pandas.concat(all_aois, ignore_index=True)

        with gzip.open("intuitive_aois.csv.gz", "w") as out_file:
            aois_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_intuitive_aois],
        "file_dep" : ["programs.csv.gz"],
        "targets"  : ["intuitive_aois.csv.gz"]
    }

# -------------------------------------------------- 

def task_aois():
    def make_aois():
        output_rows = []
        base_versions = {}
        intuitive_aois = pandas.read_csv(gzip.open("intuitive_aois.csv.gz", "r"))

        # Experiments
        for path in xml_paths:
            xml_node = etree.parse(path)
            exp = xml_node.xpath("/experiment")[0]
            exp_id = int(exp.attrib["id"])

            # Trials
            for trial in exp.xpath(".//trial"):
                trial_id = int(trial.attrib["id"])
                code_box_aoi = trial.xpath("./areas-of-interest//aoi[@kind='interface' and @name='code box']")[0]
                code_x      = int(code_box_aoi.attrib["x"])
                code_y      = int(code_box_aoi.attrib["y"])
                code_width  = int(code_box_aoi.attrib["width"])
                code_height = int(code_box_aoi.attrib["height"])

                for aoi in trial.xpath("./areas-of-interest//aoi"):
                    aoi_x = int(aoi.attrib["x"])
                    aoi_y = int(aoi.attrib["y"])
                    aoi_id = "{0},{1}".format(aoi_x - code_x, aoi_y - code_y)

                    output_rows.append([
                        exp_id, trial_id,
                        aoi.attrib["kind"], aoi.attrib["name"],
                        aoi_x, aoi_y,
                        int(aoi.attrib["width"]), int(aoi.attrib["height"]),
                        aoi_id
                    ])

                # Cache base/version for later
                base_versions[(exp_id, trial_id)] = (trial.attrib["base"], trial.attrib["version"])

        aois_df = pandas.DataFrame(output_rows, columns=("exp_id", "trial_id",
            "kind", "name", "x", "y", "width", "height", "local_id"))

        # Add program specific AOIs
        new_aois = []
        for (exp_id, trial_id), t_aois in aois_df.groupby(["exp_id", "trial_id"]):
            line_aois = eyecode.util.filter_aois(t_aois, "line")
            line_env = eyecode.aoi.envelope(line_aois).irow(0)

            # Create grid-based AOIs using line AOIs
            grid_aois = eyecode.aoi.make_grid(line_env["x"], line_env["y"],
                    line_env["width"], line_env["height"],
                    aoi_width=GRID_AOI_SIZE[0], aoi_height=GRID_AOI_SIZE[1],
                    kind="code-grid")
            grid_aois["exp_id"] = exp_id
            grid_aois["trial_id"] = trial_id
            new_aois.append(grid_aois)

            base, version = base_versions[(exp_id, trial_id)]
            if base == "counting":
                num_list_aois = t_aois[t_aois.local_id.isin(["126,-2", "294,-2"])]
                num_list = eyecode.aoi.envelope(num_list_aois, kind="syntax-meta", name="number-list")
                num_list["exp_id"] = exp_id
                num_list["trial_id"] = trial_id
                num_list["local_id"] = "126,-2"
                new_aois.append(num_list)

            # Create whitespace-separated token AOIs
            program_path = os.path.join("programs", "{0}_{1}.py".format(base, version))
            code_box = eyecode.util.filter_aois(t_aois, "interface", "code box").irow(0)
            t_line_aois = eyecode.util.filter_aois(t_aois, "line")

            code_x, code_y = code_box["x"], code_box["y"] - 2
            char_w = 14

            token_aois = []
            token_id = 1
            for line_num, line in enumerate(open(program_path, "r")):
                if len(line.strip()) == 0:
                    continue
                line = line.rstrip()
                line_aoi = t_line_aois[t_line_aois.name == "line {0}".format(line_num + 1)].irow(0)
                line_y = line_aoi["y"]
                line_h = line_aoi["height"]
                for start, token in eyecode.util.split_whitespace_tokens(line):
                    x = code_x + (start * char_w)
                    y = line_y
                    w = len(token) * char_w
                    h = line_h
                    name = "token {0}".format(token_id)
                    local_id = "{0},{1}".format(x, y)
                    token_aois.append([exp_id, trial_id, "whitespace-token", name, local_id, x, y, w, h])
                    token_id += 1
                    
            columns = ["exp_id", "trial_id", "kind", "name", "local_id",
                       "x", "y", "width", "height"]
            token_aois = pandas.DataFrame(token_aois, columns=columns)
            new_aois.append(token_aois)

            # Add intuitive AOIs
            t_intuit_aois = eyecode.util.filter_program(intuitive_aois, base, version)
            t_intuit_aois["x"] += code_x
            t_intuit_aois["y"] += code_y - 1

            for idx, row in t_intuit_aois.iterrows():
                aoi_x, aoi_y = row["x"], row["y"]
                local_id = "{0},{1}".format(aoi_x - code_x, aoi_y - code_y)
                t_intuit_aois.ix[idx, "local_id"] = local_id

            t_intuit_aois["exp_id"] = exp_id
            t_intuit_aois["trial_id"] = trial_id
            new_aois.append(t_intuit_aois)

        # Append new AOIs
        if len(new_aois) > 0:
            aois_df = pandas.concat([aois_df] + new_aois, ignore_index=True)

        with gzip.open("aois.csv.gz", "w") as out_file:
            aois_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_aois],
        "file_dep" : xml_paths + ["intuitive_aois.csv.gz"],
        "targets"  : ["aois.csv.gz"]
    }

# -------------------------------------------------- 

def task_all_fixations():
    def make_polygon(aoi):
        from shapely.geometry import box
        x = aoi["x"]
        y = aoi["y"]
        width = aoi["width"]
        height = aoi["height"]
        return box(x, y, x + width, y + height)

    def make_all():
        from shapely.geometry import Point
        output_rows = []
        #hit_kinds = { "point" : eyecode.aoi.hit_point, "circle" : eyecode.aoi.hit_circle }
        hit_kinds = { "circle" : eyecode.aoi.hit_circle }
        hit_radius = 20
        aoi_kinds = ["interface", "line", "syntax", "block", "code-grid",
                     "whitespace-token", "intuitive"]
        aois = pandas.read_csv(gzip.open("aois.csv.gz", "r"))

        # Experiments
        for path in xml_paths:
            xml_node = etree.parse(path)
            exp = xml_node.xpath("/experiment")[0]
            exp_id = int(exp.attrib["id"])

            # Trials
            for trial in exp.xpath(".//trial"):
                trial_id = int(trial.attrib["id"])
                offsets = trial.xpath(".//offset")

                # Extract AOIs for this trial
                t_aois = eyecode.util.filter_trial(aois, exp_id, trial_id)
                code_box_aoi = t_aois[(t_aois.kind == "interface") &
                        (t_aois.name == "code box")].irow(0)
                
                assert code_box_aoi is not None, "code box AOI not found"
                code_x = code_box_aoi["x"]
                code_y = code_box_aoi["y"]

                # Group AOIs by kind and create polygons
                aoi_groups = defaultdict(dict)
                for idx, aoi in t_aois.iterrows():
                    aoi_kind = aoi["kind"]
                    aoi_groups[aoi_kind][idx] = make_polygon(aoi)

                # Hit test fixations
                fixes = trial.xpath(".//fixation")
                for offset in offsets:
                    offset_kind = offset.attrib["kind"]

                    #print "Using offset method: {0}".format(offset.attrib["kind"])
                    offset_x = int(offset.attrib["x"])
                    offset_y = int(offset.attrib["y"])

                    for hit_kind, hit_fun in hit_kinds.iteritems():
                        #fix_aois = {}
                        for fix in fixes:
                            # Correct using offset
                            fix_x = int(fix.attrib["x"]) + offset_x
                            fix_y = int(fix.attrib["y"]) + offset_y
                            fix_pt = Point(fix_x, fix_y)
                            pupil_left = float(fix.attrib["pupil_left"])
                            pupil_right = float(fix.attrib["pupil_right"])

                            hit_names = []
                            hit_ids = []

                            # Do hit testing by AOI group to avoid overlapping of AOIs
                            for aoi_kind in aoi_kinds:
                                hit_name = ""
                                hit_id = ""

                                if aoi_kind in aoi_groups:
                                    aoi_polys = aoi_groups[aoi_kind]
                                    hit_idx = hit_fun(fix_pt, aoi_polys, radius=hit_radius)
                                    if hit_idx is not None:
                                        hit_aoi = t_aois.ix[hit_idx]
                                        hit_name = hit_aoi["name"]
                                        hit_x = int(hit_aoi["x"]) - code_x
                                        hit_y = int(hit_aoi["y"]) - code_y
                                        hit_id = "{0},{1}".format(hit_x, hit_y)

                                hit_names.append(hit_name)
                                hit_ids.append(hit_id)

                            output_rows.append([
                                exp_id, trial_id,
                                trial.attrib["base"], trial.attrib["version"],
                                offset_kind, offset_x, offset_y, hit_kind,
                                int(fix.attrib["start"]), int(fix.attrib["end"]),
                                fix_x, fix_y, pupil_left, pupil_right
                            ] + hit_names + hit_ids)

        cols = ["exp_id", "trial_id", "base", "version",
                "offset_kind", "offset_x", "offset_y", "hit_kind",
                "start_ms", "end_ms", "fix_x", "fix_y",
                "pupil_left", "pupil_right"] \
                + ["aoi_{0}".format(k) for k in aoi_kinds] \
                + ["hit_id_{0}".format(k) for k in aoi_kinds]

        fixes_df = pandas.DataFrame(output_rows, columns=cols)

        # Add duration column
        fixes_df["duration_ms"] = fixes_df.apply(lambda r: r["end_ms"] - r["start_ms"], axis=1)

        with gzip.open("all_fixations.csv.gz", "w") as out_file:
            fixes_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_all],
        "file_dep" : xml_paths + ["aois.csv.gz"],
        "targets"  : ["all_fixations.csv.gz"]
    }

# -------------------------------------------------- 

def task_trial_metrics():
    def make_metrics():
        import nltk
        from collections import Counter

        all_fixes = pandas.read_csv(gzip.open("all_fixations.csv.gz", "r"))
        line_fixes = pandas.read_csv(gzip.open("line_fixations.csv.gz", "r"))
        aois = pandas.read_csv(gzip.open("aois.csv.gz", "r"))
        line_categories = pandas.read_csv(gzip.open("line_categories.csv.gz", "r"))
        trials = pandas.read_csv(gzip.open("trials.csv.gz", "r"))

        metric_rows = []
        for (exp_id, trial_id), trial_fixes in all_fixes.groupby(["exp_id", "trial_id"]):
            trial = trials[(trials.exp_id == exp_id) & (trials.trial_id == trial_id)]
            if len(trial) == 0:
                print "make_metrics: Skipping trial {0} {1}".format(exp_id, trial_id)
                continue

            trial = trial.irow(0)
            trial_aois = aois[(aois.exp_id == exp_id) & (aois.trial_id == trial_id)]
            trial_line_fixes = line_fixes[(line_fixes.exp_id == exp_id) & (line_fixes.trial_id == trial_id)]
            trial_line_cats = eyecode.util.filter_program(line_categories, trial["base"], trial["version"])
            nonblank_lines = set(trial_line_cats[trial_line_cats.categories != "blank line"].line)
            
            # Basic metrics
            num_fixes = len(trial_fixes)
            fix_duration = eyecode.metrics.avg_fixation_duration(trial_fixes)
            scanpath_length = eyecode.metrics.scanpath_length(trial_fixes)
            fixes_per_sec = num_fixes / float(trial["duration_sec"])
            
            # AOI first fixations
            first_fixes = eyecode.metrics.first_fix_ms_aoi(trial_fixes)
            first_output_box = np.nan
            try:
                first_output_box = first_fixes.ix["interface", "output box"]
            except KeyError:
                pass
            
            # Voluntary/involuntary fixations
            voluntary = sum(trial_fixes.duration_ms > 320)
            involuntary = sum(trial_fixes.duration_ms < 240)
            
            # Code box/output box transitions
            trans_counter = Counter(nltk.ngrams(trial_fixes.aoi_interface.values, 2))
            transitions = trans_counter[("code box", "output box")] + \
                          trans_counter[("output_box", "code box")]
                
            # Spatial density
            code_box = eyecode.util.filter_aois(trial_aois, "interface", "code box")
            output_box = eyecode.util.filter_aois(trial_aois, "interface", "output box")
            code_density = eyecode.metrics.spatial_density(trial_fixes, code_box)
            output_density = eyecode.metrics.spatial_density(trial_fixes, output_box)
            
            # Convex hull area
            code_fixes = trial_fixes[trial_fixes.aoi_interface == "code box"]
            code_area = eyecode.metrics.convex_hull_area(code_fixes)
            code_duration_ms = code_fixes.duration_ms.sum()


            output_fixes = trial_fixes[trial_fixes.aoi_interface == "output box"]
            output_duration_ms = output_fixes.duration_ms.sum()

            # Uwano review percent
            time_cutoff = 0.3 * trial["duration_ms"]  # First 30% of trial
            lines_fixated = set(trial_line_fixes[trial_line_fixes.end_ms < time_cutoff].line.unique())
            lines_fixated = lines_fixated.intersection(nonblank_lines)
            percent_lines = len(lines_fixated) / float(len(nonblank_lines))
            
            metric_rows.append([exp_id, trial_id, fix_duration, first_output_box,
                                voluntary, involuntary, transitions, code_density,
                                output_density, scanpath_length, code_area,
                                num_fixes, fixes_per_sec, percent_lines,
                                code_duration_ms, output_duration_ms])
            
        cols = ["exp_id", "trial_id", "avg_fixation_duration", "first_output_fix_ms", "voluntary_fixes",
                "involuntary_fixes", "code_output_transitions", "code_density", "output_density",
                "scanpath_length", "code_area", "num_fixes", "fixes_per_sec", "uwano_review_percent",
                "code_duration_ms", "output_duration_ms"]

        metrics_df = pandas.DataFrame(metric_rows, columns=cols)

        with gzip.open("trial_metrics.csv.gz", "w") as out_file:
            metrics_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_metrics],
        "file_dep" : ["all_fixations.csv.gz", "aois.csv.gz", "line_fixations.csv.gz",
                      "line_categories.csv.gz", "trials.csv.gz"],
        "targets"  : ["trial_metrics.csv.gz"]
    }


# -------------------------------------------------- 

def task_all_saccades():
    def make_all():
        output_rows = []

        # Experiments
        for path in xml_paths:
            xml_node = etree.parse(path)
            exp = xml_node.xpath("/experiment")[0]
            exp_id = int(exp.attrib["id"])

            # Trials
            for trial in exp.xpath(".//trial"):
                trial_id = int(trial.attrib["id"])

                for sacc in trial.xpath(".//saccade"):
                    output_rows.append([
                        exp_id, trial_id,
                        trial.attrib["base"], trial.attrib["version"],
                        int(sacc.attrib["start"]), int(sacc.attrib["end"]),
                        float(sacc.attrib["x1"]), float(sacc.attrib["y1"]),
                        float(sacc.attrib["x2"]), float(sacc.attrib["y2"])
                    ])

        cols = ["exp_id", "trial_id", "base", "version",
                "start_ms", "end_ms", "sacc_x1", "sacc_y1",
                "sacc_x2", "sacc_y2"]

        saccs_df = pandas.DataFrame(output_rows, columns=cols)

        # Add duration column
        saccs_df["duration_ms"] = saccs_df \
                .apply(lambda r: r["end_ms"] - r["start_ms"], axis=1)

        # Exclude zero duration saccades
        saccs_df = saccs_df[saccs_df.duration_ms > 0]

        # Add Euclidean distance between start and end points
        dist = scipy.spatial.distance.euclidean
        saccs_df["dist_euclid"] = saccs_df \
                .apply(lambda r: dist(
                    r[["sacc_x1", "sacc_y1"]].values,
                    r[["sacc_x2", "sacc_y2"]].values
                ), axis=1)

        with gzip.open("all_saccades.csv.gz", "w") as out_file:
            saccs_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_all],
        "file_dep" : xml_paths,
        "targets"  : ["all_saccades.csv.gz"]
    }

# -------------------------------------------------- 

def task_line_fixations():
    def make_lines():
        offset_kind = "manual experiment"
        hit_kind = "circle"

        all_fixes = pandas.read_csv(gzip.open("all_fixations.csv.gz", "r"))
        line_fixes = all_fixes[(all_fixes.offset_kind == offset_kind) &
                               (all_fixes.hit_kind == hit_kind) &
                               -pandas.isnull(all_fixes.aoi_line)]

        line_fixes["line"] = line_fixes.aoi_line\
                .apply(lambda s: int(s.split(" ")[1]))

        drop_cols = ["offset_kind", "hit_kind"] +\
                [c for c in line_fixes.columns if c.startswith("aoi_")]

        line_fixes = line_fixes.drop(drop_cols, axis=1)

        with gzip.open("line_fixations.csv.gz", "w") as out_file:
            line_fixes.to_csv(out_file, index=False)

    return {
        "actions"  : [make_lines],
        "file_dep" : xml_paths + ["all_fixations.csv.gz"],
        "targets"  : ["line_fixations.csv.gz"]
    }

# -------------------------------------------------- 

def task_experiments():
    def make_experiments():
        rows = []
        responses = etree.parse("response_data.xml.gz")

        for e in responses.xpath("//experiment"):
            exp_id = int(e.attrib["id"])
            location = e.attrib["location"]

            age           = int(e.xpath(".//question[@name='age']/text()")[0])
            degree        = e.xpath(".//question[@name='education']/text()")[0]
            gender        = e.xpath(".//question[@name='gender']/text()")[0]
            py_years      = float(e.xpath(".//question[@name='python_years']/text()")[0])
            prog_years    = float(e.xpath(".//question[@name='programming_years']/text()")[0])
            cs_major      = e.xpath(".//question[@name='major']/text()")[0]
            difficulty    = e.xpath(".//question[@name='difficulty']/text()")[0]
            guess_correct = e.xpath(".//question[@name='correct']/text()")[0]
            total_grade   = sum([int(v) for v in e.xpath(".//trial/@grade-value")])

            started = datetime.strptime(e.attrib["started"], TIME_FORMAT)
            ended = datetime.strptime(e.attrib["ended"], TIME_FORMAT)
            duration_sec = (ended - started).total_seconds()

            rows.append([exp_id, age, degree, gender, py_years, prog_years,
                cs_major, difficulty, guess_correct, total_grade,
                duration_sec, location])

        # Convert to data frame
        cols = ["exp_id", "age", "degree", "gender", "py_years", "prog_years",
                "cs_major", "difficulty", "guess_correct", "total_grade",
                "duration_sec", "location"]

        exp_df = pandas.DataFrame(rows, columns=cols)

        # Add numeric representations of some columns
        for col in ["gender", "degree", "cs_major", "difficulty", "guess_correct"]:
            values = list(exp_df[col].unique())
            exp_df[col + "_num"] = exp_df[col].apply(lambda v: values.index(v))

        # Write to CSV
        with gzip.open("experiments.csv.gz", "w") as out_file:
            exp_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_experiments],
        "file_dep" : ["response_data.xml.gz"],
        "targets"  : ["experiments.csv.gz"]
    }

# -------------------------------------------------- 

def task_trials():
    def make_trials():
        rows = []
        responses = etree.parse("response_data.xml.gz")

        for e in responses.xpath("//experiment"):
            exp_id = int(e.attrib["id"])
            exp_started = datetime.strptime(e.attrib["started"], TIME_FORMAT)

            age           = int(e.xpath(".//question[@name='age']/text()")[0])
            degree        = e.xpath(".//question[@name='education']/text()")[0]
            gender        = e.xpath(".//question[@name='gender']/text()")[0]
            py_years      = float(e.xpath(".//question[@name='python_years']/text()")[0])
            prog_years    = float(e.xpath(".//question[@name='programming_years']/text()")[0])
            cs_major      = e.xpath(".//question[@name='major']/text()")[0]
            is_expert     = (py_years >= 5) or (prog_years >= 10)

            for t in e.xpath(".//trial"):
                trial_id = int(t.attrib["id"])
                base = t.attrib["base"]
                version = t.attrib["version"]

                # Grades
                grade_category = t.attrib["grade-category"]
                grade_value = int(t.attrib["grade-value"])

                # Trial time
                started = datetime.strptime(t.attrib["started"], TIME_FORMAT)
                started_ms = int((started - exp_started).total_seconds() * 1000)
                ended = datetime.strptime(t.attrib["ended"], TIME_FORMAT)
                ended_ms = int((ended - exp_started).total_seconds() * 1000)

                duration_ms = ended_ms - started_ms

                # Keystrokes
                keystroke_times = [int(v) for v in t.xpath("./responses//response/@timestamp")]
                keystroke_duration_ms = 0
                
                if len(keystroke_times) > 0:
                    keystroke_duration_ms = max(keystroke_times) - min(keystroke_times)

                keystroke_count = len(t.xpath("./responses//response"))
                response_proportion = keystroke_duration_ms / float(duration_ms)

                # Corrections
                corrections = 0
                last_len = 0
                in_correction = False
                for resp in t.xpath("./responses/response/text()"):
                    len_resp = len(resp)
                    if len_resp > last_len:
                        in_correction = False
                        last_len = len_resp
                    elif (len_resp < last_len) and not in_correction:
                        in_correction = True
                        last_len = len_resp
                        corrections += 1

                # Output
                true_output = t.xpath("./true-output/text()")[0]
                pred_output = t.xpath("./predicted-output/text()")[0]

                # Output edit distance
                perfect_dist      = nltk.metrics.edit_distance(pred_output.rstrip(),
                                                               true_output.rstrip())
                max_len           = max(len(pred_output.rstrip()), len(true_output.rstrip()))
                perfect_dist_norm = perfect_dist / float(max_len)

                pred_correct      = eyecode.util.correct_string(pred_output)
                true_correct      = eyecode.util.correct_string(true_output)
                correct_dist      = nltk.metrics.edit_distance(pred_correct, true_correct)
                max_len           = max(len(pred_correct), len(true_correct))
                correct_dist_norm = correct_dist / float(max_len)

                # Metrics
                code_chars   = int(t.xpath("./metrics/metric[@name = 'code chars']/@value")[0])
                code_lines   = int(t.xpath("./metrics/metric[@name = 'code lines']/@value")[0])
                cyclo_comp   = int(t.xpath("./metrics/metric[@name = 'cyclomatic complexity']/@value")[0])
                hal_effort   = float(t.xpath("./metrics/metric[@name = 'halstead effort']/@value")[0])
                hal_volume   = float(t.xpath("./metrics/metric[@name = 'halstead volume']/@value")[0])
                output_chars = int(t.xpath("./metrics/metric[@name = 'output chars']/@value")[0])
                output_lines = int(t.xpath("./metrics/metric[@name = 'output lines']/@value")[0])

                keystroke_coefficient = 0.0
                
                if keystroke_count > 0:
                    # Subtract 1 for the extra newline at the end of the true output.
                    # No one types this, so it would be possible to get a perfect score
                    # while having a coefficient < 1.
                    keystroke_coefficient = keystroke_count / float(output_chars - 1)

                rows.append([trial_id, exp_id, base, version, grade_value, grade_category,
                    started_ms, ended_ms, duration_ms, keystroke_duration_ms,
                    keystroke_count, keystroke_coefficient, response_proportion,
                    corrections, code_chars, code_lines, cyclo_comp, hal_effort, hal_volume,
                    output_chars, output_lines, true_output, pred_output,
                    age, degree, gender, py_years, prog_years, cs_major,
                    perfect_dist, perfect_dist_norm, correct_dist,
                    correct_dist_norm, is_expert])

        cols = ["trial_id", "exp_id", "base", "version", "grade_value",
                "grade_category", "started_ms", "ended_ms", "duration_ms",
                "keystroke_duration_ms", "keystroke_count", "keystroke_coefficient",
                "response_proportion", "response_corrections", "code_chars", "code_lines",
                "cyclo_comp", "hal_effort", "hal_volume", "output_chars", "output_lines",
                "true_output", "pred_output", "age", "degree", "gender",
                "py_years", "prog_years", "cs_major", "perfect_dist",
                "perfect_dist_norm", "correct_dist", "correct_dist_norm", "is_expert"]

        trial_df = pandas.DataFrame(rows, columns=cols)

        # Add derived columns
        trial_df["grade_perfect"] = trial_df.perfect_dist == 0
        trial_df["grade_correct"] = trial_df.correct_dist == 0
        trial_df["grade_common"] = trial_df.grade_category.str.startswith("common")

        trial_df["duration_sec"] = trial_df.duration_ms / 1000.0
        trial_df["duration_ms_log"] = np.log(trial_df.duration_ms)
        trial_df["duration_sec_log"] = np.log(trial_df.duration_ms) / 1000.0

        trial_df["program_name"] = trial_df.apply(lambda t: "{0}_{1}".format(t["base"], t["version"]), axis=1)

        # Add numeric representations of some columns
        for col in ["base", "version", "gender", "degree", "cs_major"]:
            values = list(trial_df[col].unique())
            trial_df[col + "_num"] = trial_df[col].apply(lambda v: values.index(v))

        # Exclude trials whose durations are 3 standard deviations or more away
        # from the mean (in log space).
        exclude_idxs = []
        for base, b_trials in trial_df.groupby("base"):
            cutoff = b_trials.duration_ms_log.mean() + (3 * b_trials.duration_ms_log.std())
            b_trials = b_trials[b_trials.duration_ms_log > cutoff]
            exclude_idxs = exclude_idxs + list(b_trials.index.values)

        if len(exclude_idxs) > 0:
            trial_df = trial_df.drop(exclude_idxs)
            print "Excluded {0} trials (duration outliers)".format(len(exclude_idxs))

        # Write to CSV
        with gzip.open("trials.csv.gz", "w") as out_file:
            trial_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_trials],
        "file_dep" : ["response_data.xml.gz"],
        "targets"  : ["trials.csv.gz"]
    }

# -------------------------------------------------- 

def task_line_categories():
    def make_categories():
        output_rows = []

        # Experiments
        for path in sorted(code_paths):
            file_name = os.path.splitext(os.path.split(path)[1])[0]
            base, version = file_name.split("_", 1)

            code_lines = open(path, "r").readlines()
            line_cats = eyecode.util.python_line_categories(code_lines)
            line_metrics = eyecode.util.python_token_metrics(code_lines)
            metric_cols = ["line_length", "keywords", "identifiers",
                           "operators", "whitespace_prop", "line_indent"]

            metrics_vals = line_metrics.sort("line")[metric_cols].values
            for i, (cats, metrics) in enumerate(zip(line_cats, metrics_vals)):
                output_rows.append(
                    [base, version, i + 1, ",".join(cats)] + list(metrics))

        columns = ["base", "version", "line", "categories"] + metric_cols
        cats_df = pandas.DataFrame(output_rows, columns=columns)

        with gzip.open("line_categories.csv.gz", "w") as out_file:
            cats_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_categories],
        "file_dep" : code_paths,
        "targets"  : ["line_categories.csv.gz"]
    }

# -------------------------------------------------- 

def task_trial_responses():
    def make_responses():
        rows = []
        responses = etree.parse("response_data.xml.gz")

        for e in responses.xpath("//experiment"):
            exp_id = int(e.attrib["id"])
            location = e.attrib["location"]

            # Need to fix timestamps for Mechanical Turk participants
            if location != "bloomington":
                continue

            for t in e.xpath(".//trial"):
                trial_id = int(t.attrib["id"])
                base = t.attrib["base"]
                version = t.attrib["version"]

                started = datetime.strptime(t.attrib["started"], TIME_FORMAT)
                started_timestamp = int(mktime(started.utctimetuple()) * 1e3)

                # Keystrokes
                for response in t.xpath("./responses/response"):
                    r_timestamp = int(response.attrib["timestamp"])
                    rows.append([exp_id, trial_id, base, version,
                        r_timestamp - started_timestamp, response.text])

        cols = ["exp_id", "trial_id", "base", "version", "time_ms", "response"]
        trial_df = pandas.DataFrame(rows, columns=cols)

        # Write to CSV
        with gzip.open("trial_responses.csv.gz", "w") as out_file:
            trial_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_responses],
        "file_dep" : ["response_data.xml.gz"],
        "targets"  : ["trial_responses.csv.gz"]
    }

# -------------------------------------------------- 

def task_json():
    raw_fixes = pandas.read_csv(gzip.open("raw_fixations.csv.gz", "r"))
    trial_ids = sorted(raw_fixes.trial_id.unique())

    json_paths = [os.path.join("js", "trials", "{0}.fixations.js".format(int(t_id)))
                  for t_id in trial_ids]

    aois_paths = [os.path.join("js", "trials", "{0}.aois.js".format(int(t_id)))
                  for t_id in trial_ids]

    resp_paths = [os.path.join("js", "trials", "{0}.responses.js".format(int(t_id)))
                  for t_id in trial_ids]

    def make_json():
        aois = pandas.read_csv(gzip.open("aois.csv.gz", "r"))

        # Write fixations and aois for each trial
        all_fixes = pandas.read_csv(gzip.open("all_fixations.csv.gz", "r"))
        trial_responses = pandas.read_csv(gzip.open("trial_responses.csv.gz", "r"))

        for t_id, js_path, aoi_path, resp_path in \
                zip(trial_ids, json_paths, aois_paths, resp_paths):

            t_fixes = all_fixes[all_fixes.trial_id == t_id]
            js_fixes = eyecode.aoi.fixations_to_json(t_fixes)

            t_aois = aois[aois.trial_id == t_id]
            js_aois = eyecode.aoi.aois_to_json(t_aois)

            t_resps = trial_responses[trial_responses.trial_id == t_id]
            js_resps = t_resps[["time_ms", "response"]].to_json(orient="records")

            js_dir = os.path.dirname(js_path)
            if not os.path.exists(js_dir):
                os.makedirs(js_dir)

            with open(js_path, "w") as js_file:
                js_file.write(pretty_json(js_fixes))

            with open(aoi_path, "w") as aoi_file:
                aoi_file.write(pretty_json(js_aois))

            with open(resp_path, "w") as resp_file:
                resp_file.write(pretty_json(js_resps))

    return {
        "actions"  : [make_json],
        "file_dep" : ["raw_fixations.csv.gz", "all_fixations.csv.gz", "aois.csv.gz"],
        "targets"  : json_paths + aois_paths + resp_paths
    }
