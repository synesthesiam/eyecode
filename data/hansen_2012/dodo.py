import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import gzip, pandas, json
import eyecode.aoi, eyecode.util
import numpy as np
from lxml import etree
from glob import glob
from collections import defaultdict
from datetime import datetime
from time import mktime

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
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

def task_aois():
    def make_aois():
        output_rows = []

        # Experiments
        for path in xml_paths:
            xml_node = etree.parse(path)
            exp = xml_node.xpath("/experiment")[0]
            exp_id = int(exp.attrib["id"])

            # Trials
            for trial in exp.xpath(".//trial"):
                trial_id = int(trial.attrib["id"])

                for aoi in trial.xpath("./areas-of-interest//aoi"):
                    output_rows.append([
                        exp_id, trial_id,
                        aoi.attrib["kind"], aoi.attrib["name"],
                        aoi.attrib["x"], aoi.attrib["y"],
                        aoi.attrib["width"], aoi.attrib["height"]
                    ])

        aois_df = pandas.DataFrame(output_rows, columns=("exp_id", "trial_id",
            "kind", "name", "x", "y", "width", "height"))

        with gzip.open("aois.csv.gz", "w") as out_file:
            aois_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_aois],
        "file_dep" : xml_paths,
        "targets"  : ["aois.csv.gz"]
    }

# -------------------------------------------------- 

def task_all_fixations():
    def make_polygon(aoi):
        from shapely.geometry import box
        x = int(aoi.attrib["x"])
        y = int(aoi.attrib["y"])
        width = int(aoi.attrib["width"])
        height = int(aoi.attrib["height"])
        return box(x, y, x + width, y + height)

    def make_all():
        from shapely.geometry import Point
        output_rows = []
        hit_kinds = { "point" : eyecode.aoi.hit_point, "circle" : eyecode.aoi.hit_circle }
        hit_radius = 20
        aoi_kinds = ["interface", "line", "syntax", "block"]

        # Experiments
        for path in xml_paths:
            xml_node = etree.parse(path)
            exp = xml_node.xpath("/experiment")[0]
            exp_id = int(exp.attrib["id"])

            # Trials
            for trial in exp.xpath(".//trial"):
                trial_id = int(trial.attrib["id"])
                offsets = trial.xpath(".//offset")

                aois = [a for a in trial.xpath(".//aoi")]
                aoi_groups = defaultdict(dict)
                
                for aoi in aois:
                    aoi_kind = aoi.attrib["kind"]
                    aoi_groups[aoi_kind][aoi] = make_polygon(aoi)

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

                            hit_names = []

                            # Do hit testing by AOI group to avoid overlapping of AOIs
                            for aoi_kind in aoi_kinds:
                                hit_name = ""

                                if aoi_kind in aoi_groups:
                                    aoi_polys = aoi_groups[aoi_kind]
                                    hit_aoi = hit_fun(fix_pt, aoi_polys, radius=hit_radius)
                                    if hit_aoi is not None:
                                        hit_name = hit_aoi.attrib["name"]                                        

                                hit_names.append(hit_name)

                            output_rows.append([
                                exp_id, trial_id,
                                trial.attrib["base"], trial.attrib["version"],
                                offset_kind, offset_x, offset_y, hit_kind,
                                int(fix.attrib["start"]), int(fix.attrib["end"]),
                                fix_x, fix_y
                            ] + hit_names)

        cols = ["exp_id", "trial_id", "base", "version",
                "offset_kind", "offset_x", "offset_y", "hit_kind",
                "start_ms", "end_ms",
                "fix_x", "fix_y"] + ["aoi_{0}".format(k) for k in aoi_kinds]

        fixes_df = pandas.DataFrame(output_rows, columns=cols)

        # Add duration column
        fixes_df["duration_ms"] = fixes_df.apply(lambda r: r["end_ms"] - r["start_ms"], axis=1)

        with gzip.open("all_fixations.csv.gz", "w") as out_file:
            fixes_df.to_csv(out_file, index=False)

    return {
        "actions"  : [make_all],
        "file_dep" : xml_paths,
        "targets"  : ["all_fixations.csv.gz"]
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
                    age, degree, gender, py_years, prog_years, cs_major])

        cols = ["trial_id", "exp_id", "base", "version", "grade_value",
                "grade_category", "started_ms", "ended_ms", "duration_ms",
                "keystroke_duration_ms", "keystroke_count", "keystroke_coefficient",
                "response_proportion", "response_corrections", "code_chars", "code_lines",
                "cyclo_comp", "hal_effort", "hal_volume", "output_chars", "output_lines",
                "true_output", "pred_output", "age", "degree", "gender",
                "py_years", "prog_years", "cs_major"]

        trial_df = pandas.DataFrame(rows, columns=cols)

        # Add derived columns
        trial_df["grade_perfect"] = trial_df.grade_value == 10
        trial_df["grade_correct"] = trial_df.grade_category.str.startswith("correct")
        trial_df["grade_common"] = trial_df.grade_category.str.startswith("common")

        trial_df["duration_sec"] = trial_df.duration_ms / 1000.0
        trial_df["duration_ms_log"] = np.log(trial_df.duration_ms)
        trial_df["duration_sec_log"] = np.log(trial_df.duration_ms) / 1000.0

        trial_df["program_name"] = trial_df.apply(lambda t: "{0}_{1}".format(t["base"], t["version"]), axis=1)

        # Add numeric representations of some columns
        for col in ["base", "version", "gender", "degree", "cs_major"]:
            values = list(trial_df[col].unique())
            trial_df[col + "_num"] = trial_df[col].apply(lambda v: values.index(v))

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
            for i, cats in enumerate(line_cats):
                output_rows.append([base, version, i+1, ",".join(cats)])

        cats_df = pandas.DataFrame(output_rows, columns=("base", "version",
            "line", "categories"))

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
