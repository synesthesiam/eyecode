import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas, numpy as np, json
import eyecode.aoi, eyecode.util
from StringIO import StringIO

# -------------------------------------------------- 

def task_raw_fixations():
    file_names = ["subject1_rectangle.txt", "subject2_rectangle.txt",
            "subject1_basketball.txt"]

    def make_raw():
        frames = []
        exp_trials = [(0, 8), (1, 9), (2, 10)]

        for file_name, (exp_id, trial_id) in zip(file_names, exp_trials):
            subject_fixes = pandas.read_csv(file_name, sep=r"\s+")\
                    .drop(["ID", "SubjectName", "TrialSequence", "CountInTrial"], axis=1)
            subject_fixes.columns = ["trial_id", "start_ms", "duration_ms", "fix_x", "fix_y"]
            subject_fixes["exp_id"] = exp_id
            subject_fixes["trial_id"] = trial_id
            subject_fixes["end_ms"] = subject_fixes["start_ms"] + subject_fixes["duration_ms"]
            subject_fixes["program"] = os.path.splitext(file_name)[0].split("_")[1]
            frames.append(subject_fixes)

        raw_fixes = pandas.concat(frames, ignore_index=True)
        raw_fixes.to_csv("raw_fixations.csv", index=False)

    return {
        "actions"  : [make_raw],
        "file_dep" : file_names,
        "targets"  : ["raw_fixations.csv"]
    }

# -------------------------------------------------- 

def task_all_fixations():
    def make_all():
        raw_fixations = pandas.read_csv("raw_fixations.csv")
        aois = pandas.read_csv("areas_of_interest.csv")
        frames = []
        for program, prog_fixes in raw_fixations.groupby("program"):
            prog_aois = aois[aois.program == program]
            fixes_df = eyecode.aoi.hit_test(prog_fixes, prog_aois)
            frames.append(fixes_df)

        all_fixes = pandas.concat(frames, ignore_index=True)
        all_fixes.to_csv("all_fixations.csv", index=False)

    return {
        "actions"  : [make_all],
        "file_dep" : ["raw_fixations.csv", "areas_of_interest.csv"],
        "targets"  : ["all_fixations.csv"]
    }

# -------------------------------------------------- 

def task_areas_of_interest():
    program_names = ["rectangle", "basketball"]
    img_paths = [os.path.join("images", "{0}.png".format(name)) for name in program_names]
    code_paths = [os.path.join("programs", "{0}.java".format(name)) for name in program_names]

    def make_aois():
        from PIL import Image
        frames = []

        for program, img_path, code_path in zip(program_names, img_paths, code_paths):

            # Compute line AOIs from image
            img = Image.open(img_path)
            aoi_df = eyecode.aoi.find_rectangles(img)
            aoi_df["program"] = program

            # Adjust line AOIs
            line_numbers = [i + 1 for i, s in enumerate(open(code_path, "r").readlines())
                            if len(s.strip()) > 0]

            line_map = { i + 1 : line_numbers[i]
                         for i in range(len(line_numbers)) }

            # Rename to match actual lines in the program
            for idx, row in aoi_df.iterrows():
                parts = row["name"].split(" ")
                new_line = line_map[int(parts[1])]
                new_name = "line {0}".format(new_line)
                if row["kind"] == "sub-line":
                    new_name += " part {0}".format(parts[-1])
                aoi_df.ix[idx, "name"] = new_name

            # Pad line rectangles and offset
            line_df = aoi_df[aoi_df.kind == "line"]
            padding_x = 8
            padding_y = 8 if program == "rectangle" else 3

            line_df.x -= padding_x
            line_df.width += padding_x * 2
            line_df.y -= padding_y
            line_df.height += padding_y * 2

            aoi_df.update(line_df)

            # Add derived AOIs
            new_aois = []

            # Entire code box
            code_box = eyecode.aoi.envelope(line_df) + [program, "interface", "code box"]
            new_aois.append(code_box)

            # Line chunks
            chunk_size = 25
            for n, x, y, w, h in line_df[["name", "x", "y", "width", "height"]].values:
                num_chunks = int(np.ceil(w / float(chunk_size)))
                for i in range(num_chunks):
                    ch_x = x + (i * chunk_size)
                    new_aois.append([ch_x, y, chunk_size, h, program, "line chunks",
                        "{0} chunk {1}".format(n, i + 1)])

            # Manual sub-line and signatures
            subline_df = aoi_df[aoi_df.kind == "sub-line"]

            # Pad sub-line rectangles
            padding_x = 0
            padding_y = 8 if program == "rectangle" else 3
            subline_df.x -= padding_x
            subline_df.width += padding_x * 2
            subline_df.y -= padding_y
            subline_df.height += padding_y * 2
            aoi_df.update(subline_df)

            def add_subline_aoi(line, parts, kind, name):
                names = ["line {0} part {1}".format(line, p) for p in parts]
                matching_aois = subline_df[subline_df.name.isin(names)]
                assert len(matching_aois) > 0, "{0} {1}".format(line, parts)
                env = eyecode.aoi.envelope(matching_aois) + [program, kind, name]
                new_aois.append(env)

            # Signatures, bodies of functions
            if program == "rectangle":

                # Block AOIs (derived from lines)
                names = ["attributes", "constructor", "width", "height", "area", "main"]
                lines  = [2, range(4, 10), 11, 13, 15, range(17, 23)]

                for n, l in zip(names, lines):
                    block_aois = eyecode.aoi.only_lines(line_df, l)
                    block = eyecode.aoi.envelope(block_aois) + [program, "block", n]
                    new_aois.append(block)

                # Sub-block AOIs (derived from lines)
                names = ["constructor body", "constructor signature", "main body", "main signature"]
                lines  = [range(5, 9), [4], range(18, 22), [17]]

                for n, l in zip(names, lines):
                    block_aois = eyecode.aoi.only_lines(line_df, l)
                    block = eyecode.aoi.envelope(block_aois) + [program, "sub-block", n]
                    new_aois.append(block)

                # Sub-line AOIs
                add_subline_aoi(4, [1], "signature", "constructor type")
                add_subline_aoi(4, [2], "signature", "constructor name")
                add_subline_aoi(4, range(3, 16), "signature", "constructor params")

                add_subline_aoi(11, range(1, 7), "sub-block", "width signature")
                add_subline_aoi(11, range(7, 12), "sub-block", "width body")
                add_subline_aoi(11, [1, 2], "signature", "width type")
                add_subline_aoi(11, [3], "signature", "width name")

                add_subline_aoi(13, range(1, 7), "sub-block", "height signature")
                add_subline_aoi(13, range(7, 12), "sub-block", "height body")
                add_subline_aoi(13, [1, 2], "signature", "height type")
                add_subline_aoi(13, [3], "signature", "height name")

                add_subline_aoi(15, range(1, 7), "sub-block", "area signature")
                add_subline_aoi(15, range(7, 16), "sub-block", "area body")
                add_subline_aoi(15, [1, 2], "signature", "area type")
                add_subline_aoi(15, [3], "signature", "area name")

                add_subline_aoi(17, [1, 2, 3], "signature", "main type")
                add_subline_aoi(17, [4], "signature", "main name")
                add_subline_aoi(17, range(5, 11), "signature", "main params")

                # Method calls
                add_subline_aoi(15, [8], "method-call", "area-width name")
                add_subline_aoi(15, [9, 10], "method-call", "area-width params")
                add_subline_aoi(15, [12], "method-call", "area-height name")
                add_subline_aoi(15, [13, 14], "method-call", "area-height params")

                add_subline_aoi(18, [5], "method-call", "main-constructor1 name")
                add_subline_aoi(18, range(6, 15), "method-call", "main-constructor1 params")

                add_subline_aoi(19, [1], "method-call", "main-print1 name")
                add_subline_aoi(19, range(2, 7), "method-call", "main-print1 params")

                add_subline_aoi(20, [5], "method-call", "main-constructor2 name")
                add_subline_aoi(20, range(6, 15), "method-call", "main-constructor2 params")

                add_subline_aoi(21, [1], "method-call", "main-print2 name")
                add_subline_aoi(21, range(2, 7), "method-call", "main-print2 params")

            elif program == "basketball":

                # Block AOIs (derived from lines)
                names = ["attributes", "main", "basketballSub", "tail"]
                lines  = [[2, 3], range(5, 9), range(10, 21), range(22, 29)]

                for n, l in zip(names, lines):
                    block_aois = eyecode.aoi.only_lines(line_df, l)
                    block = eyecode.aoi.envelope(block_aois) + [program, "block", n]
                    new_aois.append(block)

                # Sub-block AOIs (derived from lines)
                names = ["main body", "main signature", "basketballSub body", "basketballSub signature",
                         "tail body", "tail signature"]
                lines  = [[6, 7], [5], range(11, 20), [10], range(23, 28), [22]]

                for n, l in zip(names, lines):
                    block_aois = eyecode.aoi.only_lines(line_df, l)
                    block = eyecode.aoi.envelope(block_aois) + [program, "sub-block", n]
                    new_aois.append(block)

                # Sub-line AOIs
                add_subline_aoi(5, range(1, 4), "signature", "main type")
                add_subline_aoi(5, [4], "signature", "main name")
                add_subline_aoi(5, range(5, 11), "signature", "main params")

                add_subline_aoi(10, range(1, 4), "signature", "basketballSub type")
                add_subline_aoi(10, [4], "signature", "basketballSub name")
                add_subline_aoi(10, range(5, 12), "signature", "basketballSub params")

                add_subline_aoi(22, range(1, 6), "signature", "tail type")
                add_subline_aoi(22, [6], "signature", "tail name")
                add_subline_aoi(22, range(7, 14), "signature", "tail params")

                # Method calls
                add_subline_aoi(7, [1], "method-call", "basketballSub1 name")
                add_subline_aoi(7, range(2, 7), "method-call", "basketballSub1 params")

                add_subline_aoi(13, [3], "method-call", "tail name")
                add_subline_aoi(13, range(4, 8), "method-call", "tail params")

                add_subline_aoi(18, [1], "method-call", "basketballSub2 name")
                add_subline_aoi(18, range(2, 7), "method-call", "basketballSub2 params")

            # Combine manual and auto-generated AOIs
            aoi_df = aoi_df.append(pandas.DataFrame(new_aois,
                columns=["x", "y", "width", "height", "program", "kind", "name"]))

            frames.append(aoi_df)

        all_aois = pandas.concat(frames, ignore_index=True)
        all_aois.to_csv("areas_of_interest.csv", index=False)

    return {
        "actions"  : [make_aois],
        "file_dep" : img_paths + code_paths,
        "targets"  : ["areas_of_interest.csv"]
    }

# -------------------------------------------------- 

def task_line_fixations():
    def make_lines():
        offset_kind = "none"
        all_fixes = pandas.read_csv("all_fixations.csv")
        line_fixes = all_fixes[(all_fixes.offset_kind == offset_kind) &
                               -pandas.isnull(all_fixes.aoi_line)]

        line_fixes["line"] = line_fixes.aoi_line\
                .apply(lambda s: int(s.split(" ")[1]))

        drop_cols = ["offset_kind"] +\
                [c for c in line_fixes.columns if c.startswith("aoi_")]

        line_fixes = line_fixes.drop(drop_cols, axis=1)
        line_fixes.to_csv("line_fixations.csv", index=False)

    return {
        "actions"  : [make_lines],
        "file_dep" : ["all_fixations.csv"],
        "targets"  : ["line_fixations.csv"]
    }

# -------------------------------------------------- 

def task_json():
    trial_ids = [8, 9, 10]
    json_paths = [os.path.join("js", "trials", "{0}.fixations.js".format(int(t_id)))
                  for t_id in trial_ids]
    aois_path = os.path.join("js", "aois.js")

    def make_json():
        # Write AOIs
        aois = pandas.read_csv("areas_of_interest.csv")
        js_aois = eyecode.aoi.aois_to_json(aois)
        with open(aois_path, "w") as aois_file:
            aois_file.write(js_aois)

        # Write fixations
        all_fixes = pandas.read_csv("all_fixations.csv")
        for t_id, js_path in zip(trial_ids, json_paths):
            t_fixes = all_fixes[all_fixes.trial_id == t_id]
            js_fixes = eyecode.aoi.fixations_to_json(t_fixes)

            js_dir = os.path.dirname(js_path)
            if not os.path.exists(js_dir):
                os.makedirs(js_dir)
            with open(js_path, "w") as js_file:
                js_file.write(js_fixes)

    return {
        "actions"  : [make_json],
        "file_dep" : ["all_fixations.csv", "areas_of_interest.csv"],
        "targets"  : json_paths + [aois_path]
    }

# -------------------------------------------------- 

def task_code():
    trial_ids = [8, 9, 10]
    json_paths = [os.path.join("js", "trials", "{0}.tags.js".format(int(t_id)))
                  for t_id in trial_ids]
    xml_names = {
        8  : "rectangle_subject1.eaf",
        9  : "rectangle_subject2.eaf",
        10 : "basketball_subject1.eaf"
    }

    def make_codes():
        from urlparse import urlparse
        from lxml import etree
        parser = etree.XMLParser(remove_blank_text=True)

        all_fixes = pandas.read_csv("all_fixations.csv")
        text_order = pandas.read_csv("text_order.csv")
        exec_order = pandas.read_csv("exec_order.csv")

        urls = {
            8  : urlparse("file:///C:/Users/Mike/Documents/Education/Indiana University/Fall 2013/Gaze Workshop/rectangle_subject1.avi"),
            9  : urlparse("file:///C:/Users/Mike/Documents/Education/Indiana University/Fall 2013/Gaze Workshop/rectangle_subject2.avi"),
            10 : urlparse("file:///C:/Users/Mike/Documents/Education/Indiana University/Fall 2013/Gaze Workshop/basketball_subject1.avi"),
        }

        manual_tags = pandas.read_csv("manual_tags.csv")
        all_tags = []

        for t_id, js_path in zip(trial_ids, json_paths):
            t_fixes = all_fixes[all_fixes.trial_id == t_id]
            t_prog = t_fixes["program"].values[0]

            prog_text_order = text_order[text_order.program == t_prog]
            prog_exec_order = exec_order[exec_order.program == t_prog]
            tags_df = eyecode.aoi.code_fixations(t_fixes, prog_text_order, prog_exec_order)
            tags_df["trial_id"] = t_id

            t_manual_tags = manual_tags[manual_tags.trial_id == t_id]
            tags_df = pandas.concat([tags_df, t_manual_tags], ignore_index=True)

            all_tags.append(tags_df)

            js_dir = os.path.dirname(js_path)
            if not os.path.exists(js_dir):
                os.makedirs(js_dir)

            js_buf = StringIO()
            tags_df.to_json(js_buf, orient="records")

            with open(js_path, "w") as js_file:
                js_file.write(json.dumps(json.loads(js_buf.getvalue()), indent=4))

            xml_root = etree.parse("coding_base.eaf", parser).getroot()
            eyecode.aoi.make_xml_coding(xml_root, tags_df, urls[t_id])
            with open(xml_names[t_id], "w") as xml_file:
                xml_file.write(etree.tostring(xml_root, encoding="UTF-8",
                        xml_declaration=True, pretty_print=True))

        pandas.concat(all_tags, ignore_index=True).to_csv("tags.csv", index=False)

    return {
        "actions"  : [make_codes],
        "file_dep" : ["all_fixations.csv", "text_order.csv",
                      "exec_order.csv", "coding_base.eaf",
                      "manual_tags.csv"],
        "targets"  : json_paths + ["tags.csv"] + list(xml_names.values())
    }


