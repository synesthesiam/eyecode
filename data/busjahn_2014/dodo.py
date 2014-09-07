import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas, numpy as np, json
import eyecode.aoi, eyecode.util, eyecode.plot
from StringIO import StringIO

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
    program_names = ["nt1"]
    img_paths = [os.path.join("images", "{0}.png".format(name)) for name in program_names]

    def make_aois():
        from PIL import Image
        frames = []
        images = {}

        for program, img_path in zip(program_names, img_paths):

            # Compute line AOIs from image
            img = Image.open(img_path)
            images[program] = img

            aoi_df = eyecode.aoi.find_rectangles(img)
            aoi_df["program"] = program
            frames.append(aoi_df)

            # Pad line rectangles and offset
            line_df = aoi_df[aoi_df.kind == "line"]
            padding_x = 8
            padding_y = 8

            line_df.x -= padding_x
            line_df.width += padding_x * 2
            line_df.y -= padding_y
            line_df.height += padding_y * 2

            aoi_df.update(line_df)

            # Entire text box
            text_box = eyecode.aoi.envelope(line_df, kind="interface", name="text box")
            frames.append(text_box)

        all_aois = pandas.concat(frames, ignore_index=True)
        all_aois.to_csv("areas_of_interest.csv", index=False)

        for (program, kind), kind_aois in all_aois.groupby(["program", "kind"]):
            img = images[program]
            aoi_img = eyecode.plot.draw_rectangles(kind_aois, img)
            aoi_img.save(os.path.join("images", "{0}_{1}-aoi.png".format(program, kind)))

    return {
        "actions"  : [make_aois],
        "file_dep" : img_paths, 
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
    exp_trial_ids = [(1, 1), (4, 1), (7, 1)]
    fixes_paths = []
    aois_paths = []

    for exp_id, t_id in exp_trial_ids:
        fixes_paths.append(os.path.join("js", "trials",
            "{0}_{1}.fixations.js".format(exp_id, t_id)))

        aois_paths.append(os.path.join("js", "trials",
            "{0}_{1}.aois.js".format(exp_id, t_id)))

    def make_json():
        # Write AOIs
        aois = pandas.read_csv("areas_of_interest.csv")
        js_aois = eyecode.aoi.aois_to_json(aois)

        # Write fixations
        all_fixes = pandas.read_csv("all_fixations.csv")
        for (exp_id, t_id), (fix_path, aoi_path) in \
                zip(exp_trial_ids, zip(fixes_paths, aois_paths)):
            t_fixes = eyecode.util.filter_trial(all_fixes, exp_id, t_id)
            assert len(t_fixes) > 0, "No fixations for {0} {1}".format(exp_id, t_id)
            js_fixes = eyecode.aoi.fixations_to_json(t_fixes)

            js_dir = os.path.dirname(fix_path)
            if not os.path.exists(js_dir):
                os.makedirs(js_dir)

            with open(fix_path, "w") as js_file:
                js_file.write(js_fixes)

            with open(aoi_path, "w") as aois_file:
                aois_file.write(js_aois)

    return {
        "actions"  : [make_json],
        "file_dep" : ["all_fixations.csv", "areas_of_interest.csv"],
        "targets"  : fixes_paths + aois_paths
    }

