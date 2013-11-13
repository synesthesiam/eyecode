import pandas

def make_raw_fixations(xml_node):
    output_rows = []

    # Experiment
    exp = xml_node.xpath("/experiment")[0]
    exp_id = int(exp.attrib["id"])

    # Trials
    for trial in exp.xpath(".//trial"):
        trial_id = int(trial.attrib["id"])

        # Fixations
        for fix in trial.xpath(".//fixation"):
            fix_id = int(fix.attrib["id"])
            fix_x = int(fix.attrib["x"])
            fix_y = int(fix.attrib["y"])

            output_rows.append([
                exp_id, trial_id,
                fix_id,
                int(fix.attrib["start"]), int(fix.attrib["end"]),
                fix_x, fix_y
            ])

    fixes_df = pandas.DataFrame(output_rows, columns=("exp_id", "trial_id",
        "fix_id", "start_ms", "end_ms", "fix_x", "fix_y"))

    # Add duration column
    fixes_df["duration_ms"] = fixes_df.apply(lambda r: r["end_ms"] - r["start_ms"], axis=1)

    return fixes_df
