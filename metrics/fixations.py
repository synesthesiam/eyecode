import pandas, numpy as np
from ..aoi import get_aoi_columns, get_aoi_kinds, make_aoi_column, make_aoi_columns

def fixations_per_trial(fixations):
    """Number of fixations"""
    return fixations.groupby(["exp_id", "trial_id"]).size()

def fixation_ms_per_trial(fixations):
    return fixations.groupby("trial_id")["duration_ms"].mean()

def fixations_per_aoi(aoi_fixations, text_lengths=None):
    aoi_cols = get_aoi_columns(aoi_fixations)
    aoi_kinds = get_aoi_kinds(aoi_fixations)

    rows = []
    for _, fix in aoi_fixations.iterrows():
        for col, kind in zip(aoi_cols, aoi_kinds):
            rows.append([kind, fix[col]])

    num_fixes = pandas.DataFrame(rows, columns=["kind", "name"]).\
        groupby(["kind", "name"]).size()

    if text_lengths is None:
        return num_fixes
    else:
        # Indexes will be aoi names
        len_series = pandas.Series(text_lengths)

        # Missing aois will have NaN
        norm_fixes = num_fixes / len_series
        return norm_fixes.dropna()

def fixation_ms_per_aoi(aoi_fixations, text_lengths=None):
    aoi_cols = get_aoi_columns(aoi_fixations)
    aoi_kinds = get_aoi_kinds(aoi_fixations)

    rows = []
    for _, fix in aoi_fixations.iterrows():
        for col, kind in zip(aoi_cols, aoi_kinds):
            time = 0 if pandas.isnull(fix[col]) else fix["duration_ms"]
            rows.append([kind, fix[col], time])

    time_fixes = pandas.DataFrame(rows, columns=["kind", "name", "duration_ms"]).\
        groupby(["kind", "name"])["duration_ms"].sum()

    if text_lengths is None:
        return time_fixes
    else:
        # Indexes will be aoi names
        len_series = pandas.Series(text_lengths)

        # Missing aois will have NaN
        norm_fixes = time_fixes / len_series
        return norm_fixes.dropna()

def first_fixation_ms(aoi_fixations):
    aoi_cols = get_aoi_columns(aoi_fixations)
    aoi_kinds = get_aoi_kinds(aoi_fixations)

    rows = []
    for col, kind in zip(aoi_cols, aoi_kinds):
        min_times = aoi_fixations[[col, "start_ms"]].groupby(col).start_ms.min()
        for name, time in min_times.iterkv():
            rows.append([kind, name, time])

    if len(rows) == 0:
        rows = None

    first_df = pandas.DataFrame(rows, columns=["kind", "name", "start_ms"])
    first_df.set_index(["kind", "name"], inplace=True)

    return first_df.start_ms

def percent_fixations_on_aoi(aoi_fixations):
    num_fixes = fixations_per_aoi(aoi_fixations)
    return num_fixes.groupby(level=0).apply(lambda f: f / float(f.sum()))

def fixation_spatial_density(fixations, grid_bbox=None, num_cols=10, num_rows=10, threshold=1):
    if grid_bbox is None:
        min_x, min_y = np.floor(fixations.fix_x.min()), np.floor(fixations.fix_y.min())
        max_x, max_y = np.ceil(fixations.fix_x.max()), np.ceil(fixations.fix_y.max())
        grid_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        print grid_bbox

    grid_width, grid_height = grid_bbox[2], grid_bbox[3]
    h_size, v_size = grid_width / float(num_cols), grid_height / float(num_rows)

    counts = np.zeros(shape=(num_cols, num_rows))
    for x, y in fixations[["fix_x", "fix_y"]].values:
        grid_x = int((x - grid_bbox[0]) / h_size)
        grid_y = int((y - grid_bbox[1]) / v_size)
        if (0 <= grid_x < num_cols) and (0 <= grid_y < num_rows):
            counts[grid_x, grid_y] += 1

    density = sum(counts.flatten() >= threshold) / float(num_cols * num_rows)
    return density, counts

def lines_by_fixation(fixations):
    # Group fixations by trial
    trial_groups = fixations.groupby("trial_id")
    num_trials = len(trial_groups)
    assert num_trials > 0, "No trials"

    # Maximum number of lines fixated
    num_steps = trial_groups.agg(len).max()[0]

    # Fill in matrix of lines visited
    trial_ids = []
    visited = np.zeros(shape=(num_trials, num_steps), dtype=int)
    row = 0
    for trial_id, trial in trial_groups:
        trial_ids.append(trial_id)
        col = 0
        for line in trial.sort("start_ms").line.values:
            visited[row, col] = line
            col += 1
        row += 1

    return visited, trial_ids

def lines_by_time(fixations, num_steps=None, step_size=None):
    assert num_steps or step_size, "Must provide one of num_steps or step_size"

    # Group fixations by trial
    trial_groups = fixations.groupby("trial_id")
    num_trials = len(trial_groups)
    assert num_trials > 0, "No trials"

    # Calculate step info
    last_time = max(trial_groups.end_ms.max())

    if num_steps is None:
        num_steps = int(np.ceil(last_time / float(step_size)))
    elif step_size is None:
        step_size = int(np.ceil(last_time / float(num_steps)))

    # Fill in matrix of lines visited
    trial_ids = []
    all_times = np.arange(0, last_time + step_size, step=step_size)
    visited = np.zeros(shape=(num_trials, num_steps), dtype=int)
    row = 0
    for trial_id, trial in trial_groups:
        trial_ids.append(trial_id)
        col = 0
        last_time = 0
        for time in all_times[1:]:
            lines = trial[(last_time <= trial.start_ms) & (trial.end_ms < time)].line
            if len(lines) > 0:
                visited[row, col] = lines.value_counts().index[0]
            last_time = time
            col += 1
        row += 1

    return visited, trial_ids

def time_to_all_aois(fixations, aoi_names=None):
    kinds = get_aoi_kinds(fixations) if aoi_names is None else aoi_names.keys()
    columns = make_aoi_columns(kinds)
    aoi_fixes = { k: fixations[[c, "start_ms"]].dropna() for (c, k) in zip(columns, kinds) }

    if aoi_names is None:
        aoi_names = { k: None for k in aoi_fixes.keys() }

    # Create dictionary of aoi names
    for kind, names in aoi_names.iteritems():
        col = make_aoi_column(kind)
        if names is None or len(names) == 0:
            # Fill in missing aoi names
            aoi_names[kind] = aoi_fixes[kind][col].unique()
        else:
            # Filter out unwanted names
            f = aoi_fixes[kind]
            aoi_fixes[kind] = f[f[col].isin(names)]

    rows = []
    for kind, fixes in aoi_fixes.iteritems():
        col = make_aoi_column(kind)
        names_left = set(aoi_names[kind])
        time_to_all = np.NaN
        for _, f in fixes.sort("start_ms").iterrows():
            name = f[col]
            if name in names_left:
                names_left.remove(name)
                if len(names_left) == 0:
                    time_to_all = f["start_ms"]
                    break

        rows.append([kind, time_to_all])

    return pandas.DataFrame(rows, columns=["kind", "time_ms"])

def fixation_ms_proportion(aoi_fixations):
    ms_per_aoi = fixation_ms_per_aoi(aoi_fixations)
    total_ms = ms_per_aoi.groupby(level="kind").sum().astype(float)
    return ms_per_aoi.div(total_ms, level="kind")
