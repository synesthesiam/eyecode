import operator
import numpy as np
import pandas
from ..aoi import get_aoi_columns, get_aoi_kinds, kind_to_col, kinds_to_cols

# Fixation Metrics {{{

def fixation_count(fixes):
    return len(fixes)

def avg_fixation_duration(fixes):
    return fixes["duration_ms"].mean()

def fixes_per_trial(fixes_df):
    """Number of fixations by trial.
    
    Parameters
    ----------
    fixes_df : pandas DataFrame
        Fixations dataframe with exp_id and trial_id columns

    Returns
    -------
    df : pandas DataFrame
        DataFrame with number of fixations per experiment/trial

    """
    return fixes_df.groupby(["exp_id", "trial_id"]).size()

def fixes_ms_per_trial(fixes_df):
    """Mean fixation duration by trial.

    Parameters
    ----------
    fixes_df : pandas DataFrame
        Fixations dataframe with exp_id, trial_id, and duration_ms columns

    Returns
    -------
    df : pandas DataFrame
        DataFrame with mean fixation duration (ms) per experiment/trial

    """
    return fixes_df.groupby(["exp_id", "trial_id"])["duration_ms"].mean()

def fixes_per_aoi(aoi_fixes_df):
    """Number of fixations by AOI.

    Parameters
    ----------
    aoi_fixes_df : pandas DataFrame
        Fixations dataframe with AOI columns (hit names as values)

    Returns
    -------
    df : pandas DataFrame
        DataFrame with mean fixation duration (ms) per AOI kind/name

    """
    aoi_cols = get_aoi_columns(aoi_fixes_df)
    aoi_kinds = get_aoi_kinds(aoi_fixes_df)

    # Create new frame with a row for each AOI kind and hit name
    rows = []
    for _, fix in aoi_fixes_df.iterrows():
        for col, kind in zip(aoi_cols, aoi_kinds):
            rows.append([kind, fix[col]])

    num_fixes = pandas.DataFrame(rows, columns=["kind", "name"]).\
        groupby(["kind", "name"]).size()

    return num_fixes

def fixes_ms_per_aoi(aoi_fixes_df):
    """Total fixation duration per AOI.

    Parameters
    ----------
    aoi_fixes_df : pandas DataFrame
        Fixations dataframe with AOI columns/names and duration_ms

    Returns
    -------
    df : pandas DataFrame
        DataFrame with total fixation duration (ms) per AOI kind/name

    """
    aoi_cols = get_aoi_columns(aoi_fixes_df)
    aoi_kinds = get_aoi_kinds(aoi_fixes_df)

    # Create new frame with a row for each AOI/name with duration
    rows = []
    for _, fix in aoi_fixes_df.iterrows():
        for col, kind in zip(aoi_cols, aoi_kinds):
            time = 0 if pandas.isnull(fix[col]) else fix["duration_ms"]
            rows.append([kind, fix[col], time])

    tf_cols = ["kind", "name", "duration_ms"]
    time_fixes = pandas.DataFrame(rows, columns=tf_cols).\
        groupby(["kind", "name"])["duration_ms"].sum()

    return time_fixes

def first_fix_ms_aoi(aoi_fixes_df):
    """Time of first fixation on each AOI.

    Parameters
    ----------
    aoi_fixes_df : pandas DataFrame
        Fixations dataframe with AOI columns/names and start_ms

    Returns
    -------
    df : pandas DataFrame
        DataFrame with first fixation time (ms) per AOI kind/name

    """
    aoi_cols = get_aoi_columns(aoi_fixes_df)
    aoi_kinds = get_aoi_kinds(aoi_fixes_df)

    # Create a new frame with AOI kinds/names and first fixation time
    rows = []
    for col, kind in zip(aoi_cols, aoi_kinds):
        min_times = aoi_fixes_df[[col, "start_ms"]]\
                .groupby(col).start_ms.min()

        for name, time in min_times.iterkv():
            rows.append([kind, name, time])

    if len(rows) == 0:
        rows = None

    fst_cols = ["kind", "name", "start_ms"]
    first_df = pandas.DataFrame(rows, columns=fst_cols)

    # Re-index by AOI kind/name
    first_df.set_index(["kind", "name"], inplace=True)

    return first_df.start_ms

def fixation_saccade_proportion(fixations, saccades, group_col=None):
    """Calculates the proportion of fixation and saccade durations.

    Parameters
    ----------
    fixations : pandas DataFrame
        DataFrame with one row per fixation. Must have duration_ms column.

    saccades : pandas DataFrame
        DataFrame with one row per saccade. Must have duration_ms column.

    group_col : str, list, or None
        Column or list of columns by which to group fixations and saccades. If
        None, no grouping occurs (default: None).

    Returns
    -------
    prop : float or pandas Series
        Proportion of fixation and saccade durations. Float when group_col is
        None, Series otherwise.

    """
    if group_col is not None:
        fixations = fixations.groupby(group_col)
        saccades = saccades.groupby(group_col)

    return fixations.duration_ms.sum() / \
            saccades.duration_ms.sum().astype(float)


def scanpath_length(fixations, metric="euclidean"):
    """Length of fixation scanpath in pixels (i.e., sum of distances between
    fixations).

    Parameters
    ----------
    fixations : pandas DataFrame
        DataFrame with one fixation per row. Requires fix_x and fix_y columns.

    metric : str, optional
        Distance metric passed to scipy.spatial.distance.pdist. Default is
        "euclidean".

    Returns
    -------
    length : float
        Total length of fixation scanpath (pixels)

    See also
    --------
    scipy.spatial.distance.pdist

    """
    from scipy import spatial
    return spatial.distance.pdist(fixations[["fix_x", "fix_y"]], metric=metric).sum()

def avg_saccade_length(saccades):
    """Average saccade distance (Euclidean).

    Parameters
    ----------
    saccades : pandas DataFrame
        DataFrame with one saccade per row. Requires dist_euclid column.

    Returns
    -------
    length : float
        Average saccade distance

    """
    return saccades["dist_euclid"].mean()

def convex_hull_area(fixations):
    """Area of convex hull around all fixation points.

    Parameters
    ----------
    fixations : pandas DataFrame
        DataFrame with one row per fixation. Requires fix_x and fix_y columns.

    Returns
    -------
    area : float
        Area of convex hull around fixations

    Notes
    -----
    Requires the shapely module to be installed

    """
    from shapely.geometry import MultiPoint
    xy_tuple = [tuple(x) for _, x in fixations[["fix_x", "fix_y"]].iterrows()]
    return MultiPoint(xy_tuple).convex_hull.area

def convex_hull_envelope(fixations):
    """Envelope of convex hull around all fixation points.

    Parameters
    ----------
    fixations : pandas DataFrame
        DataFrame with one row per fixation. Requires fix_x and fix_y columns.

    Returns
    -------
    env : tuple of float
        Envelope bounds of convex hull around fixations (x1, y1, x2, y2)

    Notes
    -----
    Requires the shapely module to be installed

    """
    from shapely.geometry import MultiPoint
    xy_tuple = [tuple(x) for _, x in fixations[["fix_x", "fix_y"]].iterrows()]
    return MultiPoint(xy_tuple).envelope.bounds

def transition_density(trans_matrix, threshold=1):
    num_cells = reduce(operator.mul, trans_matrix.shape, 1)
    return sum(trans_matrix.flatten() >= threshold) / float(num_cells)

def spatial_density(fixations, grid_bbox=None, num_cols=10,
        num_rows=10, threshold=1, return_counts=False):
    if grid_bbox is None:
        # Compute bounding box from fixations positions
        min_x, min_y = np.floor(fixations.fix_x.min()),\
            np.floor(fixations.fix_y.min())
        max_x, max_y = np.ceil(fixations.fix_x.max()),\
            np.ceil(fixations.fix_y.max())
        grid_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    elif isinstance(grid_bbox, pandas.DataFrame):
        # Use first row of DataFrame
        grid_bbox = list(grid_bbox[["x", "y", "width", "height"]]\
                .values[0])
 
    # Compute cell size
    grid_width, grid_height = grid_bbox[2], grid_bbox[3]
    h_size, v_size = grid_width / float(num_cols), grid_height / float(num_rows)

    # Count number of fixations in each cell
    counts = np.zeros(shape=(num_cols, num_rows))
    for x, y in fixations[["fix_x", "fix_y"]].values:
        # Calculate which cell the fixation was inside
        grid_x = int((x - grid_bbox[0]) / h_size)
        grid_y = int((y - grid_bbox[1]) / v_size)
        if (0 <= grid_x < num_cols) and (0 <= grid_y < num_rows):
            counts[grid_x, grid_y] += 1

    # Compute density -- number of cells with fixations >= threshold
    density = sum(counts.flatten() >= threshold) / float(num_cols * num_rows)
    if return_counts:
        # Return both spatial density and count matrix
        return density, counts
    else:
        # Return just the spatial density
        return density

def rolling_window(frame, fun, window_size_ms, step_ms):
    start, end = 0, window_size_ms
    if not isinstance(fun, dict):
        fun = { "value" : fun }

    values = { k : [] for k, v in fun.iteritems() }
    times = []
    while start < frame.end_ms.max():
        times.append(start + (window_size_ms / 2))
        win_frame = frame[
                ((frame.start_ms >= start) & (frame.start_ms < end)) |
                ((frame.end_ms >= start) & (frame.end_ms < end))]

        for k, f in fun.iteritems():
            values[k].append(f(win_frame))

        start += step_ms
        end += step_ms

    df = pandas.DataFrame(values, index=times)
    return df

# }}}


def percent_fixations_on_aoi(aoi_fixations):
    num_fixes = fixations_per_aoi(aoi_fixations)
    return num_fixes.groupby(level=0).apply(lambda f: f / float(f.sum()))

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
    columns = kinds_to_cols(kinds)
    aoi_fixes = { k: fixations[[c, "start_ms"]].dropna() for (c, k) in zip(columns, kinds) }

    if aoi_names is None:
        aoi_names = { k: None for k in aoi_fixes.keys() }

    # Create dictionary of aoi names
    for kind, names in aoi_names.iteritems():
        col = kind_to_col(kind)
        if names is None or len(names) == 0:
            # Fill in missing aoi names
            aoi_names[kind] = aoi_fixes[kind][col].unique()
        else:
            # Filter out unwanted names
            f = aoi_fixes[kind]
            aoi_fixes[kind] = f[f[col].isin(names)]

    rows = []
    for kind, fixes in aoi_fixes.iteritems():
        col = kind_to_col(kind)
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
