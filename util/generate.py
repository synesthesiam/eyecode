import pandas, numpy as np
from ..aoi import make_aoi_column

def gen_fixations_with_aoi(xy_aoi, exp_id=0, trial_id=0, aoi_name="A",
        duration=150.0, saccade=50.0):

    aoi_col = make_aoi_column([aoi_name])
    df = pandas.DataFrame(xy_aoi, columns=["fix_x", "fix_y", aoi_col])
    df["exp_id"] = exp_id
    df["trial_id"] = trial_id
    df["fix_id"] = np.arange(len(df))
    df["duration_ms"] = duration
    df["start_ms"] = np.arange(len(df)) * (duration + saccade)
    df["end_ms"] = df["start_ms"] + df["duration_ms"]

    return df
