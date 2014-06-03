import numpy as np, itertools as it, pandas
import scipy.stats
import matplotlib
import matplotlib.colors
from pretty_plot import shade_axis
from matplotlib import ticker as ticker
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from kelly_colors import kelly_colors
from pandas.tools.plotting import boxplot_frame_groupby

PIE_COLORS = [
    "#33FDC0",
    "#86BCFF",
    "#FF8A8A",
    "#FFFFAA",
    "#AAFFFF"
]

def grade_pie(trials_df, ax=None, figsize=None, colors=PIE_COLORS, text_size=16):
    tight = False

    if ax is None:
        if figsize is None:
            figsize = (5, 5)
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()
        tight = True

    perfect = sum(trials_df["grade_value"] == 10)
    correct = sum(trials_df["grade_value"] >= 7) - perfect
    common_error = sum(trials_df["grade_category"].str.startswith("common"))
    incorrect = len(trials_df) - perfect - correct - common_error

    bins = [perfect, correct, incorrect, common_error]

    patches, _, _ = ax.pie(bins, autopct="%1.1f%%", shadow=False, colors=colors)
    shade_axis(ax, size=text_size)

    if tight:
        ax.figure.tight_layout()

    ax.legend(patches, ["Perfect", "Correct", "Incorrect", "Common Error"],
              loc="lower left", ncol=2)

    return ax

def grade_pie_versions(responses, axes=None, figsize=None,
        colors=PIE_COLORS, **kwargs):
    base = responses.iloc[0]["base"]
    versions = sorted(responses.version.unique())
    num_cols = len(versions) + 1

    if axes is None:
        if figsize is None:
            figsize = (num_cols * 6, 5)
        fig, axes = pyplot.subplots(nrows=1, ncols=num_cols, figsize=figsize)

    assert len(axes) >= num_cols, "Not enough axes"

    ax = axes[0]
    grade_pie(responses, ax=ax, colors=colors, **kwargs)
    ax.set_title("{0} (all)".format(base))

    for v, ax, color in zip(versions, axes[1:], it.cycle(colors)):
        grade_pie(responses[responses.version == v], ax=ax,
                colors=colors, **kwargs)
        ax.set_title("{0} ({1})".format(base, v))

    return axes

def feature_importances(importances, ax=None, figsize=None):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    importances = importances.sort("importance", ascending=True)
    #colors = ["g" if corr > 0 else "r" for corr in importances["label_corr"]]
    y = np.arange(len(importances))
    ax.barh(y, importances.importance.values, color=kelly_colors, ecolor="black",
            xerr=importances.importance_std.values, align="center")

    max_i = importances.importance.argmax()
    max_row = importances.iloc[max_i]

    ax.set_xlim(0, max_row["importance"] + max_row["importance_std"] + 0.1)
    ax.set_ylim([-1, len(importances)])
    ax.set_yticks(y)
    ax.set_yticklabels(importances.column.values)
    ax.grid()

    return ax

def cross_validation(frame, ax=None, figsize=None):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    boxplot_frame_groupby(frame.groupby("classifier"), subplots=False, ax=ax)
    ax.set_xticklabels(sorted(frame.classifier.unique()))
    ax.set_title("CV Scores by Classifier")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("CV Score")

    return ax

def column_distributions(responses, column, xlabel=None, title_prefix=None,
        axes=None, figsize=None, colors=kelly_colors, xlim=None, counts=False):
    versions = sorted(responses.version.unique())
    num_cols = len(versions) + 1

    if axes is None:
        if figsize is None:
            figsize = (num_cols * 4.5, 4.5)
        fig, axes = pyplot.subplots(nrows=1, ncols=num_cols, figsize=figsize)

    assert len(axes) >= num_cols, "Not enough axes"

    if title_prefix is None:
        title_prefix = column

    ax = axes[0]
    if counts:
        responses[column].value_counts().sort_index()\
                .plot(kind="bar", ax=ax)
    else:
        responses[column].hist(ax=ax)
    ax.set_title("{0} (all)".format(title_prefix))

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    for v, ax, color in zip(versions, axes[1:], it.cycle(colors)):
        if counts:
            responses[responses.version == v][column].value_counts()\
                    .sort_index().plot(kind="bar", ax=ax, color=color)
        else:
            responses[responses.version == v][column].hist(ax=ax, color=color)
        ax.set_title("{0} ({1})".format(title_prefix, v))

        if xlabel:
            ax.set_xlabel(xlabel)

    if xlim is not None:
        for ax in axes:
            ax.set_xlim(xlim)

    if counts:
        # Undo x tick label rotation
        for ax in axes:
            pyplot.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    return axes

def grade_distributions(responses, **kwargs):
    return column_distributions(responses, "grade_value",
            title_prefix="Trial Grades",
            xlabel="Grade", xlim=(0, 10), **kwargs)

def output_dist_distributions(responses, col="correct_dist_norm", **kwargs):
    return column_distributions(responses, col,
            title_prefix="Norm. Output Distance",
            xlabel="Output Distance", xlim=(0, 1), **kwargs)

def duration_distributions(responses, log=False, **kwargs):
    resp_copy = responses[["version", "duration_ms"]]
    resp_copy["duration_sec"] = responses.duration_ms / 1000.0
    if log:
        resp_copy["duration_sec"] = np.log(resp_copy["duration_sec"])
    return column_distributions(resp_copy, "duration_sec",
            title_prefix="Trial Times",
            xlabel="Trial Time ({0}sec)".format("log " if log else ""), **kwargs)

def python_experience_distributions(responses, **kwargs):
    return column_distributions(responses, "py_years",
            title_prefix="Python Experience",
            xlabel="Years of Python Experience", **kwargs)

def programming_experience_distributions(responses, **kwargs):
    return column_distributions(responses, "prog_years",
            title_prefix="Programming Experience",
            xlabel="Years of Programming Experience", **kwargs)

def keycoeff_distributions(responses, log=False, **kwargs):
    if log:
        responses = responses.copy()
        responses["keystroke_coefficient"] = np.log(responses["keystroke_coefficient"])
    return column_distributions(responses, "keystroke_coefficient",
            title_prefix="Trial Key Coeff.",
            xlabel="Keystroke Coefficient{0}".format(" (log)" if log else ""), **kwargs)

def respprop_distributions(responses, **kwargs):
    return column_distributions(responses, "response_proportion",
            title_prefix="Trial Resp. Prop.",
            xlabel="Response Proportion", **kwargs)

def respcorr_distributions(responses, **kwargs):
    return column_distributions(responses, "response_corrections",
            title_prefix="Trial Resp. Corr.",
            xlabel="Response Corrections", counts=True, **kwargs)


def total_grades_distribution(trials, ax=None, figsize=None, **kwargs):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    num_experiments = len(trials.exp_id.unique())
    total_grades = trials.groupby("exp_id").grade_value.sum()
    total_grades.hist(ax=ax, **kwargs)

    ax.set_title("Exp. Grade Distribution ({0} experiments)".format(num_experiments))
    ax.set_xlabel("Experiment Grade (10 trials)")
    ax.set_xlim(0, 100)
    return ax

def trial_grades_distribution(trials, ax=None, figsize=None, **kwargs):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    trials.grade_value.hist(ax=ax, **kwargs)

    ax.set_title("Trial Grade Distribution ({0} trials)".format(len(trials)))
    ax.set_xlabel("Trial Grade (0-10)")
    ax.set_xlim(0, 10)
    return ax

def total_duration_distribution(experiments, ax=None, figsize=None, **kwargs):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    experiments.duration_sec.hist(ax=ax, **kwargs)

    ax.set_title("Exp. Duration Distribution ({0} experiments)".format(len(experiments)))
    ax.set_xlabel("Experiment Duration (sec)")
    return ax

def trial_duration_distribution(trials, ax=None, figsize=None,
        column="duration_sec", **kwargs):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    durations = trials[column]
    durations.hist(ax=ax, **kwargs)

    ax.set_title("Trial Duration Distribution ({0} trials)".format(len(trials)))
    ax.set_xlabel("Trial Duration (sec)")
    return ax

def trial_keycoeff_distribution(trials, ax=None, figsize=None, **kwargs):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    trials.keystroke_coefficient.hist(ax=ax, **kwargs)

    ax.set_title("Trial Key Coeff Distribution ({0} trials)".format(len(trials)))
    ax.set_xlabel("Trial Keystroke Coefficient")
    return ax

def total_keycoeff_distribution(trials, ax=None, figsize=None, **kwargs):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    num_experiments = len(trials.exp_id.unique())
    mean_coeffs = trials.groupby("exp_id").keystroke_coefficient.mean()
    mean_coeffs.hist(ax=ax, **kwargs)

    ax.set_title("Exp. Key Coeff Distribution ({0} experiments)".format(num_experiments))
    ax.set_xlabel("Mean Experiment Keystroke Coefficient")
    return ax

def trial_respprop_distribution(trials, ax=None, figsize=None, **kwargs):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    trials.response_proportion.hist(ax=ax, **kwargs)

    ax.set_title("Trial Resp. Prop. Distribution ({0} trials)".format(len(trials)))
    ax.set_xlabel("Trial Response Proportion")
    ax.set_xlim(0, 1)
    return ax

def total_respprop_distribution(trials, ax=None, figsize=None, **kwargs):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    num_experiments = len(trials.exp_id.unique())
    mean_props = trials.groupby("exp_id").response_proportion.mean()
    mean_props.hist(ax=ax, **kwargs)

    ax.set_title("Exp. Resp. Prop. Distribution ({0} experiments)".format(num_experiments))
    ax.set_xlabel("Mean Experiment Response Proportion")
    ax.set_xlim(0, 1)
    return ax

def grades_by_base(responses, axes=None, figsize=None, colors=kelly_colors):
    tight = False
    bases = sorted(responses.base.unique())

    if axes is None:
        fig, _ = pyplot.subplots(nrows=2, ncols=5, figsize=figsize)
        fig.suptitle("Grade Distributions by Program")
        axes = fig.axes
        tight = True

    for ax, base, color in zip(axes, bases, it.cycle(colors)):
        grades = responses[responses.base == base].grade_value
        grades.hist(ax=ax, color=color)
        ax.set_title("{0}, {1} trials".format(base, len(grades)))
        ax.set_xlabel("Grade")
        ax.set_xlim(0, 10)

    if tight:
        fig = axes[0].figure
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, hspace=0.25)

    return axes

def output_dists_by_base(responses, col="perfect_dist_norm",
        axes=None, figsize=None, colors=kelly_colors):
    tight = False
    bases = sorted(responses.base.unique())

    if axes is None:
        fig, _ = pyplot.subplots(nrows=2, ncols=5, figsize=figsize)
        fig.suptitle("Output Distance Distributions by Program")
        axes = fig.axes
        tight = True

    for ax, base, color in zip(axes, bases, it.cycle(colors)):
        dists = responses[responses.base == base][col]
        dists.hist(ax=ax, color=color)
        ax.set_title("{0}, {1} trials".format(base, len(dists)))
        ax.set_xlabel("Output Dist.")
        ax.set_xlim(0, 1)

    if tight:
        fig = axes[0].figure
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, hspace=0.25)

    return axes

def durations_by_base(trials, axes=None, figsize=None, colors=kelly_colors,
        norm_by_lines=False, log=False):
    tight = False
    bases = sorted(trials.base.unique())

    if axes is None:
        fig, _ = pyplot.subplots(nrows=2, ncols=5, figsize=figsize)
        extra_title = " (normalized)" if norm_by_lines else ""
        fig.suptitle("Duration Distributions by Program{0}".format(extra_title))
        axes = fig.axes
        tight = True

    for ax, base, color in zip(axes, bases, it.cycle(colors)):
        # Convert to seconds
        b_trials = trials[trials.base == base]
        durations = b_trials.duration_ms / 1000.0

        if log:
            durations = np.log(durations)

        if norm_by_lines:
            durations /= b_trials.code_lines

        durations.hist(ax=ax, color=color)
        ax.set_title("{0}, {1} trials".format(base, len(b_trials)))

        if log:
            ax.set_xlabel("Trial Duration (log sec)")
        else:
            ax.set_xlabel("Trial Duration (sec)")

        if not norm_by_lines:
            max_time = np.ceil(durations.max()) + 1
            ax.set_xlim(0, max_time)

    if tight:
        fig = axes[0].figure
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, hspace=0.25)

    return axes

def keystrokes_by_base(trials, axes=None, figsize=None, colors=kelly_colors, norm_by_chars=False):
    tight = False
    bases = sorted(trials.base.unique())
    max_ks = trials.keystroke_count.max()

    if axes is None:
        fig, _ = pyplot.subplots(nrows=2, ncols=5, figsize=figsize)
        extra_title = " (normalized)" if norm_by_chars else ""
        fig.suptitle("Keystroke Distributions by Program{0}".format(extra_title))
        axes = fig.axes
        tight = True

    for ax, base, color in zip(axes, bases, it.cycle(colors)):
        b_trials = trials[trials.base == base]
        keystrokes = b_trials.keystroke_count

        if norm_by_chars:
            keystrokes /= b_trials.output_chars

        keystrokes.hist(ax=ax, color=color)
        ax.set_title("{0}, {1} trials".format(base, len(b_trials)))
        ax.set_xlabel("Keystrokes")

        if not norm_by_chars:
            ax.set_xlim(0, max_ks)

    if tight:
        fig = axes[0].figure
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, hspace=0.25)

    return axes

def keystroke_coefficient_by_base(trials, axes=None, figsize=None, colors=kelly_colors):
    tight = False
    bases = sorted(trials.base.unique())

    if axes is None:
        fig, _ = pyplot.subplots(nrows=2, ncols=5, figsize=figsize)
        fig.suptitle("Keystroke Coefficient by Program")
        axes = fig.axes
        tight = True

    for ax, base, color in zip(axes, bases, it.cycle(colors)):
        b_trials = trials[trials.base == base]
        ks_eff = b_trials.keystroke_coefficient

        ks_eff.hist(ax=ax, color=color)
        ax.set_title("{0}, {1} trials".format(base, len(b_trials)))
        ax.set_xlabel("Keystroke Coefficient")

    if tight:
        fig = axes[0].figure
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, hspace=0.25)

    return axes

def response_proportion_by_base(trials, axes=None, figsize=None, colors=kelly_colors):
    tight = False
    bases = sorted(trials.base.unique())

    if axes is None:
        fig, _ = pyplot.subplots(nrows=2, ncols=5, figsize=figsize)
        fig.suptitle("Response Proportion by Program")
        axes = fig.axes
        tight = True

    for ax, base, color in zip(axes, bases, it.cycle(colors)):
        b_trials = trials[trials.base == base]
        rs_prop = b_trials.response_proportion

        rs_prop.hist(ax=ax, color=color)
        ax.set_title("{0}, {1} trials".format(base, len(b_trials)))
        ax.set_xlabel("Response Proportion")
        ax.set_xlim(0, 1)

    if tight:
        fig = axes[0].figure
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, hspace=0.25)

    return axes

def response_corrections_by_base(trials, axes=None, figsize=None, colors=kelly_colors):
    tight = False
    bases = sorted(trials.base.unique())
    max_corr = max(trials.response_corrections.value_counts().index)

    if axes is None:
        fig, _ = pyplot.subplots(nrows=2, ncols=5, figsize=figsize)
        fig.suptitle("Response Corrections by Program")
        axes = fig.axes
        tight = True

    for ax, base, color in zip(axes, bases, it.cycle(colors)):
        b_trials = trials[trials.base == base]
        rs_corr = b_trials.response_corrections.value_counts()

        # Add 0 counts
        for i in range(max_corr + 1):
            to_add = {}
            if i not in rs_corr.index:
                to_add[i] = 0
            if len(to_add) > 0:
                rs_corr = rs_corr.append(pandas.Series(to_add))

        rs_corr = rs_corr.sort_index()
        rs_corr.plot(kind="bar", ax=ax, color=color)
        ax.set_title("{0}, {1} trials".format(base, len(b_trials)))
        ax.set_xlabel("Response Corrections")
        pyplot.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    if tight:
        fig = axes[0].figure
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, hspace=0.25)

    return axes

def hist_by_base(frame, column, base_fun=None, title=None, axes=None, figsize=None, colors=kelly_colors):
    tight = False
    bases = sorted(frame.base.unique())

    if axes is None:
        fig, _ = pyplot.subplots(nrows=2, ncols=5, figsize=figsize)
        if title is not None:
            fig.suptitle(title)
        axes = fig.axes
        tight = True

    for ax, base, color in zip(axes, bases, it.cycle(colors)):
        b_frame = frame[frame.base == base]
        b_frame[column].hist(ax=ax, color=color)
        if base_fun is not None:
            base_fun(b_frame, ax)
        else:
            ax.set_title("{0} ({1} items)".format(base, len(b_frame)))

    if tight:
        fig = axes[0].figure
        fig.tight_layout()
        fig.subplots_adjust(top=0.90, hspace=0.25)

    return axes

def correlation_scatter(responses, column_1, column_2, ax=None, figsize=None):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    ax.scatter(responses[column_1], responses[column_2])

    return ax

def demographics(experiments, font_family=["Arial"], colors=PIE_COLORS,
        text_size=14, title_size=18, figsize=(16, 9)):

    fig = pyplot.figure(figsize=figsize)

    # Bin and plot ages
    ax = pyplot.subplot(2, 3, 1)
    ax.set_title("Ages", family=font_family, size=title_size)
    ages = experiments["age"]
    age_bins = [0, 0, 0, 0, 0]
    age_bins[0] = len(ages[ages <= 20])
    age_bins[1] = len(ages[(20 < ages) & (ages < 25)])
    age_bins[2] = len(ages[(25 <= ages) & (ages <= 30)])
    age_bins[3] = len(ages[(30 < ages) & (ages <= 35)])
    age_bins[4] = len(ages[35 < ages])

    ax.pie(age_bins, labels=["18-20", "20-24", "25-30", "31-35", "> 35"],
            autopct="%1.1f%%", shadow=False, colors=colors)

    shade_axis(ax, size=text_size)

    # Bin and plot Python experience
    ax = pyplot.subplot(2, 3, 2)
    ax.set_title("Years of\nPython Experience", family=font_family, size=title_size)
    py = experiments["py_years"]
    py_bins = [0, 0, 0, 0, 0]
    py_bins[0] = len(py[py < .5])
    py_bins[1] = len(py[(.5 <= py) & (py <= 1)])
    py_bins[2] = len(py[(1 < py) & (py <= 2)])
    py_bins[3] = len(py[(2 < py) & (py <= 5)])
    py_bins[4] = len(py[5 < py])

    ax.pie(py_bins, labels=["< 1/2", "1/2-1", "1-2", "2-5", "> 5"],
            autopct="%1.1f%%", shadow=False, colors=colors)

    shade_axis(ax, size=text_size)

    # Bin and plot programming experience
    ax = pyplot.subplot(2, 3, 3)
    ax.set_title("Years of\nProgramming Experience", family=font_family, size=title_size)
    prog = experiments["prog_years"]
    prog_bins = [0, 0, 0, 0, 0]
    prog_bins[0] = len(prog[prog < 2])
    prog_bins[1] = len(prog[(2 <= prog) & (prog <= 3)])
    prog_bins[2] = len(prog[(3 < prog) & (prog <= 5)])
    prog_bins[3] = len(prog[(5 < prog) & (prog <= 10)])
    prog_bins[4] = len(prog[10 < prog])

    ax.pie(prog_bins, labels=["< 2", "2-3", "3-5", "5-10", "> 10"],
            autopct="%1.1f%%", shadow=False, colors=colors)

    shade_axis(ax, size=text_size)

    # Bin and plot education
    ax = pyplot.subplot(2, 3, 4)
    ax.set_title("Highest Degree\nReceived", family=font_family, size=title_size)
    degrees = experiments["degree"].value_counts()

    ax.pie(degrees.values, labels=[x.capitalize() for x in degrees.keys()],
            autopct="%1.1f%%", shadow=False, colors=colors)

    shade_axis(ax, size=text_size)

    # Bin and plot gender
    ax = pyplot.subplot(2, 3, 5)
    ax.set_title("Gender", family=font_family, size=title_size)
    genders = experiments["gender"].value_counts()

    ax.pie(genders.values, labels=[x.capitalize() for x in genders.keys()],
            autopct="%1.1f%%", shadow=False, colors=colors)

    shade_axis(ax, size=text_size)

    # Bin and plot CS major
    ax = pyplot.subplot(2, 3, 6)
    ax.set_title("CS Major", family=font_family, size=title_size)
    cs_majors = experiments["cs_major"].value_counts()

    ax.pie(cs_majors.values, labels=[x.capitalize() for x in cs_majors.keys()],
            autopct="%1.1f%%", shadow=False, colors=colors)

    shade_axis(ax, size=text_size)

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, wspace=0.5)

    return fig

def plot3d_views(plot_fun, view_angles=np.array([[-120, -90, -60, -30], [0, 30, 60, 90]]), yaw=20, fig=None, figsize=None):
    if fig is None:
        fig = pyplot.figure(figsize=figsize)

    num_rows = view_angles.shape[0]
    num_cols = view_angles.shape[1]

    for i, angle in enumerate(view_angles.flatten()):
        ax = fig.add_subplot(num_rows, num_cols, i+1, projection="3d")
        ax.view_init(yaw, angle)
        plot_fun(ax)

    return fig

def render_code(lines, font_path, font_size, image=None, line_height=1.0, offset=(0, 0), fill="#000000"):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, font_size)

    if image is None:
        temp_image = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(temp_image)
        max_x, max_y = offset
        for line in lines:
            line_size = draw.textsize(line.rstrip(), font=font)
            max_x = max(max_x, offset[0] + line_size[0])
            max_y += font_size * line_height

        max_x = int(np.ceil(max_x))
        max_y = int(np.ceil(max_y))
        image = Image.new("RGB", (max_x, max_y), "#FFFFFF")
        del draw, temp_image

    draw = ImageDraw.Draw(image)

    y = offset[1]
    for i, line in enumerate(lines):
        x = offset[0]
        draw.text((x, y), line.rstrip(), font=font, fill=fill)
        y += font_size * line_height

    del draw
    return image

def add_line_numbers(image, font_path, font_size, lines, line_height=1.0, offset=(0, 0), fill="#000000"):
    from PIL import ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, font_size)

    draw = ImageDraw.Draw(image)

    y = offset[1]
    for i, line in enumerate(lines):
        x = offset[0]
        draw.text((x, y), str(line), font=font, fill=fill)
        y += font_size * line_height

    del draw
    return image

def importances_and_crossval(importances, cross_val, label, axes=None,
        figsize=None, regressor=False, cv=10, repeat=1):

    if axes is None:
        fig, axes = pyplot.subplots(1, 2, figsize=figsize)

    # Feature importance
    feature_importances(importances, ax=axes[0])
    axes[0].set_title("Feature Importances ({0})".format(label))

    # Cross-validation
    cross_validation(cross_val, ax=axes[1])
    if regressor:
        axes[1].set_title("$R^2$ ({0}, CV={1}, {2}x)".format(label, cv, repeat))
    else:
        axes[1].set_title("AUC ({0}, CV={1}, {2}x)".format(label, cv, repeat))
        
    return axes

def classify_boxplots(frame, columns, labels, figsize=None, regressor=False):
    from .. import classify
    rows = len(labels)

    if figsize is None:
        figsize = (15, 5 * rows)

    fig, axes = pyplot.subplots(rows, 2, figsize=figsize)
    if len(labels) == 1:
        axes = np.reshape(axes, (1, 2))

    for i, label in enumerate(labels):
        # Feature importance
        fi_df = classify.feature_importances(frame, columns, label, regressor=regressor)
        ax = feature_importances(fi_df, ax=axes[i, 0])
        ax.set_title("Feature Importances ({0})".format(label))

        # Cross-validation
        cv_df = classify.cross_validation(frame, columns, label, regressor=regressor)
        ax = cross_validation(cv_df, ax=axes[i, 1])
        if regressor:
            ax.set_title("$R^2$ ({0}, CV=10)".format(label))
        else:
            ax.set_title("AUC ({0}, CV=10)".format(label))
        
    fig.tight_layout()
    return fig

def grade_correlations(trials, ax=None, figsize=None, label_size="small"):
    from .. import util
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    bases = sorted(list(trials.base.unique()))
    rows = []
    for exp_id, exp_trials in trials.groupby("exp_id"):
        row = [exp_id]
        for b in bases:
            t = util.filter_program(exp_trials, b)
            if len(t) > 0:
                row.append(t.grade_value.values[0])
            else:
                row.append(np.NaN)
        rows.append(row)

    grades = pandas.DataFrame(rows, columns=["exp_id"] + bases).dropna()

    #plt.subplots_adjust(left=0.15, bottom=0.20)
    ax.set_title("Grade Correlations")

    corr2d = np.zeros(shape=(len(bases), len(bases)))
    sig2d = np.empty(shape=(len(bases), len(bases)), dtype=object)
    num_steps = 100

    for i, b1 in enumerate(bases):
        for j, b2 in enumerate(bases):
            rp = scipy.stats.pearsonr(grades[b1], grades[b2])

            corr2d[i, j] = num_steps + (num_steps * rp[0])
            if (b1 != b2) and rp[1] < 0.05:
                sig2d[i, j] = ".{0:.0f}{1}".format(rp[0] * 100, util.significant(rp[1]))

    cdict = { "blue" : [(0.0, 0.0, 0.0),
                        (0.5, 1.0, 1.0),
                        (1.0, 0.0, 0.0)],

              "red"    : [(0.0, 0.0, 0.0),
                          (0.5, 1.0, 1.0),
                          (1.0, 1.0, 1.0)],
              
              "green" : [(0.0, 1.0, 1.0),
                         (0.5, 1.0, 1.0),
                         (1.0, 0.0, 0.0)]}

    cmap = matplotlib.colors.LinearSegmentedColormap("heat", cdict, N=(num_steps * 2))
    pyplot.pcolor(corr2d, cmap=cmap, edgecolors="#000000", vmin=0, vmax=(num_steps * 2))
    #plt.grid()
    
    for i in range(len(bases)):
        for j in range(len(bases)):
            if (i != j) and sig2d[i, j] is not None:
                pyplot.text(i + 0.5, j + 0.5, sig2d[i, j],
                         horizontalalignment="center",
                         verticalalignment="center",
                         size=label_size)

    cb = pyplot.colorbar(ticks=[0, num_steps, num_steps * 2],
                     format=ticker.FixedFormatter(["-1", "0", "1"]))

    cb.set_label("Pearson Correlation")

    # Label columns
    loc = ticker.FixedLocator([0.5 + x for x in range(len(bases))])
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    tic = ticker.FixedFormatter(bases)
    ax.xaxis.set_major_formatter(tic)
    ax.yaxis.set_major_formatter(tic)

    pyplot.xticks(rotation=90)
    fig = ax.figure
    fig.tight_layout()

    return ax

def get_fit_name(n):
    n = n[n.rfind("[")+1:]
    if n.endswith("]"):
        n = n[:-1]
    if (len(n) > 3) and (n[1] == "."):
        n = n[2:]
    return n

def fit_coefficients(fit, ax=None, figsize=None, skip_intercept=False):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    means = fit.conf_int().apply(lambda x: np.mean(x), axis=1)
    err = fit.conf_int().apply(lambda x: (x[1] - x[0]) / 2.0, axis=1)
    names = fit.model.exog_names

    if skip_intercept:
        means, err = means[1:], err[1:]
        names = names[1:]

    ax = means.plot(kind="bar", yerr=err, error_kw={ "ecolor": "black"}, color=kelly_colors)
    ax.set_xticklabels([get_fit_name(n) for n in names])

    return ax

def fit_coefficients_base(fit, ax=None, figsize=None, intercept=False,
        color_offset=0):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    names = [get_fit_name(n) for n in fit.model.exog_names]
    means = fit.conf_int().apply(lambda x: np.mean(x), axis=1)
    err = fit.conf_int().apply(lambda x: (x[1] - x[0]) / 2.0, axis=1)

    if not intercept:
        means, err = means[1:], err[1:]
        names = names[1:]

    ax = means.plot(kind="bar", yerr=err, error_kw={ "ecolor": "black"},
            color=kelly_colors[color_offset:])
    ax.set_title("Binomial Coefficients for Correct Grade by Program")
    ax.set_ylabel("Binomial Coefficients (95% CI)")
    ax.set_xlabel("Program Base")
    ax.set_xticklabels(names)

    return ax

def fit_coefficients_version(fit, ax=None, figsize=None, intercept=True):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    names = [get_fit_name(n) for n in fit.model.exog_names]
    means = fit.conf_int().apply(lambda x: np.mean(x), axis=1)
    err = fit.conf_int().apply(lambda x: (x[1] - x[0]) / 2.0, axis=1)

    # Assuming p-values come in the same order as confidence intervals
    for i, p_val in enumerate(fit.pvalues.values):
        if np.isclose(p_val, 1.0):
            err[i] = 0.0

    if not intercept:
        means, err = means[1:], err[1:]
        names = names[1:]

    # Assign colors by base
    colors = []
    last_base = None
    color_i = -1
    for n in names:
        base = n.split("_")[0]
        if base != last_base:
            color_i += 1
            last_base = base
        colors.append(kelly_colors[color_i])

    # Plot bars
    ax = means.plot(kind="bar", yerr=err, error_kw={ "ecolor": "black"}, color=colors)
    ax.set_title("Coefficients by Program/Version")
    ax.set_ylabel("Coefficients (95% CI)")
    ax.set_xlabel("Program Base/Version")
    ax.set_xticklabels(names)

    return ax

def version_balance(trials, ax=None, figsize=None):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    counts = trials.program_name.value_counts().sort_index()

    # Assign colors by base
    colors = []
    last_base = None
    color_i = -1
    for n in counts.index:
        base = n.split("_")[0]
        if base != last_base:
            color_i += 1
            last_base = base
        colors.append(kelly_colors[color_i])

    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Version Balance ({0} trials)".format(len(trials)))

    return ax
