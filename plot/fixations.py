import numpy as np, itertools as it
from kelly_colors import kelly_colors
from matplotlib import pyplot, cm, colors
from matplotlib.ticker import MultipleLocator, FuncFormatter
from PIL import Image, ImageDraw, ImageEnhance
from StringIO import StringIO
from ..util import contrast_color, transition_matrix, make_heatmap

# Raw fixations and heatmaps {{{

def circles(fixations, screen_image, radius_min=10, radius_max=35, fill="red",
        outline="black", alpha=0.8, saccade_lines=True, line_fill=None):
    """Draws fixation circles on to an image with optional saccade lines."""
    poly_image = Image.new("RGBA", screen_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(poly_image)
    min_duration, max_duration = fixations.duration_ms.min(), fixations.duration_ms.max()

    if line_fill is None:
        line_fill = fill

    # Draw fixation circles
    last_xy = None
    for _, fix in fixations.sort("start_ms").iterrows():
        x, y = fix["fix_x"], fix["fix_y"]

        # Draw saccade line
        if saccade_lines and last_xy:
            draw.line((last_xy[0], last_xy[1], x, y), fill=line_fill)

        # Calculate circle radius (transform from duration to radius)
        r = (radius_min + 
                ((radius_max - radius_min) *
                    ((fix["duration_ms"] - min_duration) / float(max_duration - min_duration))
                )
            )

        bbox = (x - r, y - r, x + r, y + r)
        draw.ellipse(bbox, fill=fill, outline=outline)
        last_xy = (x, y)

    # Flush drawing operations
    del draw

    # Extract alpha channel and increase from 0 to alpha parameter
    poly_alpha = poly_image.split()[3]
    poly_image.putalpha(ImageEnhance.Brightness(poly_alpha).enhance(alpha))

    # Blend circles on to screen image with alpha
    return Image.composite(poly_image, screen_image, poly_image)


#def spatial_density_heatmap(fixation_counts, screen_image, alpha=0.8, cmap=None, dpi=90):
    #width, height = screen_image.size
    #width_inches = width / float(dpi)
    #height_inches = height / float(dpi)

    #fig = pyplot.figure(figsize=(width_inches, height_inches), dpi=dpi, frameon=False)
    #ax = pyplot.Axes(fig, [0, 0, 1, 1])
    #ax.set_axis_off()
    #fig.add_axes(ax)

    #cmap.set_bad("b", 0)
    #cmap.set_under("b", 0)
    #cmap.set_over("b", 0)
    
    #masked_counts = np.ma.masked_where(fixation_counts < 1, fixation_counts)
    #ax.imshow(masked_counts.T, interpolation="none", cmap=cmap)
    #ax.invert_yaxis()

    ## Convert to image
    #plot_buffer = StringIO()
    #fig.savefig(plot_buffer, format="png", dpi=dpi)
    #plot_buffer.pos = 0
    #plot_image = Image.open(plot_buffer)

    ##plot_image.save("/home/hansenm/Desktop/test.png")
    #r, g, b, a = plot_image.split()
    #heatmap_image = Image.merge("RGB", (r, g, b))
    #heatmap_alpha = np.array(a) * alpha

    ##np.savetxt("/home/hansenm/Desktop/test.txt", heatmap_alpha)
    #mask = Image.fromarray(heatmap_alpha.astype(np.uint8), "L")

    #return Image.composite(heatmap_image, screen_image, mask)

#def fixation_heatmap(fixations, screen_image, alpha=0.8, dot_size=200, cmap=None, dpi=90):
    #width, height = screen_image.size
    #width_inches = width / float(dpi)
    #height_inches = height / float(dpi)

    #fig = pyplot.figure(figsize=(width_inches, height_inches), dpi=dpi, frameon=False)
    #ax = pyplot.Axes(fig, [0, 0, 1, 1])
    #ax.set_axis_off()
    #fig.add_axes(ax)

    #points = fixations[["fix_x", "fix_y"]].values
    #hm_data = make_heatmap(points, screen_image.size, dot_size)
    #masked_data = np.ma.masked_where(hm_data == 0, hm_data)
    #ax.imshow(masked_data, interpolation="none", cmap=cmap)

    ## Convert to image
    #plot_buffer = StringIO()
    #fig.savefig(plot_buffer, format="png", dpi=dpi)
    #plot_buffer.pos = 0
    #heatmap_image = Image.open(plot_buffer)

    #final_image = screen_image.copy()
    #final_image.paste(heatmap_image)

    #return final_image

# }}}

# Code plots {{{

def line_code_image(fixations, code_image, num_lines=None, image_padding=10, image_dpi=120,
        bar_height=0.75, bar_mult=1.0, width_inches=5, method="time"):
    """Plots fixation times or counts as bars next to code lines."""
    if num_lines is None:
        num_lines = fixations.line.max()

    if method == "time":
        method = lambda frame, lines: [sum(frame[frame.line == line].duration_ms) for line in lines]
    elif method == "count":
        method = lambda frame, lines: [sum(frame.line == line) for line in lines]

    lines = np.arange(1, num_lines + 1)
        
    # Plot bar graph
    height_inches = (code_image.size[1] - (image_padding * 2)) / float(image_dpi)
    fig = pyplot.figure(figsize=(width_inches, height_inches), dpi=image_dpi)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.barh(lines * bar_mult, method(fixations, lines), height=bar_height)
    
    # Show every line
    ax.set_yticks(lines)
    
    # Don't include line 0
    ax.set_ylim(1, num_lines + 1)
    
    # Lines start at 1 on top
    ax.invert_yaxis()
    
    # Bars go from right to left
    ax.invert_xaxis()
    
    # Combine with code image
    plot_buffer = StringIO()
    fig.savefig(plot_buffer, format="png", dpi=image_dpi)
    pyplot.close(fig)
    plot_buffer.pos = 0
    plot_image = Image.open(plot_buffer)
    
    master_image = Image.new("RGBA", (plot_image.size[0] + code_image.size[0], code_image.size[1]),
                             (255, 255, 255, 255))
    master_image.paste(plot_image, (0, image_padding))
    master_image.paste(code_image, (plot_image.size[0], 0))
    
    return master_image

# }}}

# Transition matrix {{{

#def line_transitions(fixations, trans_matrix=None, num_lines=None, show_probs=True, ax=None, cmap=None, figsize=None): 
    #if trans_matrix is None:
        #sorted_lines = fixations.sort("start_ms").line
        #trans_matrix = transition_matrix(sorted_lines, num_lines)
    
    #num_lines = trans_matrix.shape[0]
    #lines = np.arange(1, num_lines + 1)
    
    #if ax is None:
        #pyplot.figure(figsize=figsize)
        #ax = pyplot.axes()

    #if cmap is None:
        #cmap = cm.gist_gray_r

    #polys = ax.pcolor(trans_matrix, cmap=cmap, edgecolors="#000000", vmin=0, vmax=1)
    #ax.set_title("Line Transitions")
    #ax.set_xlim(0, num_lines)
    #ax.set_xticks(lines - 0.5)
    #ax.set_xticklabels(lines)
    #ax.set_ylim(0, num_lines)
    #ax.set_yticks(lines - 0.5)
    #ax.set_yticklabels(lines)
    #ax.set_ylabel("From Line")
    #ax.set_xlabel("To Line")
    #ax.invert_yaxis()
    
    #cb = ax.figure.colorbar(polys)
    #cb.set_label("Transition Probability")

    #if show_probs:
        #for row in range(num_lines):
            #for col in range(num_lines):
                #prob = trans_matrix[row, col]
                #if prob > 0:
                    #cell_rgba = cmap(prob)
                    #text_color = contrast_color(cell_rgba)
                    #ax.text(col + 0.5, row + 0.5, "{0:0.2f}".format(prob), ha="center", va="center", color=text_color)

    #return ax

# }}}

# Timeline plots {{{

def line_timeline_steps(fixations, cmap=None, line_numbers=True, ax=None, figsize=None):
    """Plots multiple timelines of line fixations, one step for each fixations."""
    # Group fixations by trial
    trial_groups = fixations.groupby("trial_id")
    num_trials = len(trial_groups)
    assert num_trials > 0, "No trials"

    # Maximum number of lines fixated
    num_steps = trial_groups.agg(len).max()[0]

    # Number of code lines (1-based) 
    num_lines = fixations.line.max()

    # Create figure, if necessary
    if ax is None:
        width = int(np.ceil(num_steps / 3.0))
        height = int(np.ceil(num_trials / 1.5))
        pyplot.figure(figsize=(width, height))
        ax = pyplot.axes()

    if cmap is None:
        line_colors = ["white"] + list(it.islice(it.cycle(kelly_colors), num_lines))
        cmap = colors.ListedColormap(line_colors)

    # Every code line is a different color
    norm = colors.BoundaryNorm(np.arange(0, num_lines + 1), num_lines)

    # Fill in matrix of lines visited
    visited = np.zeros(shape=(num_trials, num_steps), dtype=int)
    row = 0
    for _, trial in trial_groups:
        col = 0
        for line in trial.sort("start_ms").line.values:
            visited[row, col] = line
            col += 1
        row += 1

    # Create grid and set up axes
    ax.pcolor(visited, cmap=cmap, norm=norm, edgecolors="black")
    ax.set_xlim(0, num_steps)
    ax.set_xlabel("Line Step")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylim(0, num_trials)
    ax.set_ylabel("Participant")
    ax.set_title("Lines Fixated by Participant and Step")

    # Align participant labels in the center
    parts = np.arange(0, num_trials)
    ax.set_yticks(parts + 0.5)
    ax.set_yticklabels(parts + 1)

    # Add line numbers
    if line_numbers:
        for row in range(visited.shape[0]):    
            for col, line in enumerate(visited[row, :]):
                if line < 1:
                    break
                cell_rgba = cmap(line / float(num_lines + 1))
                text_color = contrast_color(cell_rgba)
                ax.text(col + 0.5, row + 0.5, str(line), ha="center", va="center", color=text_color)

    return ax

def line_timeline(fixations, cmap=None, line_numbers=True, ax=None,
        figsize=None, num_steps=None, step_size=None):
    """Plots multiple timelines of line fixations, one step for each unit of time."""
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

    # Number of code lines (1-based) 
    num_lines = fixations.line.max()

    # Create figure, if necessary
    if ax is None:
        width = int(np.ceil(num_steps / 3.0))
        height = int(np.ceil(num_trials / 1.5))
        pyplot.figure(figsize=(width, height))
        ax = pyplot.axes()

    if cmap is None:
        line_colors = ["white"] + list(it.islice(it.cycle(kelly_colors), num_lines))
        cmap = colors.ListedColormap(line_colors)

    # Every code line is a different color
    norm = colors.BoundaryNorm(np.arange(0, num_lines + 1), num_lines)

    # Fill in matrix of lines visited
    all_times = np.arange(0, last_time + step_size, step=step_size)
    visited = np.zeros(shape=(num_trials, num_steps), dtype=int)
    row = 0
    for _, trial in trial_groups:
        col = 0
        last_time = 0
        for time in all_times[1:]:
            lines = trial[(last_time <= trial.start_ms) & (trial.end_ms < time)].line
            if len(lines) > 0:
                visited[row, col] = lines.value_counts().index[0]
            last_time = time
            col += 1
        row += 1

    # Create grid and set up axes
    ax.pcolor(visited, cmap=cmap, norm=norm, edgecolors="black")
    ax.set_xlim(0, num_steps)
    ax.set_xlabel("Time (step={0} ms)".format(step_size))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylim(0, num_trials)
    ax.set_ylabel("Participant")
    ax.set_title("Lines Fixated by Participant and Time")

    # Align participant labels in the center
    parts = np.arange(0, num_trials)
    ax.set_yticks(parts + 0.5)
    ax.set_yticklabels(parts + 1)

    # Add line numbers
    if line_numbers:
        for row in range(visited.shape[0]):    
            for col, line in enumerate(visited[row, :]):
                if line > 0:
                    cell_rgba = cmap(line / float(num_lines + 1))
                    text_color = contrast_color(cell_rgba)
                    ax.text(col + 0.5, row + 0.5, str(line), ha="center", va="center", color=text_color)

    return ax

def line_timeline_single(line_fixations, output_fixations=None,
        num_lines=None, ax=None, figsize=None):
    """Plots a single timeline of line fixations by seconds."""
    if num_lines is None:
        num_lines = line_fixations.line.max()
    
    # Gather list of times and line numbers (output box is line 0)
    times_lines = dict(list(line_fixations[["start_ms", "line"]].values) +
                       list(line_fixations[["end_ms", "line"]].values))

    if output_fixations is not None:
        times_lines.update(dict([(t, 0) for t in output_fixations.start_ms] +
                                [(t, 0) for t in output_fixations.end_ms]))
    
    sorted_times = sorted(times_lines.keys())
    sorted_lines = [times_lines[k] for k in sorted_times]
    max_time = max(sorted_times)
    
    # Plot fixation lines and points
    if ax is None:
        if figsize is None:
            fig_width = int(np.ceil(max_time / 1000.0 / 2))
            fig_height = int(np.ceil((num_lines + 1) / 2.0))
            figsize = (fig_width, fig_height)
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    ax.plot(sorted_times, sorted_lines, linewidth=2)
    ax.scatter(sorted_times, sorted_lines, marker="o", alpha=0.5, color="red", s=50)
    ax.grid()
    ax.set_title("Fixations By Line")
    
    # Line 1 is at the top (output box is above it)
    if output_fixations is not None:
        lines = np.arange(0, num_lines + 1)
        ax.set_yticks(lines)
        ax.set_ylim(-0.5, num_lines + 0.5)
        ax.set_yticklabels(["Output\nTextbox"] + [str(l) for l in lines[1:]])

        # Separate output box from other lines
        ax.axhline(0.5, color="black", linestyle="--", linewidth=2)
    else:
        lines = np.arange(1, num_lines + 1)
        ax.set_ylim(0.5, num_lines + 0.5)
        ax.set_yticks(lines)
        ax.set_yticklabels(lines)

    ax.set_ylabel("Line")
    ax.invert_yaxis()
    
    # Show time in seconds instead of millis
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x / 1000)))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.set_xlim(-500, max_time + 1000)
    ax.set_xlabel("Time (seconds)")
    
    return ax

# }}}
