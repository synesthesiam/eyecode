import operator, itertools as it

import numpy as np
from matplotlib import pyplot, cm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from PIL import Image, ImageDraw, ImageEnhance
from StringIO import StringIO

from kelly_colors import kelly_colors
from ..aoi import get_aoi_kinds, kind_to_col
from ..util import contrast_color

# AOI plots {{{

def draw_rectangles(aoi_rectangles, screen_image, colors=None,
        outline="black", alpha=0.5):
    """Draws AOI rectangles on to an image.
    
    Parameters
    ----------
    aoi_rectangles : pandas DataFrame
        DataFrame with a row for each rectangle (x, y, width, height columns)

    screen_image : PIL Image
        Image on top of which AOI rectangles will be drawn

    colors : list or None, optional
        List of PIL fill colors for each AOI kind. Colors will by cycled through
        if there are more AOI kinds than colors. If None, kelly_colors are used.
        Default is None (kelly_colors).

    outline : str, optional
        Rectangle outline color (default is 'black')

    alpha : float, optional
        Transparency of AOI rectangles (0-1, 1 = opaque).
        Default is 0.5 (50% transparency).

    Returns
    -------
    img : PIL Image
        Screen image with AOI rectangles drawn on top
    
    """
    rect_image = Image.new("RGBA", screen_image.size)
    draw = ImageDraw.Draw(rect_image)

    if colors is None:
        colors = kelly_colors
    colors = it.cycle(colors)

    for kind, kind_rows in aoi_rectangles.groupby("kind"):
        for x, y, w, h in kind_rows[["x", "y", "width", "height"]].values:
            draw.rectangle([(x, y), (x + w - 1, y + h - 1)],
                    fill=colors.next(), outline=outline)

    del draw

    # Extract alpha channel and increase from 0 to alpha parameter
    rect_alpha = rect_image.split()[3]
    rect_image.putalpha(ImageEnhance.Brightness(rect_alpha).enhance(alpha))

    # Blend on to screen image with alpha
    return Image.composite(rect_image, screen_image, rect_image)

def aoi_transitions(trans_matrix, name_map=None,
        show_probs=True, ax=None, cmap=None, figsize=None): 
    """Plots an AOI transition matrix with a colorbar.

    See also
    --------
    aoi.transition_matrix

    aoi.scanpath_from_fixations
    
    """
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    if cmap is None:
        cmap = cm.gist_gray_r

    rows = np.arange(trans_matrix.shape[0])
    cols = np.arange(trans_matrix.shape[1])

    polys = ax.pcolor(trans_matrix, cmap=cmap, edgecolors="#000000", vmin=0, vmax=1)
    ax.set_title("AOI Transitions")

    # x-axis
    ax.set_xlim(0, len(cols))
    ax.set_xticks(cols + 0.5)

    if name_map is None:
        ax.set_xticklabels(cols)
    else:
        ax.set_xticklabels([name_map[c] for c in cols])
        pyplot.setp(ax.get_xticklabels(), rotation=90)

    ax.set_xlabel("To AOI")

    # y-axis
    ax.set_ylim(0, len(rows))
    ax.set_yticks(rows + 0.5)

    if name_map is None:
        ax.set_yticklabels(rows)
    else:
        ax.set_yticklabels([name_map[r] for r in rows])

    ax.set_ylabel("From AOI")
    ax.invert_yaxis()
    
    # Probability colorbar
    cb = ax.figure.colorbar(polys)
    cb.set_label("Transition Probability")

    # Probability text labels
    if show_probs:
        for row in rows:
            for col in cols:
                prob = trans_matrix[row, col]
                if prob > 0:
                    cell_rgba = cmap(prob)
                    text_color = contrast_color(cell_rgba)
                    ax.text(col + 0.5, row + 0.5, "{0:0.2f}".format(prob),
                            ha="center", va="center", color=text_color)

    return ax

def aoi_barplot(fixations, method="time", ylabel=None,
        scalar=1e-3, ax=None, figsize=None):
    """Plots fixation time or counts for all AOI kinds and names.

    Parameters
    ----------
    fixations : pandas DataFrame
        DataFrame with one row per fixation and AOI hits. Must have duration_ms
        and start_ms columns.
    
    method : str or callable, optional
        Method for determining height of bars. May be one of:
            * 'time' - total fixation duration
            * 'count' - number of fixations
            * 'first' - time of first fixation
        or a callable with 2 argments (frame, column) where
            * frame is the fixations DataFrame
            * column is the name of the AOI column
        that returns the height of the bar for the given AOI.
        Default is 'time'.

    ylabel : str or None, optional
        Label used for the y-axis. If None, a label is chosen depending on the
        method (time, count, or first).

    scalar : float, optional
        Number used to scale the height of bars. This is 0.001 by default,
        meaning time units will be in seconds rather than milliseconds.

    ax : matplotlib Axes or None, optional
        An Axes to plot onto or None to create a new one (default: None)

    figsize : tuple or None, optional
        Size of figure in inches when creating new Axes (width, height).
        If None, default figsize is used (default: None).

    Returns
    -------
    ax : matplotlib Axes

    """

    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    aoi_kinds = get_aoi_kinds(fixations)
    colors = it.cycle(kelly_colors)

    if method == "time":
        method = lambda f, c: f[[c, "duration_ms"]]\
                .groupby(c)["duration_ms"].sum()
        ylabel = ylabel or "Duration (sec)"
    elif method == "first":
        method = lambda f, c: f[[c, "start_ms"]]\
                .groupby(c)["start_ms"].sum()
        ylabel = ylabel or "First Fixation (sec)"
    elif method == "count":
        method = lambda f, c: f[[c]]\
                .groupby(c).size()
        ylabel = ylabel or "Fixation Count"
        scalar = 1.0

    bottom_start = 0
    xticks = []
    xlabels = []
    width = 1.0
    for kind, color in zip(aoi_kinds, colors):
        col = kind_to_col(kind)
        times = method(fixations, col) * scalar
        ind = np.arange(len(times)) + bottom_start
        data = np.array(times)
        ax.bar(ind, data, width=width, color=color, label=kind)
        
        for i, name in enumerate(times.index):
            xticks.append(bottom_start + (width / 1.75) + i)
            xlabels.append(name)
        
        bottom_start += len(times) + 2

    ax.set_title("Fixations by AOI")
    ax.set_xlim(-1, len(xticks) + (len(aoi_kinds) * 2) - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90)

    ax.grid()

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.legend(bbox_to_anchor=(1.1, 1.05))

    return ax

# }}}

# Fixation plots {{{

def fix_circles(fixations, screen_image, radius_min=10, radius_max=35, fill="red",
        outline="black", alpha=0.8, saccade_lines=True, line_fill=None):
    """Draws fixation circles on an image with optional saccade lines.

    Circle radii are scaled by fixation duration, with the longest fixation mapping
    to radius_max and the shortest mapping to radius_min.
    
    Parameters
    ----------
    fixations : pandas DataFrame
        DataFrame with fixations and fix_x, fix_y, start_ms, duration_ms columns

    screen_image : PIL Image
        Image on top of which fixation circles will be drawn

    radius_min : int, optional
        Minimum radius of fixation circles (pixels)

    radius_max : int, optional
        Maximum radius of fixation circles (pixels)

    fill : str, optional
        PIL fill color of the fixation circles (default: 'red')

    outline : str, optional
        PIL outline color for fixation circles (default: 'black')

    alpha : float, optional
        Transparency of fixation circles (0-1, 1 = opaque).
        Default is 0.8 (20% transparent).

    saccade_lines : bool, optional
        If True, draw saccade lines between each fixation circle.
        Default is True.

    line_fill : str or None, optional
        Color of saccade lines or None to use fill color.
        Default is None.

    Returns
    -------
    img : PIL Image
        Screen image with fixation circles and saccade lines drawn on top.
    
    """
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

def highlight_code(code, lex=None, fmt=None, style="default",
        font_name="Droid Sans Mono", font_size=19,
        image_pad=10, line_pad=10, line_numbers=False):
    """Highlights Python code using Pygments.
    
    Parameters
    ----------
    code : str
        Code string to highlight

    lex : Pygments lexer or None, optional
        Lexer to use or None for python lexer (default: None)

    fmt : Pygments formatter or None, optional
        Formatter to use or None for png formatter with options
        specified by arguments below (default: None)

    style : str
        Highlighting style (default: 'default')

    font_name : str
        Name of font to use (default: 'Droid Sans Mono')

    font_size : int
        Size of font to use in points (default: 19)

    image_pad : int
        Padding around image in pixels (default: 10)

    line_pad : int
        Extra padding for new lines in pixels (default: 10)

    line_numbers : bool
        True if Pygments should add line numbers (default: False)

    Returns
    -------
    img : PIL Image
        Image with highlighted code
    
    """

    import pygments
    from pygments import formatters, lexers

    # Create lexer and formatter if not provided
    lex = lex or lexers.get_lexer_by_name("python")
    fmt = fmt or formatters.get_formatter_by_name("png",
                                           style=style,
                                           font_name=font_name,
                                           font_size=font_size,
                                           line_numbers=line_numbers,
                                           image_pad=image_pad,
                                           line_pad=line_pad)

    # Highlight code
    png_data = pygments.highlight(code, lex, fmt)

    # Convert to PIL Image
    code_image = Image.open(StringIO(png_data))
    return code_image

def line_code_image(line_fixes, code_image, num_lines, method="time",
        image_padding=10, image_dpi=120, bar_height=0.75, bar_mult=1.0,
        width_inches=5, color=None, **kwargs):
    """Plots fixation information as bars next to code lines.
    
    Parameters
    ----------
    line_fixes : pandas DataFrame
        DataFrame with one fixation per row + line annotation. Must have
        columns start_ms, duration_ms, line.

    code_image : PIL Image
        Image with highlighted code (see plot.hightlight_code)

    num_lines : int
        Number of lines in the code

    method : str or callable, optional
        Method for determining size of line bars. May be one of:
            * 'time' - total fixation duration
            * 'count' - number of fixations
            * 'first' - time of first fixation
        or a callable with 2 argments (frame, lines) where
            * frame is the line_fixes DataFrame
            * lines is a list of line numbers
        that returns a list of bar heights for each line.
        Default is 'time'.

    image_padding : int, optional
        Padding expected around image in pixels (default: 10)        

    image_dpi : int, optional
        Dots per inch for final rendered image

    bar_height : float, optional
        Height (or thickness) of horizontal bars (default: 0.75)

    bar_mult : float, optional
        Factor to multiply bars' vertical positions by (default: 1.0)

    width_inches : float, optional
        Width of final rendered image in inches

    color : str or None, optional
        Color of bars or None for automatic selection

    **kwargs : keyword arguments
        Arguments passed through to matplotlib barh function

    Returns
    -------
    img : PIL Image
        Image with bars and code combined
    
    """

    assert num_lines > 0, "Must have more than 0 lines"

    # Check for known methods (time, count, first)
    if method == "time":
        # Total fixation duration per line
        method = lambda frame, lines: [sum(frame[frame.line == line].duration_ms) for line in lines]
        color = color or kelly_colors[0]
    elif method == "count":
        # Number of fixations per line
        method = lambda frame, lines: [sum(frame.line == line) for line in lines]
        color = color or kelly_colors[1]
    elif method == "first":
        # Time of first fixation per line
        method = lambda frame, lines: [frame[frame.line == line].start_ms.min() for line in lines]
        color = color or kelly_colors[2]

    lines = np.arange(1, num_lines + 1)
        
    # Plot bar graph with no axes or labels
    height_inches = (code_image.size[1] - (image_padding * 2)) / float(image_dpi)
    fig = pyplot.figure(figsize=(width_inches, height_inches), dpi=image_dpi)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.barh(lines * bar_mult, method(line_fixes, lines), height=bar_height,
            color=color, **kwargs)
    
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
    
    # Create combined image
    master_image = Image.new("RGBA", (plot_image.size[0] + code_image.size[0], code_image.size[1]),
                             (255, 255, 255, 255))

    # Paste bar plot (left) and code (right)
    master_image.paste(plot_image, (0, image_padding))
    master_image.paste(code_image, (plot_image.size[0], 0))
    
    return master_image


def fix_timeline(line_fixes, num_lines,
        output_fixes=None, ax=None, figsize=None):
    """Plots a timeline of line fixations by seconds.

    Parameters
    ----------
    line_fixes : pandas DataFrame
        DataFrame with one fixation per row + line annotation. Must have
        columns start_ms, end_ms, line.

    num_lines : int
        Number of lines in the program

    output_fixes : pandas DataFrame or None, optional
        DataFrame with fixations on the output box (interface kind).
        If None, only line fixations are displayed (default: None).

    ax : matplotlib Axes or None, optional
        An Axes to plot onto or None to create a new one (default: None)

    figsize : tuple or None, optional
        Size of figure in inches when creating new Axes (width, height).
        If None, default figsize is used (default: None).

    Returns
    -------
    ax : matplotlib Axes
        Axes with timeline plotted on top
    
    """
    assert num_lines > 0, "Must have more than 0 lines"
    
    # Gather list of times and line numbers (output box is line 0)
    times_lines = dict(list(line_fixes[["start_ms", "line"]].values) +
                       list(line_fixes[["end_ms", "line"]].values))

    if output_fixes is not None:
        times_lines.update(dict([(t, 0) for t in output_fixes.start_ms] +
                                [(t, 0) for t in output_fixes.end_ms]))
    
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
    if output_fixes is not None:
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

