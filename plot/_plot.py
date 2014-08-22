import operator, itertools as it

import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot, cm
from matplotlib.ticker import MultipleLocator, FuncFormatter, FixedFormatter
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageEnhance
from StringIO import StringIO

from kelly_colors import kelly_colors
from ..aoi import get_aoi_kinds, kind_to_col, envelope, make_grid, hit_test, hit_point, scanpath_from_fixations
from ..util import contrast_color, significant, make_heatmap
from ..stats import permute_correlation_matrix
from ..metrics import time_between_fixes

# AOI plots {{{

def draw_rectangles(aoi_rectangles, screen_image, colors=None,
        outline="black", alpha=0.5, color_func=None):
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

    if color_func is None:
        color_func = lambda k, n, li: colors.next()

    if not callable(outline):
        outline_color = outline
        outline = lambda k, n, li: outline_color

    row_cols = ["x", "y", "width", "height", "name", "local_id"]
    for kind, kind_rows in aoi_rectangles.groupby("kind"):
        for x, y, w, h, name, local_id in kind_rows[row_cols].values:
            draw.rectangle([(x, y), (x + w - 1, y + h - 1)],
                    fill=color_func(kind, name, local_id),
                    outline=outline(kind, name, local_id))

    del draw

    # Extract alpha channel and increase from 0 to alpha parameter
    rect_alpha = rect_image.split()[3]
    rect_image.putalpha(ImageEnhance.Brightness(rect_alpha).enhance(alpha))

    # Blend on to screen image with alpha
    return Image.composite(rect_image, screen_image, rect_image)

def aoi_transitions(trans_matrix, name_map=None,
        show_probs=True, ax=None, cmap=None,
        figsize=None, show_colorbar=True,
        prob_threshold=0.0, min_size=(5, 4)): 
    """Plots an AOI transition matrix with a colorbar.

    See also
    --------
    aoi.transition_matrix

    aoi.scanpath_from_fixations
    
    """
    rows = np.arange(trans_matrix.shape[0])
    cols = np.arange(trans_matrix.shape[1])

    if ax is None:
        if figsize is None:
            w, h = trans_matrix.shape[:2]
            figsize = (max(min_size[0], w * 0.75),
                       max(min_size[1], h * 0.5))
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    if cmap is None:
        cmap = cm.gist_gray_r

    polys = ax.pcolor(trans_matrix, cmap=cmap,
            edgecolors="#000000", vmin=0, vmax=1)
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
    if show_colorbar:
        cb = ax.figure.colorbar(polys)
        cb.set_label("Transition Probability")

    # Probability text labels
    if show_probs:
        for row in rows:
            for col in cols:
                prob = trans_matrix[row, col]
                if prob > prob_threshold:
                    cell_rgba = cmap(prob)
                    text_color = contrast_color(cell_rgba)
                    ax.text(col + 0.5, row + 0.5, "{0:0.2f}".format(prob),
                            ha="center", va="center", color=text_color)

    return ax

def aoi_barplot(fixations, method="time", ylabel=None,
        scalar=1e-3, aoi_kinds=None, ax=None, figsize=None):
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

    if aoi_kinds is None:
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

def highlight_code(code, lexer=None, formatter=None, filename=None,
        style="default", font_name=None, font_size=19,
        image_pad=10, line_pad=8, line_numbers=False):
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
    if lexer is None:
        msg = "filename is required if no lexer is provided"
        assert filename is not None, msg
        lexer = pygments.lexers.guess_lexer_for_filename(filename, code)

    # Create lexer and formatter if not provided
    formatter = formatter or formatters.get_formatter_by_name("png",
            style=style, font_name=font_name,
            font_size=font_size, line_numbers=line_numbers,
            image_pad=image_pad, line_pad=line_pad)

    # Highlight code
    if not isinstance(code, str):
        # Convert to a single string
        code = "\n".join([line.rstrip() for line in code])

    png_data = pygments.highlight(code, lexer, formatter)

    # Convert to PIL Image
    code_image = Image.open(StringIO(png_data))
    return code_image

def line_code_image(line_fixes, code_image, num_lines, method="time",
        image_padding=10, image_dpi=120, bar_height=0.75, bar_mult=1.0,
        width_inches=5, color=None, horiz_sep=0,
        line_numbers=False, **kwargs):
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

    horiz_sep : int, optional
        Separation between bars and code image in pixels (default: 0)

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

    if line_numbers:
        # Leave a little room for the line numbers
        fig, ax = pyplot.subplots(figsize=(width_inches, height_inches), dpi=image_dpi)
        ax.set_position([0, 0, 0.9, 1])
        ax.set_frame_on(False)
    else:
        # Take the entire figure space
        fig = pyplot.figure(figsize=(width_inches, height_inches), dpi=image_dpi)
        ax = pyplot.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)

    ax.barh(lines * bar_mult, method(line_fixes, lines), height=bar_height,
            color=color, **kwargs)

    # Show every line
    ax.set_yticks(0.5 + lines)
    
    # Don't include line 0
    ax.set_ylim(1, num_lines + 1)
    
    # Lines start at 1 on top
    ax.invert_yaxis()
    
    # Bars go from right to left
    ax.invert_xaxis()

    if line_numbers:
        # Put line numbers on the right
        ax.set_yticklabels(np.arange(1, num_lines + 1))
        ax.yaxis.tick_right()

        # Align line number labels
        for label in ax.yaxis.get_ticklabels():
            label.set_verticalalignment("center")
        
        # Hide tick lines
        for tic in it.chain(ax.xaxis.get_major_ticks(),
                            ax.yaxis.get_major_ticks()):
            tic.tick1On = False
            tic.tick2On = False

    # Combine with code image
    plot_buffer = StringIO()
    fig.savefig(plot_buffer, format="png", dpi=image_dpi)
    pyplot.close(fig)
    plot_buffer.pos = 0
    plot_image = Image.open(plot_buffer)
    
    # Create combined image
    master_width = plot_image.size[0] + horiz_sep + code_image.size[0]
    master_image = Image.new("RGBA", (master_width, code_image.size[1]),
                             (255, 255, 255, 255))

    # Paste bar plot (left) and code (right)
    master_image.paste(plot_image, (0, image_padding))
    master_image.paste(code_image, (plot_image.size[0] + horiz_sep, 0))

    return master_image


def fix_timeline(line_fixes, num_lines, output_fixes=None,
        ax=None, figsize=None, barebones=False):
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
    if not barebones:
        ax.scatter(sorted_times, sorted_lines, marker="o", alpha=0.5, color="red", s=50)
        ax.grid()
        ax.set_title("Fixations By Line")
    
    # Line 1 is at the top (output box is above it)
    if output_fixes is not None:
        lines = np.arange(0, num_lines + 1)
        ax.set_yticks(lines)
        ax.set_ylim(-0.5, num_lines + 0.5)
        if not barebones:
            ax.set_yticklabels(["Output\nTextbox"] + [str(l) for l in lines[1:]])

        # Separate output box from other lines
        ax.axhline(0.5, color="black", linestyle="--", linewidth=2)
    else:
        lines = np.arange(1, num_lines + 1)
        ax.set_ylim(0.5, num_lines + 0.5)
        ax.set_yticks(lines)
        if not barebones:
            ax.set_yticklabels(lines)

    if not barebones:
        ax.set_ylabel("Line")
    ax.invert_yaxis()
    
    # Show time in seconds instead of millis
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x / 1000)))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.set_xlim(-500, max_time + 1000)
    if not barebones:
        ax.set_xlabel("Time (seconds)")
    else:
        ax.set_frame_on(False)
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
    
    return ax

# }}}

# Metric plots {{{

def rolling_metrics(results, columns=None, names=None, colors=None,
        markersize=5, ax=None, figsize=None):
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter

    if columns is None:
        columns = [results.columns[0]]

    fig = None
    if ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = pyplot.axes()
    else:
        fig = ax.figure
   
    axes = [ax] + [ax.twinx() for c in columns[1:]]

    if colors is None:
        colors = ["r", "g", "b", "y", "purple", "orange", "black"]
    colors = it.cycle(colors)

    if names is None:
        names = columns
    elif isinstance(names, str):
        names = [names]
    
    if len(columns) > 2:
        fig.subplots_adjust(left=0, right=0.75)
        axes[2].spines['right'].set_position(('axes', 1.1))
        axes[2].set_frame_on(True)
        axes[2].patch.set_visible(False)
    
    # Plot left y-axis
    color = next(colors)
    axes[0].plot(results.index, results[columns[0]], color=color, marker="o",
            markersize=markersize, label=names[0])
    axes[0].set_ylabel(names[0], color=color)
    axes[0].tick_params(axis="y", colors=color)
    
    if len(columns) > 1:
        # Plot right y-axis
        color = next(colors)
        axes[1].plot(results.index, results[columns[1]], color=color, marker="*",
                markersize=markersize, label=names[1])
        axes[1].set_ylabel(names[1], color=color)
        axes[1].tick_params(axis="y", colors=color)
    
    if len(columns) > 2:
        color = next(colors)
        axes[2].plot(results.index, results[columns[2]], color=color, marker="^",
                markersize=markersize, label=names[2])
        axes[2].set_ylabel(names[2], color=color)
        axes[2].tick_params(axis="y", colors=color)
    
    # Adjust x-axis to seconds
    ax.set_xlabel("Time (sec)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(int(x) / 1000)))
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    
    return ax

def find_font(name="monospace"):
    from matplotlib import font_manager
    return font_manager.findfont(name)

def text_size(font_path, font_size, text="|"):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(Image.new("1", (1, 1)))
    size = draw.textsize(text, font=font)
    del draw
    return size

def draw_code(code, font_path=None, font_size=18,
        font=None, image=None, line_offset=5, offset=(0, 0),
        padding=(0, 0), fill="black", bg_color="white"):
    """Renders code on to a background image with a given font and spacing.
    
    Parameters
    ----------
    code : str or list
        Code to render as a string or a list of lines. If a str, the code will
        by split by '\\n'.

    font_path : str, optional
        Path to a truetype font for rendering the code. If None and font is not
        provided, the system's default monospace font will be used (default:
        None).

    font_size : int, optional
        Size of font in pixels (default: 18).

    font : PIL ImageFont, optional
        Font to use for rendering. If provided, font_path is ignored (default:
        None).

    image : PIL.Image, optional
        Background image behind rendered code. If None, a new image is created
        to contain the rendered lines and filled with bg_color (default: None).

    line_offset : int, optional
        Number of pixels between lines of code (default: 5).

    offset : tuple of int, optional
        Horizontal and vertical offset from top-left corner for rendered code
        in pixels (default: (0, 0)).

    padding : tuple of int, optional
        Symmentric horizontal and vertical padding around rendered code if
        background image is not provided (default: (0, 0)).

    fill : str, optional
        Fill color for rendered code (default: 'black').

    bg_color : str, optional  
        Fill color for background if background image is not provided (default:
        'white').
    
    Returns
    -------
    code_image : PIL.Image
        Image with code rendered on top of background.

    Notes
    -----
    Requires the PIL image library: http://www.pythonware.com/products/pil/
    
    """
    from PIL import Image, ImageDraw, ImageFont

    if font is None:
        if font_path is None:
            # Look up system monospace font
            font_path = find_font()
        font = ImageFont.truetype(font_path, font_size)

    # If code is a string, split into lines
    if isinstance(code, str):
        lines = code.split("\n")
    else:
        lines = code

    # Create background image to fit rendered code
    if image is None:
        temp_image = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(temp_image)
        max_x, max_y = offset
        for line in lines:
            line_size = draw.textsize(line.rstrip(), font=font)
            max_x = max(max_x, offset[0] + line_size[0])
            max_y += font_size + line_offset

        max_x = int(np.ceil(max_x)) + padding[0]
        max_y = int(np.ceil(max_y)) + padding[1]
        image = Image.new("RGB", (max_x + 1, max_y + 1), bg_color)
        del draw, temp_image

    draw = ImageDraw.Draw(image)

    # Draw code text onto background image
    y = offset[1]
    for i, line in enumerate(lines):
        x = offset[0]
        draw.text((x, y), line.rstrip(), font=font, fill=fill)
        y += font_size + line_offset

    del draw
    return image

# }}}


def correlation_matrix(frame, cols, ax=None, figsize=None,
        label_size="small", alpha=0.05, label_threshold=0.2,
        add_legend=True, method="spearman"):
    if ax is None:
        fig, ax = pyplot.subplots(1, 1, figsize=figsize)

    title = ""
    if isinstance(method, str):
        title = method.capitalize() + " "

    ax.set_title("{0}Correlation Matrix ({1} columns)".format(title, len(cols)))

    corr2d, sig2d = permute_correlation_matrix(frame[cols], method=method)
    text2d = np.empty(shape=(len(cols), len(cols)), dtype=object)
    num_steps = 100

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i != j:
                r = corr2d[i, j]
                corr2d[i, j] = num_steps + (num_steps * r)
                if sig2d[i, j]:
                    text2d[i, j] = "{0:.0f}".format(abs(r) * 100)
                    if r < 0:
                        text2d[i, j] = "({0})".format(text2d[i, j])
                else:
                    text2d[i, j] = ""
            else:
                corr2d[i, j] = np.nan

    cdict = { "blue" : [(0.0, 0.0, 0.0),
                        (0.5, 1.0, 1.0),
                        (1.0, 0.0, 0.0)],

              "red"    : [(0.0, 1.0, 1.0),
                          (0.5, 1.0, 1.0),
                          (1.0, 0.0, 0.0)],
              
              "green" : [(0.0, 0.0, 0.0),
                         (0.5, 1.0, 1.0),
                         (1.0, 1.0, 1.0)]}

    cmap = LinearSegmentedColormap("heat", cdict, N=(num_steps * 2))
    masked = np.ma.masked_where(np.isnan(corr2d), corr2d)
    ax.set_axis_bgcolor("#CCCCCC")
    meshes = ax.pcolor(masked, cmap=cmap, edgecolors="#000000", vmin=0, vmax=(num_steps * 2))
    
    for i in range(len(cols)):
        for j in range(len(cols)):
            if (i != j) and text2d[i, j] is not None:
                pyplot.text(i + 0.5, j + 0.5, text2d[i, j],
                         horizontalalignment="center",
                         verticalalignment="center",
                         size=label_size)

    fig = ax.figure
    if add_legend:
        cb = fig.colorbar(meshes, ticks=[0, num_steps, num_steps * 2],
                          format=pyplot.FixedFormatter(["-1", "0", "1"]))

        cb.set_label("{0}Correlation (x100)".format(title))

    # Set exact limits
    ax.set_xlim((0, len(cols)))
    ax.set_ylim((0, len(cols)))

    # Label columns
    loc = pyplot.FixedLocator([0.5 + x for x in range(len(cols))])
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    tic = pyplot.FixedFormatter(cols)
    ax.xaxis.set_major_formatter(tic)
    ax.yaxis.set_major_formatter(tic)

    pyplot.xticks(rotation=90)
    fig.tight_layout()

    return ax

def super_code_image(fixes, line_fixes, num_lines, screen_img, trial,
        trial_aois, aoi_kind="code-grid", cmap=pyplot.cm.OrRd,
        aoi_alpha=0.7, code_padding=5, line_numbers=False,
        time_between=False, between_cmap=pyplot.cm.binary):

    # Crop out code image
    line_aois = trial_aois[(trial_aois.kind == "line")]
    env = envelope(line_aois, code_padding).irow(0)
    crop_rect = [env["x"], env["y"], env["x"] + env["width"], env["y"] + env["height"]]

    # Hit test against grid AOIs
    grid_aois = trial_aois[(trial_aois.kind == aoi_kind)]
    grid_col = kind_to_col(aoi_kind)
    grid_fixes = fixes.dropna(subset=[grid_col])
    grid_counts = grid_fixes.groupby(grid_col).duration_ms.sum()
    max_grid_count = float(max(grid_counts))

    def color_grid(kind, name, local_id):
        rel_count = grid_counts.get(name, default=0) / max_grid_count
        return matplotlib.colors.rgb2hex(cmap(rel_count))

    outline_func = None
    if time_between:
        sp = scanpath_from_fixations(grid_fixes, aoi_kind)
        fix_between = time_between_fixes(sp)
        between_means = np.log(fix_between.groupby("name").time_ms.mean())
        min_time_between = float(min(between_means))
        max_time_between = float(max(between_means))

        def color_outline(kind, name, local_id):
            rel_time = (between_means.get(name, default=0) - min_time_between) / \
                (max_time_between - min_time_between)
            return matplotlib.colors.rgb2hex(between_cmap(1 - rel_time))

        outline_func = color_outline

    # Create syntax-based image
    code_box = trial_aois[(trial_aois.kind == "interface") &
                          (trial_aois.name == "code box")].irow(0)
    aoi_img = draw_rectangles(grid_aois, screen_img, color_func=color_grid,
            alpha=aoi_alpha, outline=outline_func)
    code_img = aoi_img.crop(crop_rect)

    # Compute line colors based on their associated block fixation times
    block_fixes = fixes.dropna(subset=[kind_to_col("block")])
    block_counts = block_fixes.groupby("hit_id_block").duration_ms.sum()
    block_aois = trial_aois[(trial_aois.kind == "block")]
    max_block_count = float(max(block_counts))

    colors = ["w"] * num_lines
    for idx, row in line_aois.iterrows():
        # Compute associated block for line
        line_num = int(row["name"].split(" ")[1])
        line_block = block_aois[(row["y"] >= block_aois["y"]) &
                (row["y"] < (block_aois["y"] + block_aois["height"]))]

        if len(line_block) > 0:
            # Extract fixation counts for this block and compute color
            block_id = line_block.irow(0)["local_id"]
            rel_count = block_counts.get(block_id, default=0) / max_block_count
            colors[line_num - 1] = matplotlib.colors.rgb2hex(cmap(rel_count))
        
    # Create final image combining lines, blocks, and syntax fixation counts
    return line_code_image(line_fixes, code_img, num_lines, color=colors,
            image_padding=3, bar_height=0.85, bar_mult=1.001, horiz_sep=5,
            method="time", line_numbers=line_numbers)

def aoi_code_image(fixes, screen_img,
        trial_aois, kind="code-grid", cmap=pyplot.cm.OrRd,
        syntax_alpha=0.7, code_padding=5):

    # Crop out code image
    line_aois = trial_aois[(trial_aois.kind == "line")]
    env = envelope(line_aois, code_padding).irow(0)
    crop_rect = [env["x"], env["y"], env["x"] + env["width"], env["y"] + env["height"]]

    # Hit test against grid AOIs
    col = kind_to_col(kind)
    code_aois = trial_aois[(trial_aois.kind == kind)]
    code_fixes = fixes.dropna(subset=[col])
    code_counts = code_fixes.groupby(col).duration_ms.sum()
    max_code_count = float(max(code_counts))

    def color_grid(kind, name, local_id):
        rel_count = code_counts.get(name, default=0) / max_code_count
        return matplotlib.colors.rgb2hex(cmap(rel_count))

    # Create syntax-based image
    code_box = trial_aois[(trial_aois.kind == "interface") &
                          (trial_aois.name == "code box")].irow(0)
    aoi_img = draw_rectangles(code_aois, screen_img, color_func=color_grid,
            alpha=syntax_alpha, outline=None)
    code_img = aoi_img.crop(crop_rect)

    # Add colorbar
    dpi = 90
    width, height = (0.25 * dpi), code_img.size[1]
    width_inches = width / float(dpi)
    height_inches = height / float(dpi)

    fig = pyplot.figure(figsize=(width_inches, height_inches), dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    norm = matplotlib.colors.Normalize(vmin=min(code_counts),
            vmax=max_code_count)

    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
    #cb.set_label("Total Fixation Duration (ms)")
    cb.outline.set_linewidth(0)

    # Convert plot to image
    plot_buffer = StringIO()
    fig.savefig(plot_buffer, format="png", dpi=dpi)
    pyplot.close(fig)
    plot_buffer.pos = 0
    plot_img = Image.open(plot_buffer)

    # Combine AOI and colorbar images
    horz_padding = (0.25 * dpi)
    width = int(code_img.size[0] + horz_padding + plot_img.size[0])
    height = int(code_img.size[1])
    final_img = Image.new("RGBA", (width, height), color="white")
    final_img.paste(code_img, (0, 0))
    final_img.paste(plot_img, (int(horz_padding + code_img.size[0]), 0))

    return final_img


def fixation_heatmap(fixations, screen_image, alpha=0.7,
                     dot_size=200, cmap=None, dpi=90):
    points = fixations[["fix_x", "fix_y"]].values
    heatmap_data = make_heatmap(points, screen_image.size, dot_size)

    width, height = screen_image.size
    width_inches = width / float(dpi)
    height_inches = height / float(dpi)

    fig = pyplot.figure(figsize=(width_inches, height_inches), dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    if cmap is None:
        cmap = pyplot.cm.get_cmap("jet")
        cmap._init()
        alphas = np.abs(np.linspace(-1.0, 1.0, cmap.N))
        cmap._lut[:-3, -1] = alphas

    ax.imshow(heatmap_data.T, interpolation="none", cmap=cmap)
    plot_buffer = StringIO()
    fig.savefig(plot_buffer, format="png", dpi=dpi)
    pyplot.close(fig)
    plot_buffer.pos = 0
    heatmap_image = Image.open(plot_buffer)

    heatmap_alpha = heatmap_image.split()[3]
    heatmap_image.putalpha(ImageEnhance.Brightness(heatmap_alpha).enhance(alpha))

    return Image.composite(heatmap_image, screen_image, heatmap_image)

def join_vertical(images, fill="white", spacing=10, line_width=1, line_color="black"):
    width = max([img.size[0] for img in images])
    height = sum([img.size[1] for img in images]) + \
            (spacing * (len(images) - 1)) + \
            (line_width * (len(images) - 1))

    final_img = Image.new("RGBA", (width, height), color=fill)

    # Paste images
    y = 0
    for img in images:
        final_img.paste(img, (0, y))
        y += img.size[1] + spacing + line_width

    # Draw lines
    draw = ImageDraw.Draw(final_img)
    y = 0
    for img in images:
        start_y = y
        y += img.size[1] + (spacing / 2) - (line_width / 2)
        draw.line((0, y, width, y), fill=line_color, width=line_width)
        y = start_y + img.size[1] + spacing + line_width

    del draw
    return final_img


def saccade_angle_plot(saccades, size=250, color="blue", bgcolor="white",
        outline="black", center_color="white", center_radius=2):
    w, h = size, size
    max_dist = saccades["dist_euclid"].max()
    scale = w / (2.0 * max_dist)

    img = Image.new("RGBA", (w, h), bgcolor)
    draw = ImageDraw.Draw(img)
    draw.ellipse((2, 2, w - 2, h - 2), outline=outline)

    for _, row in saccades.iterrows():
        x1, y1 = row["sacc_x1"], row["sacc_y1"]
        x2, y2 = row["sacc_x2"], row["sacc_y2"]
        dist = row["dist_euclid"]
        x = float(x2 - x1) * scale
        y = float(y2 - y1) * scale
        draw.line((w/2, h/2, w/2 + x, h/2 + y), fill=color)

    if center_color is not None:
        r = center_radius
        draw.ellipse((w/2 - r, h/2 - r, w/2 + r, h/2 + r), fill=center_color)
    del draw
    return img

def transition_centrality_graph(trans_matrix, name_map=None,
        cmap=None, edge_cmap=None, node_size=1200, font_size=18,
        figsize=(10, 10),
        **kwargs):
    import networkx as nx
    graph = nx.DiGraph(trans_matrix)

    # Drop orphaned nodes (blank lines)
    for n in graph.nodes():
        if graph.degree(n) == 0:
            graph.remove_node(n)

    if name_map is not None:
        graph = nx.relabel_nodes(graph, name_map)

    bc = nx.betweenness_centrality(graph)
    sorted_bc = sorted(bc.items(), key=operator.itemgetter(1))

    min_bc = sorted_bc[0][1]
    max_bc = sorted_bc[-1][1]
    colors = [bc[n] for n in graph.nodes()]

    if cmap is None:
        cdict = {
            "red":   ((0.0, 0.75, 0.75),
                      (1.0, 1.0, 1.0)),

            "green": ((0.0, 0.0, 0.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 1.0, 1.0)),

            "blue":  ((0.0, 0.0, 0.0),
                      (0.75, 0.0, 0.0),
                      (1.0, 0.0, 0.0))
        }

        cmap = LinearSegmentedColormap("custom", cdict)

    fig = pyplot.figure(figsize=figsize)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.set_axis_off()
    nx.draw_networkx(graph, ax=ax, node_size=node_size,
                     node_color=colors, cmap=cmap,
                     vmin=min_bc, vmax=max_bc,
                     linewidths=1.5, edge_color="gray",
                     **kwargs)

    return ax
