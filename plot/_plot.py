import operator, itertools as it

import numpy as np
from matplotlib import pyplot, cm
from PIL import Image, ImageDraw, ImageEnhance

from kelly_colors import kelly_colors
from ..aoi import get_aoi_kinds, kind_to_col
from ..util import contrast_color

# AOI plots {{{

def draw_rectangles(aoi_rectangles, screen_image, colors=kelly_colors,
        outline="#000000", alpha=0.5):
    """Draws AOI rectangles on to an image.
    
    Parameters
    ----------
    aoi_rectangles : pandas DataFrame
        DataFrame with a row for each rectangle (x, y, width, height columns)

    screen_image : PIL Image
        Image on top of which AOI rectangles will be drawn

    colors : list, optional
        List of PIL fill colors for each AOI kind. Colors will by cycled through
        if there are more AOI kinds than colors. Default is kelly_colors.

    outline : str, optional
        Rectangle outline color (default is black: '#000000')

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

def aoi_transitions(trans_matrix, name_map=None, name_fun=operator.itemgetter(1),
        show_probs=True, ax=None, cmap=None, figsize=None): 
    """Plots an AOI transition matrix with a colorbar."""
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
        ax.set_xticklabels([name_fun(name_map[c]) for c in cols])
        pyplot.setp(ax.get_xticklabels(), rotation=90)

    ax.set_xlabel("To AOI")

    # y-axis
    ax.set_ylim(0, len(rows))
    ax.set_yticks(rows + 0.5)

    if name_map is None:
        ax.set_yticklabels(rows)
    else:
        ax.set_yticklabels([name_fun(name_map[r]) for r in rows])

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

def kind_bars(fixations, fix_col="duration_ms", ax=None, figsize=None):
    if ax is None:
        pyplot.figure(figsize=figsize)
        ax = pyplot.axes()

    aoi_kinds = get_aoi_kinds(fixations)
    colors = it.cycle(kelly_colors)

    bottom_start = 0
    xticks = []
    xlabels = []
    width = 1.0
    for kind, color in zip(aoi_kinds, colors):
        col = kind_to_col(kind)
        times = fixations[[col, fix_col]].groupby(col)\
                .duration_ms.sum()
        ind = np.arange(len(times)) + bottom_start
        data = np.array(times)
        ax.bar(ind, data, width=width, color=color, label=kind)
        
        for i, name in enumerate(times.index):
            xticks.append(bottom_start + (width / 1.75) + i)
            xlabels.append(name)
        
        bottom_start += len(times) + 2

    ax.set_title("Fixation Time by AOI")
    ax.set_xlim(-1, len(xticks) + (len(aoi_kinds) * 2) - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90)

    ax.grid()
    ax.set_ylabel("Duration (ms)")
    ax.legend(bbox_to_anchor=(1.1, 1.05))

    return ax

# }}}

# ----------------------------------------------------------------------------

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

# }}}

