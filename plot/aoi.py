import numpy as np, operator, itertools as it
from matplotlib import pyplot, cm
from kelly_colors import kelly_colors
from ..aoi import get_aoi_kinds, make_aoi_column
from ..util import contrast_color

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

def draw_rectangles(aoi_rectangles, screen_image, colors=kelly_colors,
        outline="#000000", alpha=0.5):
    """Draws AOI rectangles on to an image."""
    from PIL import Image, ImageDraw, ImageEnhance

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
        col = make_aoi_column(kind)
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
