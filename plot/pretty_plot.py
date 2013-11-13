import numpy as np
from matplotlib.artist import Artist
from matplotlib import font_manager as fm, colors as mc

class FilteredArtistList(Artist):
    def __init__(self, artist_list, filter):
        self._artist_list = artist_list
        self._filter = filter
        Artist.__init__(self)

    def draw(self, renderer):
        renderer.start_rasterizing()
        renderer.start_filter()
        for a in self._artist_list:
            a.draw(renderer)
        renderer.stop_filter(self._filter)
        renderer.stop_rasterizing()
        
def smooth1d(x, window_len):
    # copied from http://www.scipy.org/Cookbook/SignalSmooth

    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]

def smooth2d(A, sigma=3):

    window_len = max(int(sigma), 3)*2+1
    A1 = np.array([smooth1d(x, window_len) for x in np.asarray(A)])
    A2 = np.transpose(A1)
    A3 = np.array([smooth1d(x, window_len) for x in A2])
    A4 = np.transpose(A3)

    return A4

class BaseFilter(object):
    def prepare_image(self, src_image, dpi, pad):
        ny, nx, depth = src_image.shape
        #tgt_image = np.zeros([pad*2+ny, pad*2+nx, depth], dtype="d")
        padded_src = np.zeros([pad*2+ny, pad*2+nx, depth], dtype="d")
        padded_src[pad:-pad, pad:-pad,:] = src_image[:,:,:]

        return padded_src#, tgt_image

    def get_pad(self, dpi):
        return 0

    def __call__(self, im, dpi):
        pad = self.get_pad(dpi)
        padded_src = self.prepare_image(im, dpi, pad)
        tgt_image = self.process_image(padded_src, dpi)
        return tgt_image, -pad, -pad
    
class OffsetFilter(BaseFilter):
    def __init__(self, offsets=None):
        if offsets is None:
            self.offsets = (0, 0)
        else:
            self.offsets = offsets

    def get_pad(self, dpi):
        return int(max(*self.offsets)/72.*dpi)

    def process_image(self, padded_src, dpi):
        ox, oy = self.offsets
        a1 = np.roll(padded_src, int(ox/72.*dpi), axis=1)
        a2 = np.roll(a1, -int(oy/72.*dpi), axis=0)
        return a2

class GaussianFilter(BaseFilter):
    def __init__(self, sigma, alpha=0.5, color=None):
        self.sigma = sigma
        self.alpha = alpha
        if color is None:
            self.color=(0, 0, 0)
        else:
            self.color=color

    def get_pad(self, dpi):
        return int(self.sigma*3/72.*dpi)


    def process_image(self, padded_src, dpi):
        #offsetx, offsety = int(self.offsets[0]), int(self.offsets[1])
        tgt_image = np.zeros_like(padded_src)
        aa = smooth2d(padded_src[:,:,-1]*self.alpha,
                      self.sigma/72.*dpi)
        tgt_image[:,:,-1] = aa
        tgt_image[:,:,:-1] = self.color
        return tgt_image

class DropShadowFilter(BaseFilter):
    def __init__(self, sigma, alpha=0.3, color=None, offsets=None):
        self.gauss_filter = GaussianFilter(sigma, alpha, color)
        self.offset_filter = OffsetFilter(offsets)

    def get_pad(self, dpi):
        return max(self.gauss_filter.get_pad(dpi),
                   self.offset_filter.get_pad(dpi))

    def process_image(self, padded_src, dpi):
        t1 = self.gauss_filter.process_image(padded_src, dpi)
        t2 = self.offset_filter.process_image(t1, dpi)
        return t2

def dark_edges(ax):
    # Iterate over the patches in the axes
    for patch in ax.patches:
        # Get the facecolor of the patch
        ec = patch.get_facecolor()
        # Make that color a bit darker and set it as edge color
        patch.set_edgecolor(tuple(x * 0.7 for x in ec[:3]) + (ec[3],))
        patch.set_linewidth(1.4)
        
def change_fonts(ax, size):
    # Load a font family
    prop = fm.FontProperties(family=['Arial'], size=size)
    # Set it to all texts in the axes
    for text in ax.texts:
        text.set_fontproperties(prop)
        
def shade_patches(ax):
    # Set our custom shadow_filter for all patches in the axes
    for patch in ax.patches:
        patch.set_agg_filter(shadow_filter)

def shadow_filter(image, dpi):
    # Get the shape of the image
    nx, ny, depth = image.shape
    # Create a mash grid
    xx, yy = np.mgrid[0:nx, 0:ny]
    # Draw a circular "shadow"
    circle = (xx + nx * 4) ** 2 + (yy + ny) ** 2
    # Normalize
    circle -= circle.min()
    circle = circle / circle.max()
    # Steepness
    value = circle.clip(0.3, 0.6) + 0.4
    saturation = 1 - circle.clip(0.7, 0.8)
    # Normalize
    saturation -= saturation.min() - 0.1
    saturation = saturation / saturation.max()
    # Convert the rgb part (without alpha) to hsv
    hsv = mc.rgb_to_hsv(image[:,:,:3])
    # Multiply the value of hsv image with the shadow
    hsv[:,:,2] = hsv[:,:,2] * value
    # Highlights with saturation
    hsv[:,:,1] = hsv[:,:,1] * saturation
    # Copy the hsv back into the image (we haven't touched alpha)
    image[:,:,:3] = mc.hsv_to_rgb(hsv)
    # the return values are: new_image, offset_x, offset_y
    return image, 0, 0

def shade_axis(ax, radius=35, alpha=0.3, size=20):
    shadow = FilteredArtistList(
        ax.patches,
        DropShadowFilter(
            radius,
            offsets=(-5,-7), 
            alpha=alpha
        )
    )

    ax.add_artist(shadow)
    dark_edges(ax)
    #shade_patches(ax)
    change_fonts(ax, size)
