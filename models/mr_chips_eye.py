import ccm
import math
from retina import Retina

class MrChipsEye(ccm.Model):
    TIME_PREP       = 0.150
    TIME_MOTOR      = 0.050
    TIME_SACCADE    = 0.020
    TIME_PER_DEGREE = 0.002

    def __init__(self, text_buffer=None, retina=None, pos=(0, 0),
            char_vis=(0.226, 0.404)):
        self.busy = False
        self.text_buffer = text_buffer
        self.retina = retina
        self.pos = pos
        self.char_vis = char_vis

        if self.retina is None:
            self.retina = Retina()

    def move_to(self, new_pos):
        if self.busy:
            return

        self.busy = True
        x, y = new_pos

        self.log._ = "Eye movement preparation"
        yield MrChipsEye.TIME_PREP

        self.log._ = "Eye movement motor preparation"
        yield MrChipsEye.TIME_MOTOR

        # Distance in degrees of visual angle
        dist = math.sqrt(((self.pos[0] - x) * self.char_vis[0])**2 +
                         ((self.pos[1] - y) * self.char_vis[1])**2)

        saccade_time = MrChipsEye.TIME_SACCADE + (dist * MrChipsEye.TIME_PER_DEGREE)

        self.log._ = "Eye movement saccade"
        yield saccade_time
        self.log._ = "Finished eye movement"

        self.pos = (x, y)
        self.busy = False

    def view(self):
        assert self.text_buffer is not None
        x, y = self.pos
        return self.retina.view_buffer(self.text_buffer, x, y)
