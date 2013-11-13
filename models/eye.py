import math
from retina import Retina

class Eye:
    TIME_PREP       = 150
    TIME_MOTOR      = 50
    TIME_SACCADE    = 20
    TIME_PER_DEGREE = 2

    def __init__(self, text_buffer, retina=None, pos=(0, 0),
            char_vis=(0.226, 0.404)):
        self.text_buffer = text_buffer
        self.retina = retina
        self.pos = pos
        self.char_vis = char_vis

        if self.retina is None:
            self.retina = Retina()

    def move_to(self, new_pos):
        x, y = new_pos
        time = Eye.TIME_PREP + Eye.TIME_MOTOR + Eye.TIME_SACCADE

        # Distance in degrees of visual angle
        dist = math.sqrt(((self.pos[0] - x) * self.char_vis[0])**2 +
                         ((self.pos[1] - y) * self.char_vis[1])**2)

        time += (dist * Eye.TIME_PER_DEGREE)
        self.pos = (x, y)

        return time

    def view(self):
        x, y = self.pos
        return self.retina.view_buffer(self.text_buffer, x, y)
