import string
import math

class UnknownLowResCharacterException(Exception):
    pass

class DiscreteRetina:
    """Discrete visual sensor based on the Mr. Chips model
    of text reading (Legge, 2002).

    """
    # Numeric characters (0-9)
    NUMBERS    = string.printable[:10]

    # Alphabetic characters (a-z, A-Z)
    LETTERS    = string.printable[10:62]

    # Top, middle, and bottom characters
    TOP_OPS    = ["'", "^", "\""]
    MIDDLE_OPS = ["-", "+", "*", "="]
    BOTTOM_OPS = [".", "_", ","]

    # Bracketing characters
    BRACKETS   = ["[", "]", "(", ")", "{", "}", ":"]

    # Constants for marking slots as hi/low resolution
    SLOT_LOW_RES  = "L"
    SLOT_HIGH_RES = "H"

    # Replacement characters for low-res slots
    LOW_WHITESPACE = " "
    LOW_LETTER     = "*"
    LOW_NUMBER     = "#"
    LOW_TOP_OP     = "^"
    LOW_MIDDLE_OP  = "-"
    LOW_BOTTOM_OP  = "."
    LOW_BRACKET    = "("

    def __init__(self, slots="LLLLHHHHHHHHHLLLL"):
        """Initializes a discrete retina.

        Parameters
        ----------
        slots : str, optional
            String with low/high resolution marked with 'L' and 'H'
            respectively.

        Returns
        -------
        dr : DiscreteRetina instance

        """
        self.slots = slots

    def view_string(self, s, pad_right=True):
        """Looks at a string and returns visible characters.
        Low-res characters are replaced with their LOW_ identifiers.
        If s is smaller than the sensor, s is padded with spaces.

        Parameters
        ----------
        s : str
            String to look at

        pad_right : bool, optional
            If True, pad s on the right. If False, pad s on the left.
            Default is True (right).

        Returns
        -------
        visible : str
            Visible characters on the sensor with low-res characters
            replaced.
            
        """

        # Pad s
        if len(s) < len(self.slots):
            num_spaces = len(self.slots) - len(s)
            if pad_right:
                s += " " * num_spaces
            else:
                s = (" " * num_spaces) + s

        # Replace low-res characters
        viewable = []
        for c, slot in zip(s, self.slots):
            if slot == DiscreteRetina.SLOT_LOW_RES:
                if c == " ":
                    viewable.append(DiscreteRetina.LOW_WHITESPACE)
                elif c in DiscreteRetina.NUMBERS:
                    viewable.append(DiscreteRetina.LOW_NUMBER)
                elif c in DiscreteRetina.LETTERS:
                    viewable.append(DiscreteRetina.LOW_LETTER)
                elif c in DiscreteRetina.TOP_OPS:
                    viewable.append(DiscreteRetina.LOW_TOP_OP)
                elif c in DiscreteRetina.MIDDLE_OPS:
                    viewable.append(DiscreteRetina.LOW_MIDDLE_OP)
                elif c in DiscreteRetina.BOTTOM_OPS:
                    viewable.append(DiscreteRetina.LOW_BOTTOM_OP)
                elif c in DiscreteRetina.BRACKETS:
                    viewable.append(DiscreteRetina.LOW_BRACKET)
                else:
                    raise UnknownLowResCharacterException(c)
            else:
                viewable.append(c)

        return "".join(viewable)

    def view_line(self, line, x=0):
        """Views a line starting at position x.

        Parameters
        ----------
        line : str
            Line to view

        x : int, optional
            Starting position to view line (default: 0)

        Returns
        -------
        visible : str
            Visible characters on the sensor with low-res characters
            replaced.

        """
        return self.view_string(line[x:])

    def view_buffer(self, buffer, x=0, y=0):
        """Views a 2-D buffer starting at position x, y.

        Parameters
        ----------
        buffer : array_like of str
            2-D array of str with dimensions y, x

        x : int, optional
            Starting horizontal position (default: 0)

        y : int, optional
            Starting vertical position (default: 0)

        Returns
        -------
        visible : str
            Visible characters on the sensor with low-res characters
            replaced.

        """
        line = "".join(buffer[y, :])
        return self.view_line(line, x=x)

    def fovea_slice(self):
        """Returns an index slice for the high-res slots of the sensor (fovea).
        Assumes that high-res slots are contiguous.

        Returns
        -------

        fovea_slice : slice
            Slice from start to end of "fovea"
        """
        fovea = [i for i, s in enumerate(self.slots)
                 if s == DiscreteRetina.SLOT_HIGH_RES]

        assert len(fovea) > 0, "No fovea"

        # Get fovea bounds
        fovea_start = min(fovea)
        fovea_stop  = max(fovea)

        return slice(fovea_start, fovea_stop + 1)

# ----------------------------------------------------------------------------

class SimpleEye:
    TIME_MOVE       = 220
    TIME_PER_DEGREE = 2

    def __init__(self, text_buffer, retina=None, pos=(0, 0),
            char_vis=(0.226, 0.404)):
        """Initializes an instance of SimpleEye. Time to move the eye is a
        function of TIME_MOVE (for prep time, motor programming, etc.) and
        TIME_PER_DEGREE (multiplied by distance moved).

        Parameters
        ----------
        text_buffer : array_like of str
            2-D array of str with dimensions y, x

        retina : sensor_like or None, optional
            Sensor to view buffer. If None, a default DiscreteRetina
            is created.

        pos : tuple of int
            Starting position of sensor (x, y). Default is (0, 0).

        char_vis : tuple of float
            Size of each character in buffer in degrees of visual angle
            (horizontal, vertical). Default is (0.226, 0.404).

        Returns
        -------
        eye : SimpleEye instance

        """
        self.text_buffer = text_buffer
        self.retina = retina
        self.pos = pos
        self.char_vis = char_vis

        if self.retina is None:
            self.retina = DiscreteRetina()

    def move_to(self, new_pos):
        """Moves the eye/sensor to a new position in the buffer.

        Parameters
        ----------
        new_pos : tuple of int
            New position in the buffer (x, y)

        Returns
        -------
        time : float
            Time taken to move the eye in milliseconds

        """
        x, y = new_pos
        time = SimpleEye.TIME_MOVE

        # Distance in degrees of visual angle
        dist = math.sqrt(((self.pos[0] - x) * self.char_vis[0])**2 +
                         ((self.pos[1] - y) * self.char_vis[1])**2)

        time += (dist * SimpleEye.TIME_PER_DEGREE)
        self.pos = (x, y)

        return time

    def view(self):
        """Returns the characters visible on the sensor.

        Returns
        -------
        visible : str
            Visible characters on the sensor with low-res characters
            replaced.

        """
        x, y = self.pos
        return self.retina.view_buffer(self.text_buffer, x, y)
