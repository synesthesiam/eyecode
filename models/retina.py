import string

class UnknownLowResCharacterException(Exception):
    pass

class Retina:

    NUMBERS    = string.printable[:10]
    LETTERS    = string.printable[10:62]
    TOP_OPS    = ["'", "^", "\""]
    MIDDLE_OPS = ["-", "+", "*", "="]
    BOTTOM_OPS = [".", "_", ","]
    BRACKETS   = ["[", "]", "(", ")", "{", "}", ":"]

    SLOT_LOW_RES  = 0
    SLOT_HIGH_RES = 1

    LOW_WHITESPACE = " "
    LOW_LETTER     = "*"
    LOW_NUMBER     = "#"
    LOW_TOP_OP     = "^"
    LOW_MIDDLE_OP  = "-"
    LOW_BOTTOM_OP  = "."
    LOW_BRACKET    = "("

    def __init__(self, slots=None):
        self.slots = slots

        if self.slots is None:
            # Default retina 4 - 9 - 4
            self.slots = ([Retina.SLOT_LOW_RES] * 4) +\
                         ([Retina.SLOT_HIGH_RES] * 9) +\
                         ([Retina.SLOT_LOW_RES] * 4)

    def view_string(self, s, pad_right=True):
        if len(s) < len(self.slots):
            num_spaces = len(self.slots) - len(s)
            if pad_right:
                s += " " * num_spaces
            else:
                s = (" " * num_spaces) + s

        viewable = []
        for c, slot in zip(s, self.slots):
            if slot == Retina.SLOT_LOW_RES:
                if c == " ":
                    viewable.append(Retina.LOW_WHITESPACE)
                elif c in Retina.NUMBERS:
                    viewable.append(Retina.LOW_NUMBER)
                elif c in Retina.LETTERS:
                    viewable.append(Retina.LOW_LETTER)
                elif c in Retina.TOP_OPS:
                    viewable.append(Retina.LOW_TOP_OP)
                elif c in Retina.MIDDLE_OPS:
                    viewable.append(Retina.LOW_MIDDLE_OP)
                elif c in Retina.BOTTOM_OPS:
                    viewable.append(Retina.LOW_BOTTOM_OP)
                elif c in Retina.BRACKETS:
                    viewable.append(Retina.LOW_BRACKET)
                else:
                    raise UnknownLowResCharacterException(c)
            else:
                viewable.append(c)

        return "".join(viewable)

    def view_line(self, line, x=0):
        return self.view_string(line[x:])

    def view_buffer(self, buffer, x=0, y=0):
        line = "".join(buffer[y, :])
        return self.view_line(line, x=x)
