from ..util import file_to_text_buffer, stdoutIO
from eye import Eye
from retina import Retina

class PythonDummyModel:
    ENCODING_TIME  = 100
    KEYSTROKE_TIME = 12

    def __init__(self, code_file):
        self.pad_left = 4
        self.text_buffer = file_to_text_buffer(code_file, pad_left=self.pad_left)
        self.eye = Eye(self.text_buffer)
        self.time = 0
        self.short_memory = []
        self.long_memory = ({}, {})
        self.fixations = []
        self.responses = []

    def run(self):
        self.time = 0
        self.short_memory = []
        self.long_memory = ({}, {})
        self.responses = []

        eye = self.eye
        last_line = self.text_buffer.shape[0] - 1

        fovea = [i for i, s in enumerate(eye.retina.slots)
                 if s == Retina.SLOT_HIGH_RES]

        fovea_start = min(fovea)
        fovea_stop  = max(fovea)
        fovea_idx   = slice(fovea_start, fovea_stop + 1)

        while True:
            code_str = eye.view()
            fix_start = self.time
            self.time += PythonDummyModel.ENCODING_TIME
            self.short_memory.append(code_str[fovea_idx])

            # Record fixation
            self.fixations.append([eye.pos[0], eye.pos[1],
                fix_start, self.time - fix_start])

            # Decide on next fixation
            next_pos = (self.eye.pos[0] + len(fovea), self.eye.pos[1])

            # Look for whitespace on right
            if len(code_str[(fovea_stop + 1):].strip()) == 0:

                # Execute line in memory (if not blank)
                full_line = "".join(self.short_memory)
                if len(full_line.strip()) > 0:
                    mem_globals, mem_locals = self.long_memory
                    response = ""

                    with stdoutIO() as out:
                        exec(full_line, mem_globals, mem_locals)
                        response = out.getvalue().strip()

                    if len(response) > 0:
                        self.time += (len(response) * PythonDummyModel.KEYSTROKE_TIME)
                        self.responses.append(response)

                # Move to next line
                next_pos = (0, self.eye.pos[1] + 1)
                self.short_memory = []

            if next_pos[1] > last_line:
                break

            # Move eye
            self.time += eye.move_to(next_pos)

        return self.time, self.responses
