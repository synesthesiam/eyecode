import pandas

from . import SimpleEye, DiscreteRetina
from ..util import file_to_text_buffer, stdoutIO

class DummyModel:
    ENCODING_TIME  = 100
    PROCESS_TIME   = 250
    KEYSTROKE_TIME = 12

    STATE_END                = "end"
    STATE_READING_LINES      = "reading-lines"
    STATE_READING_LINE_PART  = "reading-line-part"
    STATE_EXECUTING_LINE     = "executing-line"
    STATE_READING_BLOCK      = "reading-block"
    STATE_READING_BLOCK_PART = "reading-block-part"
    STATE_EXECUTING_BLOCK    = "executing-block"

    def __init__(self, code_file, pad_left=4, eye=None):
        self.pad_left = pad_left
        self.text_buffer = file_to_text_buffer(code_file, pad_left=self.pad_left)
        self.eye = eye or SimpleEye(self.text_buffer)

        self.time = 0
        self.short_memory = []
        self.long_memory = ({}, {})
        self.fixations = []
        self.responses = []
        self.typed_output = ""

    def run(self):
        """Runs the model with the current eye and text buffer.

        Returns
        -------
        fixes, resps : tuple of pandas DataFrames
            DataFrames with model fixations and keystroke responses
        
        """

        # Reset clock and memory/responses
        self.time = 0
        self.short_memory = []
        self.long_memory = ({}, {})
        self.fixations = []
        self.responses = []
        self.typed_output = ""

        eye = self.eye
        last_line = self.text_buffer.shape[0] - 1

        fovea_idx = eye.retina.fovea_slice()
        fovea_start, fovea_stop = fovea_idx.start, fovea_idx.stop
        fovea_len = fovea_stop - fovea_start

        # Loop until last line is read
        state = DummyModel.STATE_READING_LINE_PART
        while state != DummyModel.STATE_END:
            if state == DummyModel.STATE_READING_LINES:
                line_idx = eye.pos[1]

                # Check if we're beyond the last line
                if line_idx < last_line:
                    state = DummyModel.STATE_READING_LINE_PART
                else:
                    # End of program
                    state = DummyModel.STATE_END

            elif state == DummyModel.STATE_READING_LINE_PART:
                # Look at current position
                code_str = eye.view()

                # Fixation starts now
                fix_start = self.time

                # Time taken to encode current sensor contents (fixed)
                self.time += DummyModel.ENCODING_TIME

                # Add fovea contents to STM
                self.short_memory.append(code_str[fovea_idx])

                # Record fixation
                self.fixations.append([eye.pos[0], eye.pos[1],
                    fix_start, self.time - fix_start])

                # Check right of fovea for whitespace
                right_of_fovea = code_str[(fovea_stop + 1):].strip()

                if len(right_of_fovea) > 0:
                    # More to read on this line. Move eye to the right.
                    next_pos = (eye.pos[0] + fovea_len, eye.pos[1])
                    self.time += eye.move_to(next_pos)
                else:
                    # Execute line in STM
                    state = DummyModel.STATE_EXECUTING_LINE

            elif state == DummyModel.STATE_READING_BLOCK:
                pass

            elif state == DummyModel.STATE_READING_BLOCK_PART:
                pass

            elif state == DummyModel.STATE_EXECUTING_LINE:
                self.__execute_line()

                # Move eye to next line, all the way to the left
                next_pos = (0, eye.pos[1] + 1)
                self.time += eye.move_to(next_pos)

                state = DummyModel.STATE_READING_LINES

            elif state == DummyModel.STATE_EXECUTING_BLOCK:
                pass

        # --------------------------------------------------------------------
        # Model is done running. Package fixations and responses as DataFrames.

        fix_cols = ["fix_x", "fix_y", "start_ms", "duration_ms"]
        fixes_df = pandas.DataFrame(self.fixations, columns=fix_cols)
        fixes_df["end_ms"] = fixes_df["start_ms"] + fixes_df["duration_ms"]

        resp_cols = ["time_ms", "response"]
        resps_df = pandas.DataFrame(self.responses, columns=resp_cols)

        return fixes_df, resps_df

    def __execute_line(self):
        """Executes the current line or block in STM and types in the resulting
        output."""
        # Execute line in STM (if not blank)
        full_line = "".join(self.short_memory)
        if len(full_line.strip()) > 0:
            
            # Use locals/globals from LTM
            mem_globals, mem_locals = self.long_memory
            response = ""

            # Execute current line and record response
            with stdoutIO() as out:
                exec(full_line, mem_globals, mem_locals)
                response = out.getvalue().strip()

            if len(response) > 0:
                # Response time is proportional to number of keystrokes
                for c in response + '\n':
                    self.typed_output += c
                    self.time += DummyModel.KEYSTROKE_TIME
                    self.responses.append([self.time, self.typed_output])

        # Clear STM
        self.short_memory = []

