from ..util import file_to_text_buffer, stdoutIO
from mr_chips_eye import MrChipsEye
from retina import Retina
from pygments.lexers import PythonLexer
from pygments.token import Token

import ccm
from ccm.lib.actr import ACTR, Buffer, Memory, DMNoise, DMBaseLevel, Motor

# ----------------------------------------------------------------------------

def drop_spaces(tokens):
    return (t for t in tokens
            if not (t[0] == Token.Text and t[1] == u" "))

def combine_strings(tokens):
    strs = []
    for t in tokens:
        if t[0] == Token.String:
            if t[1] != '"':
                strs.append(t[1])
        else:
            if len(strs) > 0:
                value = "".join(strs)
                strs = []
                yield (Token.String, value)
            yield t

    if len(strs) > 0:
        value = "".join(strs)
        yield (Token.String, value)

def simplify_tokens(tokens):
    for t in tokens:
        kind, value = t
        if kind == Token.Operator.Word and value == u"in":
            kind = Token.Keyword
        elif kind == Token.Name.Function:
            kind = Token.Name
        yield (kind, value)

def make_pattern(pattern_str):
    pattern = []
    for part in pattern_str.split(" "):
        kind = part[0]
        token, value = None, None
        if kind == "K":
            token = Token.Keyword
        elif kind == "O":
            token = Token.Operator
        elif kind == "N":
            token = Token.Name
        elif kind == "I":
            token = Token.Number.Integer
        elif kind == "S":
            token = Token.String
        elif kind == "P":
            token = Token.Punctuation
        #elif kind in ["*", "+"]:
            #token = kind
        else:
            raise NotImplemented(kind)

        if len(part) > 1:
            value = part[2:-1]
        pattern.append((token, value))
    return pattern

def pattern_match(tokens, pattern):
    for t, p in zip(tokens, pattern):
        t_kind, t_value = t
        p_kind, p_value = p

        if t_kind != p_kind:
            return False
        if (p_value is not None) and (t_value != p_value):
            return False
    return True

# ----------------------------------------------------------------------------

class AgentEnvironment(ccm.Model):
    def key_pressed(self, k):
        pass

#class SubLineReader(ccm.Model):
    #ENCODING_TIME = 0.100

    #def __init__(self):
        #self.busy = False

    #def read(self):
        #self.busy = True
        #yield ENCODING_TIME
        #self.busy = False

class Agent(ACTR):
    goal      = Buffer()
    imaginal  = Buffer()
    #dm_buffer = Buffer()
    #dm        = Memory(dm_buffer, latency=0.05, threshold=0)
    #dm_noise  = DMNoise(dm, noise=0.0, baseNoise=0.0)
    #dm_base   = DMBaseLevel(dm, decay=0.5, limit=None)
    motor     = Motor()
    eye       = MrChipsEye()
    responses = []

    lexer = PythonLexer()

    goal.set("read-line")

    def __init__(self, text_buffer):
        ACTR.__init__(self)
        self.eye.text_buffer = text_buffer

        self.last_line = text_buffer.shape[0] - 1
        self.fovea = [i for i, s in enumerate(self.eye.retina.slots)
                      if s == Retina.SLOT_HIGH_RES]

        self.fovea_start = min(self.fovea)
        self.fovea_stop  = max(self.fovea)
        self.fovea_idx   = slice(self.fovea_start, self.fovea_stop + 1)

        self.mem_locals = {}
        self.mem_globals = {}

        self.imaginal.set("line:")

    def process_tokens(self, tokens):
        return list(drop_spaces(combine_strings(simplify_tokens(tokens)))) 

    def read_line(goal="read-line", eye="busy:False", motor="busy:False"):
        code_str = eye.view()[self.fovea_idx]
        tokens = self.process_tokens(lexer.get_tokens(code_str))
        print tokens
        goal.set("stop-task")
        #imaginal.modify(line=imaginal["line"] + code_str)

        #if len(code_str[(self.fovea_stop + 1):].strip()) == 0:
            ## Finished reading line, process it
            #goal.set("understand-line")
        #else:
            ## More text is left on the line. Move eye to the right.
            #next_pos = (eye.pos[0] + len(self.fovea), eye.pos[1])
            #eye.move_to(next_pos)
            #goal.set("read-line")

    #def understand_line(goal="understand-line"):
        #full_line = imaginal["line"]
        #imaginal.clear()

        ## Execute line in memory (if not blank)
        #if len(full_line.strip()) > 0:
            #response = ""

            #with stdoutIO() as out:
                #exec(full_line, self.mem_globals, self.mem_locals)
                #response = out.getvalue().strip()

            #if len(response) > 0:
                #imaginal.set("keys:{0}".format(response))
                #goal.set("type-response")
                #responses.append(response)
                #return

        ## Move to next line
        #next_pos = (0, eye.pos[1] + 1)

        #if next_pos[1] > self.last_line:
            ## Done reading
            #goal.set("stop-task")
        #else:
            #eye.move_to(next_pos)
            #goal.set("read-line")

    def type_response(goal="type-response", motor="busy:False"):
        keys = imaginal["keys"]
        motor.press_key(keys[0])
        if len(keys) > 1:
            keys = keys[1:]
            goal.set("type-response")
            imaginal.modify(keys=keys)
        else:
            imaginal.clear()
            imaginal.set("line:")
            goal.set("read-line")

            # TODO: Move eye to next line here?

    def stop_task(goal="stop-task"):
        self.stop()

class PythonDMModel:
    KEYSTROKE_TIME = 12

    def __init__(self, code_file):
        self.pad_left = 4
        self.text_buffer = file_to_text_buffer(code_file, pad_left=self.pad_left)

    def run(self):
        agent = Agent(self.text_buffer)
        env = AgentEnvironment()
        env.model = agent
        ccm.log_everything(env)
        agent.run()

        #self.time = 0
        #self.short_memory = []
        #self.long_memory = ({}, {})
        #self.responses = []

        #eye = self.eye
        #last_line = self.text_buffer.shape[0] - 1

        #fovea = [i for i, s in enumerate(eye.retina.slots)
                 #if s == Retina.SLOT_HIGH_RES]

        #fovea_start = min(fovea)
        #fovea_stop  = max(fovea)
        #fovea_idx   = slice(fovea_start, fovea_stop + 1)

        #while True:
            #code_str = eye.view()
            #fix_start = self.time
            #self.time += PythonDummyModel.ENCODING_TIME
            #self.short_memory.append(code_str[fovea_idx])

            ## Record fixation
            #self.fixations.append([eye.pos[0], eye.pos[1],
                #fix_start, self.time - fix_start])

            ## Decide on next fixation
            #next_pos = (self.eye.pos[0] + len(fovea), self.eye.pos[1])

            ## Look for whitespace on right
            #if len(code_str[(fovea_stop + 1):].strip()) == 0:

                ## Execute line in memory (if not blank)
                #full_line = "".join(self.short_memory)
                #if len(full_line.strip()) > 0:
                    #mem_globals, mem_locals = self.long_memory
                    #response = ""

                    #with stdoutIO() as out:
                        #exec(full_line, mem_globals, mem_locals)
                        #response = out.getvalue().strip()

                    #if len(response) > 0:
                        #self.time += (len(response) * PythonDummyModel.KEYSTROKE_TIME)
                        #self.responses.append(response)

                ## Move to next line
                #next_pos = (0, self.eye.pos[1] + 1)
                #self.short_memory = []

            #if next_pos[1] > last_line:
                #break

            ## Move eye
            #self.time += eye.move_to(next_pos)

        return agent.now(), agent.responses
