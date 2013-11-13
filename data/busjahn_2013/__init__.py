import pandas, os
from PIL import Image

DATA_DIR = os.path.abspath(os.path.dirname(__file__))

def _read_csv(name):
    return pandas.read_csv(os.path.join(DATA_DIR, "{0}.csv".format(name)))

def raw_fixations():
    return _read_csv("raw_fixations")

def all_fixations():
    return _read_csv("all_fixations")

def line_fixations():
    return _read_csv("line_fixations")

def areas_of_interest(program_name=None):
    aois = _read_csv("areas_of_interest")
    if program_name is None:
        return aois
    else:
        return aois[aois.program == program_name]

def program_image(name):
    return Image.open(os.path.join(DATA_DIR, "images", "{0}.png".format(name)))

def code_image(name):
    return Image.open(os.path.join(DATA_DIR, "images", "{0}-pygment.png".format(name)))

def program_code(name):
    return open(os.path.join(DATA_DIR, "programs", "{0}.java".format(name))).readlines()

def lines_text_order(program_name):
    lines = _read_csv("text_order")
    return lines[lines.program == program_name]

def lines_execution_order(program_name):
    lines = _read_csv("exec_order")
    return lines[lines.program == program_name]

def tags():
    return _read_csv("tags")
