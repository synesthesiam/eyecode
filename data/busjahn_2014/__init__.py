import os
import pandas
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
