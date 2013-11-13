import pandas, os, gzip
from glob import glob

DATA_DIR = os.path.abspath(os.path.dirname(__file__))
XML_DIR = os.path.join(DATA_DIR, "xml")

def __read_csv(file_name):
    return pandas.read_csv(gzip.open(os.path.join(DATA_DIR, "{0}.csv.gz".format(file_name))))

def programs():
    return __read_csv("programs")

def line_categories():
    return __read_csv("line_categories")

def experiments():
    return __read_csv("experiments")

def trials():
    return __read_csv("trials")

def trial_responses():
    return __read_csv("trial_responses")

def all_fixations():
    return __read_csv("all_fixations")

def line_fixations():
    return __read_csv("line_fixations")

def raw_fixations():
    return __read_csv("raw_fixations")

def areas_of_interest():
    return __read_csv("aois")

def youtube_videos():
    return __read_csv("youtube")

def trial_screen_path(trial_id):
    return os.path.join(DATA_DIR, "screens", "{0}.png".format(trial_id))

def trial_screen(trial_id):
    from PIL import Image
    return Image.open(trial_screen_path(trial_id))

def program_code(base, version):
    return open(os.path.join(DATA_DIR, "programs", "{0}_{1}.py".format(base, version)), "r").readlines()

def program_output(base, version):
    return open(os.path.join(DATA_DIR, "programs", "output", "{0}_{1}.py.txt".format(base, version)), "r").readlines()

def program_lines_with_code(base, version):
    lines = program_code(base, version)
    return [i+1 for i, line in enumerate(lines) if len(line.strip()) > 0]

def program_image(base, version):
    from PIL import Image
    return Image.open(os.path.join(DATA_DIR, "images", "programs", "{0}_{1}.py.png".format(base, version)))

def xml_trial_ids():
    return [int(os.path.split(p)[1][:-len(".xml.gz")])
            for p in glob(os.path.join(XML_DIR, "*.xml.gz"))]

def xml_trial_path(trial_id):
    return os.path.join(XML_DIR, "{0:02d}.xml.gz".format(trial_id))

def xml_trial(trial_id):
    from lxml import etree
    return etree.parse(xml_trial_path(trial_id))
