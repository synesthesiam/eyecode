"""Complete dataset from the eyeCode experiment (http://arxiv.org/abs/1304.5257)"""

import pandas, os, gzip
from glob import glob

DATA_DIR = os.path.abspath(os.path.dirname(__file__))
XML_DIR = os.path.join(DATA_DIR, "xml")

def __read_csv(file_name):
    return pandas.read_csv(gzip.open(os.path.join(DATA_DIR, "{0}.csv.gz".format(file_name))))

def programs():
    """Code metrics for all programs.
    
    Returns
    -------
    df : pandas DataFrame

        * base: program base category
        * version: program version
        * code_chars: number of characters in code
        * code_lines: number of lines in code
        * cyclo_comp: cyclomatic complexity of program
        * hal_effort: Halstead effort of program
        * hal_volume: Halstead volume of program
        * output_chars: number of characters in correct output
        * output_lines: number of lines in correct output
        * category: 'expectation' or 'notation'
        * program_name: joined base_version
    """
    return __read_csv("programs")

def line_categories():
    """Automatically assigned categories for program lines"""
    return __read_csv("line_categories")

def experiments():
    """Experiment/participant information.
    
    Returns
    -------
    df : pandas DataFrame

        * exp_id: experiment/participant id
        * age: age of participant in years
        * degree: highest degree obtained by participant
            * One of ['bachelors', 'masters', 'phd', 'none', 'other']
        * gender: participant's gender
            * One of ['male', 'female', 'unreported']
        * py_years: years of Python experience
        * prog_years: years of overall programming experience
        * cs_major: participant is/was a Computer Science major
            * One of ['current', 'no', 'past']
        * difficulty: post-experiment perceived program difficulty
            * One of ['easy', 'medium', 'hard']
        * guess_correct: post-experiment guess at correct answers
            * One of ['most', 'half', 'few']
        * total_grade: sum of all trial grades (usually out of 100)
        * duration_sec : experiment duration in seconds
        * location: participant set
            * One of ['bloomington', 'mturk', 'web']
        * xxx_num: numeric representation of column xxx

    """
    return __read_csv("experiments")

def trials():
    """Individual trial information"""
    return __read_csv("trials")

def trial_responses():
    """Timestamped text responses for all trials"""
    return __read_csv("trial_responses")

def all_fixations():
    """Hit-tested fixations for all trials"""
    return __read_csv("all_fixations")

def line_fixations():
    """Hit-tested fixations with numeric line column for all trials"""
    return __read_csv("line_fixations")

def raw_fixations():
    """Fixations without offset correction and hit testing"""
    return __read_csv("raw_fixations")

def areas_of_interest():
    """AOI rectangles for all trials"""
    return __read_csv("aois")

def youtube_videos():
    """URLs for uploaded eye-tracking videos"""
    return __read_csv("youtube")

def trial_screen_path(trial_id):
    """Path to screenshot image for the given eye-tracking trial"""
    return os.path.join(DATA_DIR, "screens", "{0}.png".format(trial_id))

def trial_screen(trial_id):
    """Screenshot image for the given eye-tracking trial"""
    from PIL import Image
    return Image.open(trial_screen_path(trial_id))

def program_code(base, version):
    """Lines of code for the given program base and version"""
    return open(os.path.join(DATA_DIR, "programs", "{0}_{1}.py".format(base, version)), "r").readlines()

def program_output(base, version):
    """Lines of correct text output for the given program base and version"""
    return open(os.path.join(DATA_DIR, "programs", "output", "{0}_{1}.py.txt".format(base, version)), "r").readlines()

def program_lines_with_code(base, version):
    """List of non-blank lines for the given program base and version (1-based)"""
    lines = program_code(base, version)
    return [i+1 for i, line in enumerate(lines) if len(line.strip()) > 0]

def program_image(base, version):
    """Image with syntax-highlighted, line-numbered code for the given program base and version"""
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
