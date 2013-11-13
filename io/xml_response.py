import pandas
from lxml import etree
from datetime import datetime

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

def response_dataframe(xml_path, time_format=TIME_FORMAT):
    """Converts XML experiments into a data frame with a row for each trial"""
    root = etree.parse(xml_path)
    experiments = root.xpath("//experiment")

    rows = []
    for e in experiments:
        exp_id = int(e.attrib["id"])
        age = int(e.xpath(".//question[@name='age']/text()")[0])
        degree = e.xpath(".//question[@name='education']/text()")[0]
        gender = e.xpath(".//question[@name='gender']/text()")[0]
        py_years = float(e.xpath(".//question[@name='python_years']/text()")[0])
        prog_years = float(e.xpath(".//question[@name='programming_years']/text()")[0])
        location = e.attrib["location"]

        for t in e.xpath(".//trial"):
            id = int(t.attrib["id"])
            base = t.attrib["base"]
            version = t.attrib["version"]

            grade_category = t.attrib["grade-category"]
            grade_value = int(t.attrib["grade-value"])

            started = datetime.strptime(t.attrib["started"], time_format)
            ended = datetime.strptime(t.attrib["ended"], time_format)
            response_duration = float(t.attrib["response-duration"])

            code_lines = int(t.xpath(".//metric[@name='code lines']/@value")[0])

            rows.append([id, exp_id, base, version, grade_value, grade_category,
                started, ended, response_duration, py_years, prog_years, age,
                degree, gender, location, code_lines])

    cols = ("id", "exp_id", "base", "version", "grade_value",
            "grade_category", "started", "ended", "response_duration",
            "py_years", "prog_years", "age", "degree", "gender", "location",
            "code_lines")

    df = pandas.DataFrame(rows, columns=cols)

    # Derived columns
    df["duration"] = df.apply(lambda r: (r["ended"] - r["started"]).total_seconds(), axis=1)
    df["response_percent"] = df.apply(lambda r: r["duration"] / float(r["response_duration"])
        if r["response_duration"] > 0 else 0, axis=1)

    df["grade_common"] = df.grade_category.str.startswith("common")
    df["grade_perfect"] = df.grade_value == 10
    df["grade_correct"] = df.grade_value >= 7

    # Add numeric representations of some columns
    for col in ["gender", "degree"]:
        values = list(df[col].unique())
        df[col + "_num"] = df[col].apply(lambda v: values.index(v))

    return df
