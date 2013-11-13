from retina import Retina
from numpy.testing import assert_equal
from StringIO import StringIO
from ..util import file_to_text_buffer

def test_view_string():
    r = Retina()

    # Empty string (whitespace for all retina slots)
    s = r.view_string("")
    assert_equal(s, "".join([Retina.LOW_WHITESPACE] * len(r.slots)))

    # Letters
    s = r.view_string(" Hello World")
    assert_equal(s, " ***lo World     ")

    # Numbers
    s = r.view_string("12 This is a test")
    assert_equal(s, "## *his is a ****")

def test_view_line():
    r = Retina()
    s = r.view_line("x = [2, 8, 7, 9, -5, 0, 2]", 2)
    assert_equal(s, "- (#, 8, 7, 9. -#")

def test_view_buffer():
    r = Retina()
    code = """for i in [1, 2, 3, 4]:
    print "The count is", i
    print "Done counting"""

    buf = file_to_text_buffer(StringIO(code))
    s = r.view_buffer(buf, x=7, y=1)
    assert_equal(s, "** ^The count **^")
