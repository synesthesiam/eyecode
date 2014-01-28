from StringIO import StringIO
from .. import DummyModel

def _print_results(code):
    model = DummyModel(StringIO(code))
    fixes, resps = model.run()

    print "Fixations"
    print "---------"
    print fixes

    print ""
    print "Responses"
    print "---------"
    print resps

    print ""
    print "LTM"
    print "---"
    print model.long_memory[1]

def test_basic():
    code = """a = "hi"
b = "bye"
print a + b

c = "street"
d = "penny"
print c + d

e = "5"
f = "3"
print e + f"""
    _print_results(code)


def test_long_line():
    code = """x = "a really, really long line with a lot of stuff on it for the model to read"
print 5

c = "a shorter line"
print 10"""
    _print_results(code)
