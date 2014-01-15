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

def test_if():
    code = """x = 5
if x > 3:
    print "red"
else:
    print "blue" """
    _print_results(code)
