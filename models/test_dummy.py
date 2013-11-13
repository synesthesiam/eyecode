from StringIO import StringIO
from python_dummy import PythonDummyModel

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
    model = PythonDummyModel(StringIO(code))
    time, responses = model.run()
    print time, responses
    print ""
    print model.long_memory[1]
    print ""
    print model.fixations
