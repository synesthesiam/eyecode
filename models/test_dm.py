from StringIO import StringIO
from python_dm import PythonDMModel

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
    model = PythonDMModel(StringIO(code))
    time, responses = model.run()
    print time, responses
