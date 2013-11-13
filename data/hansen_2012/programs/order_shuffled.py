def h(x):
    return f(x) + g(x)

def f(x):
    return x + 4

def g(x):
    return x * 2

x = 1
a = f(x)
b = g(x)
c = h(x)
print a, b, c
