import time
import random

__author__ = 'simon'

def foo():
    yield random.random()
    print("foo - 1")
    yield random.random()
    print("foo - 2")
    yield random.random()
    print("foo - 3")

def foo2():
    return foo()

for item,item2 in zip(foo2(),foo2()):
    time.sleep(1)
    print(str(item) + " " + str(item2))

time.sleep(1)