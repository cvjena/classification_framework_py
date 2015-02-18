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

def use_generator(gen):
    for item, item2 in zip(foo2(), foo2()):
        time.sleep(1)
        print(str(item) + " " + str(item2))

gen= foo()
use_generator(gen)
time.sleep(1)