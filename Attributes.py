import logging

__author__ = 'simon'


def non_overridable(f):
    f.non_overridable = True
    return f


class NonOverrideable(type):
    def __new__(mcs, name, bases, dct):
        non_overridables = get_non_overridables(bases)
        for name in dct:
            if name in non_overridables:
                logging.error("You can not override %s, it is non-overridable" % name)
                raise Exception
        return type.__new__(mcs, name, bases, dct)


def get_non_overridables(bases):
    ret = []
    for source in bases:
        for name, attr in source.__dict__.items():
            if getattr(attr, "non_overridable", False):
                ret.append(name)
        ret.extend(get_non_overridables(source.__bases__))
    return ret