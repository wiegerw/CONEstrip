# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from fractions import Fraction
from typing import List
from conestrip.utility import pretty_print


Gamble = List[Fraction]


class ConeElement(object):
    def __init__(self, gambles: List[Gamble]):
        self.gambles = gambles

    def __str__(self):
        return pretty_print(self.gambles)


Cone = List[ConeElement]


def parse_gamble(text: str) -> Gamble:
    return [Fraction(s) for s in text.strip().split()]


def parse_gambles(text: str) -> ConeElement:
    return ConeElement(list(map(parse_gamble, text.strip().split('\n'))))


def parse_cone(text: str) -> Cone:
    return list(map(parse_gambles, re.split(r'\n\s*\n', text.strip())))


def print_cone(cone: Cone) -> None:
    if not cone:
        print("[]")
    else:
        print("[")
        print(", \n".join([' ' + pretty_print(x) for x in cone]))
        print("]")
