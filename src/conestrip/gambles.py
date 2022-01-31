# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from fractions import Fraction
from typing import List
from conestrip.utility import pretty_print


Gamble = List[Fraction]


class ConeGenerator(object):
    def __init__(self, gambles: List[Gamble]):
        self.gambles = gambles

    def __str__(self):
        return pretty_print(self.gambles)


class GeneralCone(object):
    def __init__(self, generators: List[ConeGenerator]):
        self.generators = generators

    def __str__(self):
        generators = ', \n'.join([' ' + pretty_print(x) for x in self.generators])
        return f"[\n{generators}\n]"


def parse_gamble(text: str) -> Gamble:
    return [Fraction(s) for s in text.strip().split()]


def parse_cone_generator(text: str) -> ConeGenerator:
    gambles = list(map(parse_gamble, text.strip().split('\n')))
    return ConeGenerator(gambles)


def parse_general_cone(text: str) -> GeneralCone:
    return GeneralCone(list(map(parse_cone_generator, re.split(r'\n\s*\n', text.strip()))))
