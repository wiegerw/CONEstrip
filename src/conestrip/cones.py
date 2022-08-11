# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from fractions import Fraction
from typing import Dict, List, Tuple, Optional
import cdd
from conestrip.polyhedron import Polyhedron


Gamble = List[Fraction]
ConeGenerator = List[Gamble]
GeneralCone = List[ConeGenerator]
GambleBasis = List[Gamble]          # a list of gambles that spans the probability space
ConvexCombination = List[Fraction]  # positive values that sum to one


def linear_combination(lambda_: ConvexCombination, gambles: List[Gamble]) -> Gamble:
    m = len(gambles)
    n = len(gambles[0])
    result = [Fraction(0)] * n
    for i in range(m):
        g = gambles[i]
        for j in range(n):
            result[j] += lambda_[i] * g[j]
    return result


def print_gamble(g: Gamble) -> str:
    return ' '.join(map(str, g))


def print_gambles(G: List[Gamble]) -> str:
    return '\n'.join(print_gamble(g) for g in G)


def gambles_to_polyhedron(gambles: List[Gamble]) -> Polyhedron:
    """
    Defines a cone that encloses a list of gambles
    @param gambles:
    @return:
    """
    # N.B. gambles are treated as directions
    A = [[Fraction(0)] + x for x in gambles]
    mat = cdd.Matrix(A, linear=False)
    mat.rep_type = cdd.RepType.GENERATOR
    mat.canonicalize()
    poly = Polyhedron(mat)
    poly.to_V()
    return poly


def parse_gamble(text: str) -> Gamble:
    return [Fraction(s) for s in text.strip().split()]


def parse_cone_generator(text: str) -> ConeGenerator:
    return list(map(parse_gamble, text.strip().split('\n')))


def parse_general_cone(text: str) -> GeneralCone:
    return list(map(parse_cone_generator, re.split(r'\n\s*\n', text.strip())))
