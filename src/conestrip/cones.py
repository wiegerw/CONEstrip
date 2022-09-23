# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from fractions import Fraction
from typing import List
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


def print_gamble(g: Gamble, pretty=False) -> str:
    if pretty:
        return '[{}]'.format(', '.join(f'{float(x)}' for x in g))
    else:
        return ' '.join(map(str, g))


def print_gambles(G: List[Gamble], pretty=False) -> str:
    if pretty:
        return '[{}]'.format(', '.join(print_gamble(g, pretty) for g in G))
    else:
        return '\n'.join(print_gamble(g) for g in G)


def print_cone_generator(D: ConeGenerator, pretty=False) -> str:
    if pretty:
         return print_gambles(D, pretty)
    else:
         return '\n'.join(print_gamble(g) for g in D)


def print_general_cone(R: GeneralCone, pretty=False) -> str:
    if pretty:
        items = [print_cone_generator(D, pretty) for D in R]
        return '[\n  {}\n]'.format('\n\n  '.join(map(str, items)))
    else:
        return '\n\n'.join(print_cone_generator(D) for D in R)


def parse_gamble(text: str) -> Gamble:
    return [Fraction(s) for s in text.strip().split()]


def parse_cone_generator(text: str) -> ConeGenerator:
    return list(map(parse_gamble, text.strip().split('\n')))


def parse_general_cone(text: str) -> GeneralCone:
    return list(map(parse_cone_generator, re.split(r'\n\s*\n', text.strip())))


def print_fractions(x: List[Fraction], pretty=False) -> str:
    if pretty:
        return '[{}]'.format(', '.join(f'{float(xi)}' for xi in x))
    else:
        return '[{}]'.format(', '.join(f'{xi}' for xi in x))


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
