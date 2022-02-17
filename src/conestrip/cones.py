# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from fractions import Fraction
from typing import Dict, List, Tuple, Optional
import cdd
from conestrip.polyhedron import Polyhedron


Gamble = List[Fraction]
ConvexCombination = List[Fraction]  # positive values that sum to one
PositiveCombination = List[Fraction]  # positive values

def print_gamble(g: Gamble) -> str:
    return ' '.join(map(str, g))


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


class ConeGenerator(object):
    """
    A cone generator, defined by a finite set of gambles. Each facet of the generated cone can have links to
    lower dimensional cones that are contained in this facet.
    """

    def __init__(self, gambles: List[Gamble]):
        self.gambles = gambles
        poly = gambles_to_polyhedron(gambles)
        self.vertices: List[List[Fraction]] = poly.vertices()  # Note that the vertices may be in a different order than the gambles
        facets: List[Tuple[int]] = poly.face_vertex_adjacencies()
        self.facets = [tuple(sorted(facet)) for facet in facets]

        # If self.parent == (R, i), then this generator is contained in the i-th facet of R.
        self.parent: Optional[Tuple[ConeGenerator, int]] = None
        self.children: Dict[int, List[ConeGenerator]] = {i: [] for i in range(len(self.facets))}  # maps facets to the generators contained in it

    def __getitem__(self, item):
        return self.gambles[item]

    def __len__(self):
        return len(self.gambles)

    def __str__(self):
        return '\n'.join([print_gamble(g) for g in self.gambles])


class GeneralCone(object):
    def __init__(self, generators: List[ConeGenerator]):
        self.generators = generators

    def __getitem__(self, item):
        return self.generators[item]

    def __len__(self):
        return len(self.generators)

    def __str__(self):
        return '\n\n'.join([str(cone) for cone in self.generators])


def parse_gamble(text: str) -> Gamble:
    return [Fraction(s) for s in text.strip().split()]


def parse_cone_generator(text: str) -> ConeGenerator:
    gambles = list(map(parse_gamble, text.strip().split('\n')))
    return ConeGenerator(gambles)


def parse_general_cone(text: str) -> GeneralCone:
    return GeneralCone(list(map(parse_cone_generator, re.split(r'\n\s*\n', text.strip()))))
