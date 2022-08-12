# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from fractions import Fraction
from typing import Dict, List, Tuple, Optional

from conestrip.cones import Gamble, gambles_to_polyhedron, print_gamble, parse_gamble
from conestrip.conestrip_z3 import is_in_cone_generator, is_in_closed_cone_generator, \
    is_in_cone_generator_border, solve_conestrip4, is_in_general_cone


class ExtendedConeGenerator(object):
    """
    A cone generator, enhanced with some additional structure.
    If parent is defined, then this cone generator is inside the border of the parent.
    Similarly, the children of this cone generator are contained inside the border of this cone generator.
    """

    def __init__(self, gambles: List[Gamble]):
        self.gambles = gambles
        poly = gambles_to_polyhedron(gambles)
        self.vertices: List[List[Fraction]] = poly.vertices()  # Note that the vertices may be in a different order than the gambles
        facets: List[Tuple[int]] = poly.face_vertex_adjacencies()
        self.facets = [tuple(sorted(facet)) for facet in facets]

        # If self.parent == (R, i), then this generator is contained in the i-th facet of R.
        self.parent: Optional[Tuple[ExtendedConeGenerator, int]] = None
        self.children: Dict[int, List[ExtendedConeGenerator]] = {i: [] for i in range(len(self.facets))}  # maps facets to the generators contained in it

    def __getitem__(self, item):
        return self.gambles[item]

    def __len__(self):
        return len(self.gambles)

    def __str__(self):
        return '\n'.join([print_gamble(g) for g in self.gambles])

    def to_cone_generator(self):
        return self.gambles


class ExtendedGeneralCone(object):
    def __init__(self, generators: List[ExtendedConeGenerator]):
        self.generators = generators

    def __getitem__(self, item):
        return self.generators[item]

    def __len__(self):
        return len(self.generators)

    def __str__(self):
        return '\n\n'.join([str(cone) for cone in self.generators])

    def to_general_cone(self):
        return [generator.to_cone_generator() for generator in self.generators]


def parse_extended_cone_generator(text: str) -> ExtendedConeGenerator:
    gambles = list(map(parse_gamble, text.strip().split('\n')))
    return ExtendedConeGenerator(gambles)


def parse_extended_general_cone(text: str) -> ExtendedGeneralCone:
    return ExtendedGeneralCone(list(map(parse_extended_cone_generator, re.split(r'\n\s*\n', text.strip()))))


def is_in_cone_generator_extended(R: ExtendedConeGenerator, g: Gamble, verbose: bool = False) -> bool:
    return is_in_cone_generator(R.to_cone_generator(), g, verbose)


def is_in_closed_cone_generator_extended(R: ExtendedConeGenerator, g: Gamble) -> bool:
    return is_in_closed_cone_generator(R.to_cone_generator(), g)


def is_in_cone_generator_border_extended(R: ExtendedConeGenerator, g: Gamble) -> bool:
    return is_in_cone_generator_border(R.to_cone_generator(), g)


def is_in_general_cone_extended(cone: ExtendedGeneralCone, g: Gamble, solver=solve_conestrip4) -> bool:
    return is_in_general_cone(cone.to_general_cone(), g, solver)
