# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
import re
from fractions import Fraction
from typing import Dict, List, Tuple
import cdd
from conestrip.polyhedron import Polyhedron
from conestrip.utility import random_rationals_summing_to_one, inner_product


Gamble = List[Fraction]


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
        self.facet_adjacencies: Dict[int, List[ConeGenerator]] = {i: [] for i in range(len(self.facets))}  # maps facets to the generators contained in it

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


def add_random_border_cone(R: ConeGenerator) -> ConeGenerator:

    def linear_combination(lambda_: List[Fraction], gambles: List[Gamble]) -> Gamble:
        m = len(gambles)
        n = len(gambles[0])
        result = [Fraction(0)] * n
        for i in range(m):
            g = gambles[i]
            for j in range(n):
                result[j] += lambda_[i] * g[j]
        return result

    # converts indices to points
    def make_facet(indices: Tuple[int]) -> ConeGenerator:
        return ConeGenerator([R.vertices[i] for i in indices])

    facet = random.choice(R.facets)
    facet_index = R.facets.index(facet)
    border_facet = make_facet(facet)

    # generate a cone that is contained in border_face
    m = len(border_facet.gambles)
    result = []
    for i in range(m):
        lambda_ = random_rationals_summing_to_one(m)
        result.append(linear_combination(lambda_, border_facet.gambles))
    generator = ConeGenerator(result)
    R.facet_adjacencies[facet_index].append(generator)
    return generator


def add_random_border_cones(R: GeneralCone, n: int) -> None:
    for i in range(n):
        R1 = [r for r in R.generators if len(r.gambles) >= 2]
        r = random.choice(R1)
        generator = add_random_border_cone(r)
        R.generators.append(generator)


# randomly generate a vector in R^n with coordinates in the range [-bound, ..., bound]
def random_vector(n: int, bound: int) -> List[Fraction]:
    return [Fraction(random.randrange(-bound, bound+1)) for _ in range(n)]


def random_cone_generator(dimension: int, generator_size: int, bound: int, normal=None) -> ConeGenerator:
    n = dimension

    if not normal:
        normal = random_vector(n, bound)

    # randomly generate x such that inner_product(normal, x) > 0
    def generate() -> List[Fraction]:
        while True:
            x = random_vector(n, bound)
            if inner_product(normal, x) > 0:
                return x

    # generate size points in { x \in R^n | inner_product(normal, x) > 0 }
    return ConeGenerator([generate() for _ in range(generator_size)])


def random_general_cone(cone_size: int, dimension: int, generator_size: int, bound: int) -> GeneralCone:
    return GeneralCone([random_cone_generator(dimension, generator_size, bound) for _ in range(cone_size)])
