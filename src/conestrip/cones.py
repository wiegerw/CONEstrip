# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
import re
from fractions import Fraction
from typing import List
import cdd
from polyhedron import Polyhedron
from utility import random_rationals_summing_to_one, pretty_print


Gamble = List[Fraction]


class ConeGenerator(object):
    def __init__(self, gambles: List[Gamble]):
        self.gambles = gambles

    def __getitem__(self, item):
        return self.gambles[item]

    def __len__(self):
        return len(self.gambles)

    def __str__(self):
        return pretty_print(self.gambles)


class GeneralCone(object):
    def __init__(self, generators: List[ConeGenerator]):
        self.generators = generators
        # self.facets =
        # self.border_cones =

    def __getitem__(self, item):
        return self.generators[item]

    def __len__(self):
        return len(self.generators)

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


def gambles_to_polyhedron(cone: ConeGenerator) -> Polyhedron:
    # N.B. gambles are treated as directions
    A = [[Fraction(0)] + x for x in cone.gambles]
    mat = cdd.Matrix(A, linear=False)
    mat.rep_type = cdd.RepType.GENERATOR
    mat.canonicalize()
    poly = Polyhedron(mat)
    poly.to_V()
    return poly


def convex_combination(lambda_: List[Fraction], gambles: List[Gamble]) -> Gamble:
    m = len(gambles)
    n = len(gambles[0])
    result = [Fraction(0)] * n
    for i in range(m):
        g = gambles[i]
        for j in range(n):
            result[j] += lambda_[i] * g[j]
    return result


def random_border_cone(R: ConeGenerator) -> ConeGenerator:
    poly = gambles_to_polyhedron(R)
    vertices = poly.vertices()

    # converts indices to points
    def make_face(indices: List[int]) -> ConeGenerator:
        return ConeGenerator([vertices[i] for i in indices])

    border_faces = [make_face(face) for face in poly.face_vertex_adjacencies()]
    border_face = random.choice(border_faces)

    # generate a cone that is contained in border_face
    m = len(border_face.gambles)
    result = []
    for i in range(m):
        lambda_ = random_rationals_summing_to_one(m)
        result.append(convex_combination(lambda_, border_face.gambles))
    return ConeGenerator(result)


def add_random_border_cones(R: GeneralCone, n: int) -> None:
    for i in range(n):
        R1 = [r for r in R.generators if len(r.gambles) >= 2]
        r = random.choice(R1)
        cone = random_border_cone(r)
        R.generators.append(cone)
