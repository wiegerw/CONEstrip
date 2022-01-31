import random
from fractions import Fraction
from typing import Dict, List
import cdd
from conestrip.polyhedron import Polyhedron
from conestrip.gambles import Gamble, GeneralCone, ConeGenerator


def gambles_to_polyhedron(cone: ConeGenerator) -> Polyhedron:
    # N.B. gambles are treated as directions
    A = [[Fraction(0)] + x for x in cone.gambles]
    mat = cdd.Matrix(A, linear=False)
    mat.rep_type = cdd.RepType.GENERATOR
    mat.canonicalize()
    poly = Polyhedron(mat)
    poly.to_V()
    return poly


def random_floats_summing_to_one(n: int) -> List[float]:
    values = [random.random() for i in range(n)]
    s = sum(values)
    return [x / s for x in values]


def random_rationals_summing_to_one(n: int) -> List[Fraction]:
    values = random_floats_summing_to_one(n)
    v = [int(round(1000*x)) / 1000 for x in values]
    v = v[:-1]
    v.append(1 - sum(v))
    return [Fraction(vi) for vi in v]


def convex_combination(lambda_: List[Fraction], cone: ConeGenerator) -> Gamble:
    m = len(cone.gambles)
    n = len(cone.gambles[0])
    result = [Fraction(0)] * n
    for i in range(m):
        g = cone.gambles[i]
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
        result.append(convex_combination(lambda_, border_face))

    return ConeGenerator(result)


def add_random_border_cones(R: GeneralCone, n: int) -> None:
    for i in range(n):
        R1 = [r for r in R.generators if len(r.gambles) >= 2]
        r = random.choice(R1)
        cone = random_border_cone(r)
        R.generators.append(cone)
