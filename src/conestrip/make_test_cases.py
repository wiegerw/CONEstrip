import random
from fractions import Fraction
from typing import List
import cdd
from conestrip.polyhedron import Polyhedron
from conestrip.gambles import Gamble, Cone


def gambles_to_polyhedron(gambles: List[Gamble]) -> Polyhedron:
    n = len(gambles[0])
    origin = [Fraction(0)] * n
    A = [origin] + gambles
    A = [[Fraction(1)] + x for x in A]
    mat = cdd.Matrix(A)
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


def convex_combination(lambda_: List[Fraction], gambles: List[Gamble]) -> Gamble:
    m = len(gambles)
    n = len(gambles[0])
    result = [Fraction(0)] * n
    for i in range(m):
        g = gambles[i]
        for j in range(n):
            result[j] += lambda_[i] * g[j]
    return result


def random_border_cone(R: List[Gamble]) -> List[Gamble]:
    poly = gambles_to_polyhedron(R)
    vertices = poly.vertices()

    # converts indices to points
    def make_face(indices: List[int]) -> List[Gamble]:
        return [vertices[i] for i in indices]

    border_faces = [make_face(face) for face in poly.face_vertex_adjacencies() if 0 in face]
    border_face = random.choice(border_faces)
    border_cone = border_face[1:]  # remove the origin

    # generate a cone that is contained in border_cone
    m = len(border_cone)
    result = []
    for i in range(m):
        lambda_ = random_rationals_summing_to_one(m)
        result.append(convex_combination(lambda_, border_cone))

    return result


def add_random_border_cones(R: Cone, n: int) -> None:
    for i in range(n):
        R1 = [r for r in R if len(r) >= 2]
        r = random.choice(R1)
        cone = random_border_cone(r)
        R.append(cone)