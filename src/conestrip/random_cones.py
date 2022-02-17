import random
from fractions import Fraction
from typing import List, Tuple
from z3 import *
from conestrip.cones import ConeGenerator, Gamble, GeneralCone, ConvexCombination
from conestrip.utility import random_nonzero_rationals_summing_to_one, inner_product
from conestrip.conestrip import conestrip1_constraints


def linear_combination(lambda_: ConvexCombination, gambles: List[Gamble]) -> Gamble:
    m = len(gambles)
    n = len(gambles[0])
    result = [Fraction(0)] * n
    for i in range(m):
        g = gambles[i]
        for j in range(n):
            result[j] += lambda_[i] * g[j]
    return result


def random_inside_point(R: ConeGenerator) -> Tuple[Gamble, ConvexCombination]:
    """
    Generates a random point that is inside the cone generated by R
    @param R:
    @return:
    """

    m = len(R.gambles)
    lambda_ = random_nonzero_rationals_summing_to_one(m)
    return linear_combination(lambda_, R.gambles), lambda_


def random_border_point(R: ConeGenerator) -> Tuple[Gamble, ConvexCombination]:
    """
    Generates a random point that is in the border of the cone generated by R
    @param R:
    @return:
    """

    # converts indices to points
    def make_facet(indices: Tuple[int]) -> ConeGenerator:
        return ConeGenerator([R.vertices[i] for i in indices])

    facet = random.choice(R.facets)
    border_facet = make_facet(facet)
    return random_inside_point(border_facet)


def add_random_border_cone(R: ConeGenerator) -> ConeGenerator:
    # converts indices to points
    def make_facet(indices: Tuple[int]) -> ConeGenerator:
        return ConeGenerator([R.vertices[i] for i in indices])

    facet = random.choice(R.facets)
    facet_index = R.facets.index(facet)
    border_facet = make_facet(facet)

    # generate a cone that is contained in border_face
    m = len(border_facet.gambles)
    result = []
    coefficients = []
    for i in range(m):
        g, lambda_ = random_inside_point(border_facet)
        result.append(g)
        coefficients.append(lambda_)
    generator = ConeGenerator(result)
    generator.parent = (R, facet_index, coefficients)
    R.children[facet_index].append(generator)
    return generator


def add_random_border_cones(R: GeneralCone, n: int, allow_multiple_children: bool = False) -> None:
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
