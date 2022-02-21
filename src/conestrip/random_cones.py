import random
from typing import Any, List, Tuple
from z3 import *
from conestrip.cones import ConeGenerator, Gamble, GeneralCone, ConvexCombination, linear_combination, gambles_to_polyhedron, print_gamble
from conestrip.utility import random_nonzero_rationals_summing_to_one, inner_product


def remove_redundant_vertices(vertices: List[Gamble]) -> List[Gamble]:
    poly = gambles_to_polyhedron(vertices)
    return poly.vertices()


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
    n = len(R.gambles[0])

    if len(R.gambles) == 1:
        return [Fraction(0)] * n, [Fraction(0)]

    # converts indices to points
    def make_facet(indices: Tuple[int]) -> ConeGenerator:
        return ConeGenerator([R.vertices[i] for i in indices])

    facet = random.choice(R.facets)  # contains the indices of the vertices
    border_facet = make_facet(facet)
    facet = list(facet)

    # if the border facet has the same dimension, remove a random vertex
    if len(border_facet.gambles) == len(R.gambles):
        m = len(border_facet.gambles)
        i = random.randint(0, m - 1)
        facet.pop(i)
        border_facet.gambles.pop(i)

    x, lambda_ = random_inside_point(border_facet)
    coefficients = [Fraction(0)] * len(R.gambles)
    for i, j in enumerate(facet):
        coefficients[j] = lambda_[i]
    return x, coefficients


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
    for i in range(m):
        g, lambda_ = random_inside_point(border_facet)
        result.append(g)

    generator = ConeGenerator(remove_redundant_vertices(result))
    generator.parent = (R, facet_index)
    R.children[facet_index].append(generator)
    return generator


def add_random_border_cones(R: GeneralCone, n: int, allow_multiple_children: bool = False) -> None:
    def is_allowed(r: ConeGenerator) -> bool:
        if len(r.gambles) < 2:
            return False
        if allow_multiple_children:
            return True
        for _, successors in enumerate(r.children):
            if not successors:
                return True
        return False

    for i in range(n):
        R1 = [r for r in R.generators if is_allowed(r)]
        r = random.choice(R1)
        generator = add_random_border_cone(r)
        R.generators.append(generator)


# randomly generate a vector in R^n with coordinates in the range [-bound, ..., bound]
def random_vector(n: int, bound: int) -> List[Fraction]:
    return [Fraction(random.randrange(-bound, bound+1)) for _ in range(n)]


def random_cone_generator(dimension: int, generator_size: int, bound: int, normal=None) -> ConeGenerator:
    """
    Generates a random cone generator.
    @param dimension: The size of the gambles in the cone generator.
    @param generator_size: The number of gambles in the cone generator.
    @param bound: The largest absolute value of the coordinates.
    @param normal: The normal vector of a half space (optional).
    @return: The generated cone generator.
    """
    n = dimension

    normal = None
    while not normal:
        normal = random_vector(n, bound)

    # randomly generate x such that inner_product(normal, x) > 0
    def generate() -> List[Fraction]:
        while True:
            x = random_vector(n, bound)
            if inner_product(normal, x) > 0:
                return x

    # generate size points in { x \in R^n | inner_product(normal, x) > 0 }
    vertices = []
    while len(vertices) < generator_size:
        vertices = vertices + [generate() for _ in range(generator_size - len(vertices))]
        vertices = remove_redundant_vertices(vertices)

    return ConeGenerator(vertices)


def random_general_cone(cone_size: int, dimension: int, generator_size: int, bound: int) -> GeneralCone:
    return GeneralCone([random_cone_generator(dimension, generator_size, bound) for _ in range(cone_size)])
