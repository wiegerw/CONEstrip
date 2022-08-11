# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from fractions import Fraction
from typing import Dict, List, Tuple, Optional

from more_itertools.recipes import flatten
from z3 import Real, Implies, And, Not, ForAll, Solver, sat, simplify

from conestrip.cones import Gamble, gambles_to_polyhedron, print_gamble, parse_gamble, ConvexCombination
from conestrip.conestrip import conestrip1_constraints


class ExtendedConeGenerator(object):
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


def random_between_point(R1: ExtendedConeGenerator, verbose: bool = False) -> Optional[Tuple[Gamble, ConvexCombination]]:
    """
    Generates a point that is contained in R1.parent, but not in R1.
    @precondition R1.parent != None
    @param R1: A cone generator
    """

    R0, facet_index = R1.parent

    n = len(R1.gambles[0])
    Omega_Gamma = list(range(n))
    Omega_Delta = list(range(n))

    cone0 = [R0.to_cone_generator()]
    cone1 = [R1.to_cone_generator()]

    # variables
    f = [Real(f'f{d}') for d in range(n)]
    lambda0 = [Real(f'lambda0{d}') for d in range(len(cone0))]
    nu0 = [[Real(f'nu0_{d}_{i}') for i in range(len(cone0[d]))] for d in range(len(cone0))]
    lambda1 = [Real(f'lambda1{d}') for d in range(len(cone1))]
    nu1 = [[Real(f'nu1_{d}_{i}') for i in range(len(cone1[d]))] for d in range(len(cone1))]

    # f is inside R0, and not inside R1
    constraints0 = list(flatten(conestrip1_constraints(cone0, f, Omega_Gamma, Omega_Delta, (lambda0, nu0), verbose)))
    lambda_nu_constraints, omega_constraints = conestrip1_constraints(cone1, f, Omega_Gamma, Omega_Delta, (lambda1, nu1), verbose)
    constraint1 = ForAll(lambda1 + list(flatten(nu1)), Implies(And(lambda_nu_constraints), Not(And(omega_constraints))))
    constraints = constraints0 + [constraint1]

    solver = Solver()
    solver.add(constraints)
    if solver.check() == sat:
        model = solver.model()
        lambda0_solution = [model.evaluate(lambda0[d]) for d in range(len(cone0))]
        lambda1_solution = [model.evaluate(lambda1[d]) for d in range(len(cone1))]
        nu0_solution = [[model.evaluate(nu0[d][i]) for i in range(len(cone0[d]))] for d in range(len(cone0))]
        nu1_solution = [[model.evaluate(nu1[d][i]) for i in range(len(cone1[d]))] for d in range(len(cone1))]
        f = [model.evaluate(f[d]) for d in range(len(f))]
        if verbose:
            print('--- solution ---')
            print('lambda0 =', lambda0_solution)
            print('nu0 =', nu0_solution)
            print('lambda1 =', lambda1_solution)
            print('nu1 =', nu1_solution)
        coefficients = [simplify(x) for x in nu0_solution[0]]
        return f, coefficients
    return None
