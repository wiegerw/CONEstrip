# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import itertools
import math
from fractions import Fraction
from typing import Any, List, Tuple

import z3

from conestrip.gamble_algorithms import gamble_coefficients
from conestrip.cones import Gamble, GambleBasis, GeneralCone, ConeGenerator
from conestrip.propositional_cones import PropositionalSentence, BooleanVariable, PropositionalBasis, \
    PropositionalGeneralCone, PropositionalConeGenerator, PropositionalGamble
from conestrip.propositional_conestrip import propositional_conestrip_algorithm
from conestrip.utility import is_solved


def sentence_to_gamble(phi: PropositionalSentence, B: List[BooleanVariable]) -> Gamble:
    m = len(B)
    solver = z3.Solver()
    result = []
    for i, values in enumerate(itertools.product([False, True], repeat=m)):
        constraints = [B[j] == values[j] for j in range(m)] + [phi]
        val = Fraction(1) if solver.check(constraints) == z3.sat else Fraction(0)
        result.append(val)
    return result


def gamble_to_sentence(g: Gamble, B: List[BooleanVariable]) -> PropositionalSentence:
    assert all(gi in [0, 1] for gi in g)
    m = len(B)
    clauses = [True] * m
    for i, values in enumerate(itertools.product([False, True], repeat=m)):
        if g[i] == 0:
            clause = z3.Or([b == z3.Not(val) for b, val in zip(B, values)])
            clauses.append(clause)
    return z3.simplify(z3.And(clauses))


def default_basis(n: int) -> GambleBasis:
    result = []
    x = [Fraction(0)] * n
    for i in range(n):
        y = x[:]
        y[i] = Fraction(1)
        result.append(y)
    return result


def default_propositional_basis(n: int) -> Tuple[PropositionalBasis, List[BooleanVariable]]:
    B = z3.Bools([f'b{i}' for i in range(n)])
    gambles = default_basis(2**n)
    Phi = [gamble_to_sentence(g, B) for g in gambles]
    return Phi, B


def convert_gamble(g: Gamble, Phi: GambleBasis) -> PropositionalGamble:
    return gamble_coefficients(g, Phi)


def convert_cone_generator(R: ConeGenerator, Phi: GambleBasis) -> PropositionalConeGenerator:
    return [convert_gamble(g, Phi) for g in R]


def convert_general_cone(R: GeneralCone, Phi: GambleBasis) -> PropositionalGeneralCone:
    return [convert_cone_generator(D, Phi) for D in R]


def print_propositional_gamble(g: PropositionalGamble) -> str:
    return ' '.join(map(str, g))


def print_propositional_cone_generator(D: PropositionalConeGenerator) -> str:
    return '\n'.join(print_propositional_gamble(g) for g in D)


def print_propositional_general_cone(R: PropositionalGeneralCone) -> str:
    return '\n\n'.join(print_propositional_cone_generator(D) for D in R)


def convert_conestrip_problem(R0: GeneralCone, f0: Gamble) -> Tuple[PropositionalGeneralCone, PropositionalGamble, List[BooleanVariable], PropositionalBasis, PropositionalSentence, PropositionalSentence, PropositionalSentence]:
    n = len(f0)
    m = int(math.log2(n))
    if 2**m != n:
        raise RuntimeError(f'The gamble size {n} is not a power of 2')
    Phi, B = default_propositional_basis(m)
    psi = True
    psi_Gamma = True
    psi_Delta = True
    Phi_gambles = [sentence_to_gamble(phi, B) for phi in Phi]

    R1 = convert_general_cone(R0, Phi_gambles)
    f1 = convert_gamble(f0, Phi_gambles)
    assert f0 == f1

    return R1, f1, B, Phi, psi, psi_Gamma, psi_Delta


def propositional_conestrip_solution(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int] = None, Omega_Delta: List[int] = None, verbose: bool = False) -> Tuple[Any, Any, Any, Any]:
    R1, f1, B, Phi, psi, psi_Gamma, psi_Delta = convert_conestrip_problem(R0, f0)
    return propositional_conestrip_algorithm(R1, f1, B, Phi, psi, psi_Gamma, psi_Delta, verbose)


def is_in_propositional_cone_generator(R: ConeGenerator, g: Gamble, verbose: bool = False) -> Any:
    solution = propositional_conestrip_solution([R], g, verbose=verbose)
    return is_solved(solution)
