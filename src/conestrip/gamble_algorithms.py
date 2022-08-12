# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from fractions import Fraction
from typing import Any, List, Optional
import z3
from z3 import simplify

from conestrip.cones import Gamble, GambleBasis, ConvexCombination, linear_combination


def gamble_coefficients(g: Gamble, Phi: GambleBasis) -> List[Fraction]:
    """
    Get the coefficients of gamble g with respect to the basis Phi
    @param g: a gamble
    @param Phi: a sequence of basic functions
    """

    n = len(g)
    k = len(Phi)
    x = z3.Reals(' '.join(f'x{i}' for i in range(k)))

    solver = z3.Solver()
    for j in range(n):
        eqn = sum([Phi[i][j] * x[i] for i in range(k)]) == g[j]
        solver.add(eqn)

    solver.check()
    model = solver.model()
    return [model[xi] for xi in x]


def is_convex_combination(f: Gamble, G: List[Gamble]):
    """
    Determines if f is a convex combination of the elements in G
    @param f: a gamble
    @param G: a sequence of gambles
    """

    n = len(f)
    k = len(G)
    lambda_ = z3.Reals(' '.join(f'lambda_{i}' for i in range(k)))

    solver = z3.Solver()
    solver.add(sum(lambda_) == 1)
    for x in lambda_:
        solver.add(0 <= x)
        solver.add(x <= 1)
    for j in range(n):
        eqn = sum([lambda_[i] * G[i][j] for i in range(k)]) == f[j]
        solver.add(eqn)

    if solver.check() == z3.unsat:
        return None
    model = solver.model()
    return [model[x] for x in lambda_]


def is_positive_combination(f: Gamble, G: List[Gamble]) -> Optional[List[Any]]:
    """
    Determines if f is a positive combination of the elements in G
    @param f: a gamble
    @param G: a sequence of gambles
    """

    n = len(f)
    k = len(G)
    lambda_ = z3.Reals(' '.join(f'lambda_{i}' for i in range(k)))

    solver = z3.Solver()
    for x in lambda_:
        solver.add(0 <= x)
    for j in range(n):
        eqn = sum([lambda_[i] * G[i][j] for i in range(k)]) == f[j]
        solver.add(eqn)

    if solver.check() == z3.unsat:
        return None
    model = solver.model()
    return [model[x] for x in lambda_]


def simplified_linear_combination(lambda_: ConvexCombination, gambles: List[Gamble]) -> Gamble:
    result = linear_combination(lambda_, gambles)
    return [simplify(x) for x in result]
