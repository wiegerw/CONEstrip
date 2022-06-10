# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from fractions import Fraction
from typing import List
import z3
from conestrip.cones import Gamble


def gamble_coefficients(g: Gamble, Phi: List[Gamble]) -> List[Fraction]:
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


