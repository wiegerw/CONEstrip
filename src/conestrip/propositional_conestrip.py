# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, List, Optional, Set, Tuple
from z3 import *
from conestrip.cones import GeneralCone, Gamble, ConeGenerator, ConvexCombination, linear_combination, parse_gamble

PropositionalGamble = List[Fraction]  # the coefficients g_phi of gamble g
PropositionalConeGenerator = List[PropositionalGamble]
PropositionalGeneralCone = List[PropositionalConeGenerator]

BooleanGamble = List[Bool]
PropositionalBasis = List[BooleanGamble]  # a list of basic functions


def gamble_coefficients(g: Gamble, Phi: PropositionalBasis) -> PropositionalGamble:
    """
    Get the coefficients of gamble g with respect to the basis Phi
    @param g: a gamble
    @param Phi: a sequence of basic functions
    """

    n = len(g)
    k = len(Phi)
    x = Reals(' '.join(f'x{i}' for i in range(k)))

    s = Solver()
    for j in range(n):
        eqn = sum([Phi[i][j] * x[i] for i in range(k)]) == g[j]
        s.add(eqn)

    s.check()
    model = s.model()
    return [model[xi] for xi in x]


def parse_boolean_gamble(text: str) -> Gamble:
    result = [Fraction(s) for s in text.strip().split()]
    assert all(f in [0,1] for f in result)
    return result


def parse_propositional_basis(text: str) -> PropositionalBasis:
    gambles = list(map(parse_boolean_gamble, text.strip().split('\n')))
    return gambles
