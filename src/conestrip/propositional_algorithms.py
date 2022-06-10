# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import itertools
from fractions import Fraction
from typing import List

import z3

from conestrip.cones import Gamble
from conestrip.propositional_cones import PropositionalSentence, BooleanVariable


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
