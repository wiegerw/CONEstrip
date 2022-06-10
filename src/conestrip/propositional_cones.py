# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import itertools
import re
from fractions import Fraction
from typing import Dict, List, Tuple, Optional
import z3
from conestrip.cones import Gamble

PropositionalSentence = z3.ExprRef
PropositionalGamble = List[Fraction]
PropositionalConeGenerator = List[PropositionalGamble]
PropositionalGeneralCone = List[PropositionalConeGenerator]
PropositionalBasis = List[PropositionalSentence]
BooleanVariable = z3.BoolRef


def parse_propositional_gamble(text: str) -> PropositionalGamble:
    return [Fraction(s) for s in text.strip().split()]


def parse_propositional_cone_generator(text: str) -> PropositionalConeGenerator:
    return list(map(parse_propositional_gamble, text.strip().split('\n')))


def parse_propositional_general_cone(text: str) -> PropositionalGeneralCone:
    return list(map(parse_propositional_cone_generator, re.split(r'\n\s*\n', text.strip())))


def parse_boolean_gamble(text: str) -> PropositionalGamble:
    result = [Fraction(s) for s in text.strip().split()]
    assert all(f in [0,1] for f in result)
    return result


# def parse_propositional_basis(text: str) -> List[PropositionalSentence]:
#     gambles = list(map(parse_boolean_gamble, text.strip().split('\n')))
#     return gambles


# convert a sentence to the corresponding gamble
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