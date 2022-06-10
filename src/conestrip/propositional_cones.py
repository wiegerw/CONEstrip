# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from fractions import Fraction
from typing import List
import z3

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
