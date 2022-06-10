# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import itertools
from fractions import Fraction
from unittest import TestCase
import z3
from conestrip.cones import parse_gamble
from conestrip.algorithms import gamble_coefficients
from conestrip.propositional_sentence_parser import parse_propositional_sentence
from conestrip.propositional_cones import gamble_to_sentence, sentence_to_gamble


class Test(TestCase):
    def test_gamble_coefficients(self):
        g = parse_gamble('2 3 3 2')
        Phi = [parse_gamble('1 0 1 1'),
               parse_gamble('1 1 0 1'),
               parse_gamble('0 1 1 0')
              ]
        c = gamble_coefficients(g, Phi)
        self.assertEqual([1, 1, 2], c)

    def test_sentence_to_gamble(self):
        B = z3.Bools('b1 b2')
        phi = parse_propositional_sentence('Or(b1, Not(b2))')
        g = sentence_to_gamble(phi, B)
        self.assertEqual(g, [1, 0, 1, 1])

    def test_gamble_to_sentence(self):
        B = z3.Bools('b1 b2 b3')
        m = len(B)
        for g in itertools.product([Fraction(0), Fraction(1)], repeat=2**m):
            phi = gamble_to_sentence(g, B)
            g1 = sentence_to_gamble(phi, B)
            self.assertTrue(list(g) == g1)


if __name__ == '__main__':
    import unittest
    unittest.main()
