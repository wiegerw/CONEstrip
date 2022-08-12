#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import itertools
from fractions import Fraction
from unittest import TestCase
import z3
from conestrip.cones import parse_gamble, parse_cone_generator
from conestrip.gamble_algorithms import gamble_coefficients, is_convex_combination, is_positive_combination
from conestrip.propositional_sentence_parser import parse_propositional_sentence
from conestrip.propositional_algorithms import gamble_to_sentence, sentence_to_gamble, default_basis, \
    default_propositional_basis


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

    def test_default_basis(self):
        B = default_basis(3)
        self.assertEqual(len(B), 3)

    def test_default_propositional_basis(self):
        Phi, B = default_propositional_basis(3)
        self.assertEqual(len(B), 3)
        self.assertEqual(len(Phi), 2**3)

    def test_is_convex_combination(self):
        f1 = parse_gamble('2 1')
        f2 = parse_gamble('2 2')
        R = parse_cone_generator('''
            0 2
            4 0
        ''')
        self.assertIsNotNone(is_convex_combination(f1, R))
        self.assertIsNone(is_convex_combination(f2, R))
        self.assertIsNotNone(is_positive_combination(f2, R))


if __name__ == '__main__':
    import unittest
    unittest.main()
