#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from conestrip.cones import parse_gamble, parse_general_cone, print_gambles, parse_cone_generator
from conestrip.conestrip_z3 import is_in_cone_generator
from conestrip.propositional_conestrip import propositional_conestrip_algorithm
from conestrip.propositional_algorithms import gamble_to_sentence, sentence_to_gamble, \
    default_propositional_basis, convert_general_cone, convert_gamble, is_in_propositional_cone_generator


class Test(TestCase):
    def test_propositional_conestrip(self):
        R = parse_general_cone('''
          1 0 0 0
          0 1 1 0
          1 0 1 1

          0 1 0 1
          1 0 1 0
        ''')

        f = parse_gamble('1 1 1 0')

        Phi, B = default_propositional_basis(2)
        psi = True
        psi_Gamma = True
        psi_Delta = True
        Phi_gambles = [sentence_to_gamble(phi, B) for phi in Phi]

        R1 = convert_general_cone(R, Phi_gambles)
        f1 = convert_gamble(f, Phi_gambles)

        self.assertTrue(f == f1)
        solution = propositional_conestrip_algorithm(R1, f1, B, Phi, psi, psi_Gamma, psi_Delta)
        self.assertIsNotNone(solution)

        lambda_, mu, sigma, kappa = solution
        print('lambda = ', lambda_)
        print('mu =', mu)
        print('sigma =', sigma)
        print('kappa =', kappa)

    def test1(self):
        r = parse_cone_generator('-5 9')
        x3 = parse_gamble('-2 8')
        self.assertFalse(is_in_cone_generator(r, x3))
        self.assertFalse(is_in_propositional_cone_generator(r, x3, verbose=True))

    def test2(self):
        r = parse_cone_generator('3 -4')
        x3 = parse_gamble('13 -14')
        self.assertFalse(is_in_cone_generator(r, x3))
        self.assertFalse(is_in_propositional_cone_generator(r, x3, verbose=True))


if __name__ == '__main__':
    import unittest
    unittest.main()
