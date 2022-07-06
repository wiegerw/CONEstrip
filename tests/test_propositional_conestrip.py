#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import itertools
from fractions import Fraction
from unittest import TestCase
import z3
from conestrip.cones import parse_gamble, parse_general_cone, print_gambles
from conestrip.algorithms import gamble_coefficients
from conestrip.propositional_conestrip import propositional_conestrip_algorithm
from conestrip.propositional_sentence_parser import parse_propositional_sentence
from conestrip.propositional_algorithms import gamble_to_sentence, sentence_to_gamble, default_basis, \
    default_propositional_basis, convert_general_cone, convert_gamble


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


if __name__ == '__main__':
    import unittest
    unittest.main()
