# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from fractions import Fraction
from unittest import TestCase
from conestrip.cones import parse_gamble, parse_cone_generator, parse_general_cone, print_gamble, GeneralCone, Gamble
from conestrip.conestrip_cdd import conestrip_cdd_algorithm
from conestrip.random_cones import random_cone_generator, add_random_border_cones, random_border_point, random_inside_point, random_general_cone
from conestrip.utility import remove_spaces, random_nonzero_rationals_summing_to_one
from conestrip.conestrip import conestrip_algorithm, conestrip1_solution, conestrip2_solution, conestrip3_solution, is_in_general_cone, is_in_cone_generator, is_in_cone_generator_border, random_between_point
from conestrip.propositional_conestrip import parse_propositional_basis, gamble_coefficients


class Test(TestCase):
    def test_gamble_coefficients(self):
        g = parse_gamble('2 3 4 5')
        Phi = parse_propositional_basis(
            '''
              1 0 1 2
              1 1 0 3
              0 2 3 0
            ''')
        c = gamble_coefficients(g, Phi)
        self.assertEqual([1, 1, 1], c)


if __name__ == '__main__':
    import unittest
    unittest.main()
