#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from fractions import Fraction
from unittest import TestCase
from conestrip.cones import parse_gamble, parse_cone_generator, parse_general_cone, print_gamble, GeneralCone
from conestrip.conestrip_cdd import conestrip_cdd_algorithm
from conestrip.extended_cones import parse_extended_cone_generator, parse_extended_general_cone
from conestrip.optimization import generate_mass_function, lower_prevision_set_A, incurs_sure_loss
from conestrip.random_cones import random_cone_generator, random_gambles
from conestrip.random_extended_cones import random_border_point_extended, random_inside_point_extended, \
    random_cone_generator_extended, random_general_cone_extended, add_random_border_cones_extended, \
    random_between_point_extended
from conestrip.utility import remove_spaces, random_nonzero_rationals_summing_to_one
from conestrip.conestrip_z3 import conestrip_algorithm, solve_conestrip1, solve_conestrip2, solve_conestrip3, \
    is_in_general_cone, is_in_cone_generator, is_in_cone_generator_border, \
    is_in_closed_cone_generator, is_solved


class Test(TestCase):
    def test1(self):
        N = 4   # the number of events
        Omega = list(range(N))
        bound = 10
        n = 3  # the size of the gambles
        count = 5
        K = random_gambles(count, n, bound)
        p = generate_mass_function(Omega)
        A = lower_prevision_set_A(p, K, Omega)
        result = incurs_sure_loss(A, Omega)
        self.assertEqual(1, 2)


if __name__ == '__main__':
    import unittest
    unittest.main()
