#!/usr/bin/env python3

# Copyright 2021 - 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from conestrip.cones import parse_general_cone, parse_gamble, GeneralCone
from conestrip.conestrip_z3 import solve_conestrip1, solve_conestrip2, solve_conestrip3, is_solved
from conestrip.conestrip_cdd import solve_conestrip_cdd


class Test(TestCase):
    def check_conestrip(self, R: GeneralCone, f_text: str, expected_result):
        f = parse_gamble(f_text)
        n = len(f)
        Omega_Gamma = list(range(n))
        Omega_Delta = list(range(n))
        result1 = solve_conestrip1(R, f, Omega_Gamma, Omega_Delta)
        result2 = solve_conestrip2(R, f, Omega_Gamma, Omega_Delta)
        result3 = solve_conestrip3(R, f, Omega_Gamma, Omega_Delta)
        result4 = solve_conestrip_cdd(R, f, Omega_Gamma, Omega_Delta)
        result5 = solve_conestrip_cdd(R, f, Omega_Gamma, Omega_Delta)
        results = list(map(is_solved, [result1, result2, result3, result4, result5]))
        if expected_result == True:
            self.assertTrue(all(results))
        else:
            self.assertFalse(any(results))

    def test_cdd1(self):
        text = '''
          1 2
          2 1

          1 2
        '''
        R = parse_general_cone(text)
        self.check_conestrip(R, '1 1', True)
        self.check_conestrip(R, '10 10', True)
        self.check_conestrip(R, '1 2', True)
        self.check_conestrip(R, '2 4', True)
        self.check_conestrip(R, '2 1', False)
        self.check_conestrip(R, '4 2', False)
        self.check_conestrip(R, '1 3', False)
        self.check_conestrip(R, '3 1', False)

    def test_cdd2(self):
        text = '''
          1 0
          0 1
        '''
        R = parse_general_cone(text)
        self.check_conestrip(R, '1/2 1/3', True)
        self.check_conestrip(R, '2 4', True)
        self.check_conestrip(R, '1 0', False)
        self.check_conestrip(R, '2 0', False)
        self.check_conestrip(R, '0 1', False)
        self.check_conestrip(R, '0 2', False)


if __name__ == '__main__':
    import unittest
    unittest.main()
