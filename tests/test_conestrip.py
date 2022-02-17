# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from conestrip.cones import parse_gamble, parse_cone_generator, parse_general_cone, print_gamble, GeneralCone
from conestrip.random_cones import random_cone_generator, add_random_border_cones, random_border_point, random_inside_point
from conestrip.utility import remove_spaces, random_nonzero_rationals_summing_to_one
from conestrip.conestrip import conestrip1, is_in_general_cone


class Test(TestCase):
    def test_gambles(self):
        g = parse_gamble('1/2 2 -3/4')
        self.assertEqual(g, [1/2, 2, -3/4])
        self.assertEqual(g, [0.5, 2, -0.75])
        txt = print_gamble(g)
        self.assertEqual('1/2 2 -3/4', txt)

    def test_generate(self):
        cone = random_cone_generator(4, 5, 20)
        self.assertEqual(len(cone.gambles), 5)
        for g in cone:
            self.assertEqual(len(g), 4)
            for gi in g:
                self.assertTrue(-20 <= gi <= 20)
        print(cone)

    def test_parse_print(self):
        text = '''
          4 0 0
          0 5 0
          0 0 6
    
          1 0 1
          0 7 7
    
          1 2 3
          2 4 6
        '''
        cone = parse_general_cone(text)
        self.assertEqual(remove_spaces(text), str(cone))

    def test_conestrip1(self):
        text = '''
          1 0
          0 1
        '''
        R = parse_general_cone(text)
        Omega_Gamma = [0, 1]
        Omega_Delta = []

        f = parse_gamble('1 0')
        result = conestrip1(R, f, Omega_Gamma, Omega_Delta)
        self.assertIsNone(result)

        f = parse_gamble('1/2 0')
        result = conestrip1(R, f, Omega_Gamma, Omega_Delta)
        self.assertIsNone(result)

        f = parse_gamble('0 1')
        result = conestrip1(R, f, Omega_Gamma, Omega_Delta)
        self.assertIsNone(result)

        f = parse_gamble('1 1')
        result = conestrip1(R, f, Omega_Gamma, Omega_Delta)
        self.assertIsNotNone(result)

        f = parse_gamble('0 0')
        result = conestrip1(R, f, Omega_Gamma, Omega_Delta)
        self.assertIsNone(result)

    def test_conestrip2(self):
        text = '''
          1 0
          0 1
          
          1 0
        '''
        R = parse_general_cone(text)
        Omega_Gamma = [0, 1]
        Omega_Delta = [0, 1]

        f = parse_gamble('1 0')
        result = conestrip1(R, f, Omega_Gamma, Omega_Delta)
        self.assertIsNotNone(result)

    def test_conestrip3(self):
        text = '''
          1 0
          0 1
        '''
        R = parse_general_cone(text)
        add_random_border_cones(R, 1)
        R1 = R.generators[1]
        Omega_Gamma = [0, 1]
        Omega_Delta = []
        for f in R1.vertices:
            result = conestrip1(R, f, Omega_Gamma, Omega_Delta)
            self.assertIsNotNone(result)

    def test_random_numbers(self):
        n = 10
        for _ in range(100):
            v = random_nonzero_rationals_summing_to_one(n)
            self.assertEqual(n, len(v))
            self.assertEqual(1, sum(v))

    def test_random_points(self):
        text = '''
           1  1  2
          -1  1  2
           1 -1  2
          -1 -1  2 
        '''
        R = parse_cone_generator(text)
        g1, lambda1 = random_border_point(R)
        print('g1', g1)
        g2, lambda2 = random_inside_point(R)
        self.assertFalse(is_in_general_cone(GeneralCone([R]), g1))
        self.assertTrue(is_in_general_cone(GeneralCone([R]), g2))

    def test_in_cone1(self):
        text = '''
           1  0
           0  1
        '''
        R = parse_cone_generator(text)
        r1 = parse_gamble('1 0')
        r2 = parse_gamble('0 1')
        self.assertIsNone(is_in_general_cone(GeneralCone([R]), r1))
        self.assertIsNone(is_in_general_cone(GeneralCone([R]), r2))

    def test_in_cone2(self):
        text = '''
           1  2
           2  1
        '''
        R = parse_cone_generator(text)
        r1 = parse_gamble('1 2')
        r2 = parse_gamble('2 1')
        self.assertIsNone(is_in_general_cone(GeneralCone([R]), r1))
        self.assertIsNone(is_in_general_cone(GeneralCone([R]), r2))


if __name__ == '__main__':
    import unittest
    unittest.main()
