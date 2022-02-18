# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from fractions import Fraction
from unittest import TestCase
from conestrip.cones import parse_gamble, parse_cone_generator, parse_general_cone, print_gamble, GeneralCone, Gamble
from conestrip.random_cones import random_cone_generator, add_random_border_cones, random_border_point, random_inside_point, random_general_cone
from conestrip.utility import remove_spaces, random_nonzero_rationals_summing_to_one
from conestrip.conestrip import conestrip, conestrip1, conestrip2, conestrip3, is_in_general_cone, is_in_cone_generator, is_in_cone_generator_border, random_between_point


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
        Omega_Delta = [0, 1]

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

    def check_conestrip(self, R: GeneralCone, f_text: str, expected_result=False, verbose=True):
        f = parse_gamble(f_text)
        n = len(f)
        Omega_Gamma = list(range(n))
        Omega_Delta = list(range(n))
        result1 = conestrip1(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
        result2 = conestrip2(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
        result3 = conestrip3(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
        result4 = conestrip(R, f, Omega_Gamma, Omega_Delta, verbose=False)
        print('result1', result1)
        print('result2', result2)
        print('result3', result3)
        print('result4', result4)
        self.assertEqual(expected_result, result1 is not None)
        self.assertEqual(expected_result, result2 is not None)
        self.assertEqual(expected_result, result3 is not None)
        self.assertEqual(expected_result, result4 is not None)

    def test_conestrip2(self):
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


    def test_conestrip3(self):
        text = '''
          1 0
          0 1
          
          1 0
        '''
        R = parse_general_cone(text)
        self.check_conestrip(R, '1/2 1/3', True)
        self.check_conestrip(R, '2 4', True)
        self.check_conestrip(R, '1 0', True)
        self.check_conestrip(R, '2 0', True)
        self.check_conestrip(R, '0 1', False)
        self.check_conestrip(R, '0 2', False)

    def test_conestrip4(self):
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

    def test_conestrip5(self):
        text = '''
          4 0 0
          0 5 0
          0 0 6

          1 0 1
          0 7 7

          1 2 3
          2 4 6
        '''
        R = parse_general_cone(text)
        self.check_conestrip(R, '2 5 8', True)

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

    def test_random_border_point(self):
        for _ in range(10):
            dimension = 3
            generator_size = 3
            bound = 10
            normal = [Fraction(1), Fraction(1), Fraction(1)]
            R = random_cone_generator(dimension, generator_size, bound, normal)
            x, lambda_ = random_border_point(R)
            print('\nx =', print_gamble(x))
            self.assertTrue(is_in_cone_generator_border(R, x))

    def test_random_cone_generator(self):
        cone_size = 1
        dimension = 3
        generator_size = 3
        bound = 10
        R = random_general_cone(cone_size, dimension, generator_size, bound)
        add_random_border_cones(R, 2, False)

        for r in R.generators:
            if not r.parent:
                continue

            print('r =', r, '\n')
            x1, lambda1 = random_inside_point(r)
            x2, lambda2 = random_border_point(r)
            x3 = random_between_point(r)
            print('x1', print_gamble(x1))
            print('x2', print_gamble(x2))
            print('x3', print_gamble(x3))
            self.assertTrue(is_in_cone_generator(r, x1))
            self.assertTrue(is_in_cone_generator_border(r, x2))
            self.assertFalse(is_in_cone_generator(r, x3))
            self.assertFalse(is_in_cone_generator(r, x3, with_border=True))


if __name__ == '__main__':
    import unittest
    unittest.main()
