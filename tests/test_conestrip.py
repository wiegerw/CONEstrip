#!/usr/bin/env python3

# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from fractions import Fraction
from unittest import TestCase
from conestrip.cones import parse_gamble, parse_cone_generator, parse_general_cone, print_gamble, GeneralCone
from conestrip.conestrip_cdd import conestrip_cdd_algorithm
from conestrip.extended_cones import parse_extended_cone_generator, parse_extended_general_cone
from conestrip.random_cones import random_cone_generator
from conestrip.random_extended_cones import random_border_point_extended, random_inside_point_extended, \
    random_cone_generator_extended, random_general_cone_extended, add_random_border_cones_extended, \
    random_between_point_extended
from conestrip.utility import remove_spaces, random_nonzero_rationals_summing_to_one
from conestrip.conestrip_z3 import conestrip_algorithm, solve_conestrip1, solve_conestrip2, solve_conestrip3, \
    is_in_general_cone, is_in_cone_generator, is_in_cone_generator_border, \
    is_in_closed_cone_generator, is_solved


class Test(TestCase):
    def test_gambles(self):
        g = parse_gamble('1/2 2 -3/4')
        self.assertEqual(g, [1/2, 2, -3/4])
        self.assertEqual(g, [0.5, 2, -0.75])
        txt = print_gamble(g)
        self.assertEqual('1/2 2 -3/4', txt)

    def test_generate(self):
        cone = random_cone_generator(4, 5, 20)
        self.assertEqual(len(cone), 5)
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
        cone = parse_extended_general_cone(text)
        self.assertEqual(remove_spaces(text), str(cone))

    def check_conestrip(self, R: GeneralCone, f_text: str, expected_result: bool = False, verbose=True):
        f = parse_gamble(f_text)
        n = len(f)
        Omega_Gamma = list(range(n))
        Omega_Delta = list(range(n))
        result1 = solve_conestrip1(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
        result2 = solve_conestrip2(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
        result3 = solve_conestrip3(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
        result4 = conestrip_algorithm(R, f, Omega_Gamma, Omega_Delta, verbose=False)
        result5 = conestrip_cdd_algorithm(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
        print('result1', result1)
        print('result2', result2)
        print('result3', result3)
        print('result4', result4)
        print('result5', result5)
        self.assertEqual(expected_result, is_solved(result1))
        self.assertEqual(expected_result, is_solved(result2))
        self.assertEqual(expected_result, is_solved(result3))
        self.assertEqual(expected_result, is_solved(result4))
        self.assertEqual(expected_result, is_solved(result5))

    def test_conestrip1(self):
        text = '''
          1 0
          0 1
        '''
        R = parse_general_cone(text)

        self.check_conestrip(R, '1 0', False)
        self.check_conestrip(R, '1/2 0', False)
        self.check_conestrip(R, '0 1', False)
        self.check_conestrip(R, '1 1', True)
        self.check_conestrip(R, '0 0', False)

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
        R = parse_extended_cone_generator(text)
        g1, lambda1 = random_border_point_extended(R)
        print('g1', g1)
        g2, lambda2 = random_inside_point_extended(R)
        self.assertFalse(is_in_general_cone([R.to_cone_generator()], g1))
        self.assertTrue(is_in_general_cone([R.to_cone_generator()], g2))

    def test_in_cone1(self):
        text = '''
           1  0
           0  1
        '''
        R = parse_cone_generator(text)
        r1 = parse_gamble('1 0')
        r2 = parse_gamble('0 1')
        self.assertFalse(is_in_general_cone([R], r1))
        self.assertFalse(is_in_general_cone([R], r2))

    def test_in_cone2(self):
        text = '''
           1  2
           2  1
        '''
        R = parse_cone_generator(text)
        r1 = parse_gamble('1 2')
        r2 = parse_gamble('2 1')
        self.assertFalse(is_in_general_cone([R], r1))
        self.assertFalse(is_in_general_cone([R], r2))

    def test_random_border_point(self):
        for _ in range(10):
            dimension = 3
            generator_size = 3
            bound = 10
            normal = [Fraction(1), Fraction(1), Fraction(1)]
            R = random_cone_generator_extended(dimension, generator_size, bound, normal)
            x, lambda_ = random_border_point_extended(R)
            print('\nx =', print_gamble(x))
            self.assertTrue(is_in_cone_generator_border(R.to_cone_generator(), x))

    def test_random_cone_generator(self):
        cone_size = 1
        dimension = 3
        generator_size = 3
        bound = 10
        R = random_general_cone_extended(cone_size, dimension, generator_size, bound)
        add_random_border_cones_extended(R, 2, False)

        for r in R.generators:
            if not r.parent:
                continue
            r_parent, _ = r.parent

            print('r_parent =\n', r_parent, '\n')
            print('r =\n', r, '\n')
            x1, lambda1 = random_inside_point_extended(r)
            x2, lambda2 = random_border_point_extended(r)
            x3, lambda3 = random_between_point_extended(r)
            print('x1 =', print_gamble(x1), 'lambda =', print_gamble(lambda1))
            print('x2 =', print_gamble(x2), 'lambda =', print_gamble(lambda2))
            print('x3 =', print_gamble(x3), 'lambda =', print_gamble(lambda3))
            self.assertTrue(is_in_cone_generator(r.to_cone_generator(), x1))
            self.assertTrue(is_in_cone_generator_border(r.to_cone_generator(), x2))
            self.assertFalse(is_in_cone_generator(r.to_cone_generator(), x3))
            self.assertFalse(is_in_closed_cone_generator(r.to_cone_generator(), x3))
            self.assertTrue(is_in_cone_generator(r_parent.to_cone_generator(), x3))
            self.assertTrue(is_in_general_cone(R.to_general_cone(), x1))
            print()


if __name__ == '__main__':
    import unittest
    unittest.main()
