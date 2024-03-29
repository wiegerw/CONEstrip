#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
from conestrip.cones import print_gamble, linear_combination
from conestrip.conestrip_z3 import is_in_general_cone, solve_conestrip1, solve_conestrip2, solve_conestrip3, \
    conestrip_algorithm
from conestrip.conestrip_cdd import conestrip_cdd_algorithm
from conestrip.extended_cones import is_in_cone_generator_extended, is_in_general_cone_extended, \
    is_in_closed_cone_generator_extended, is_in_cone_generator_border_extended
from conestrip.gamble_algorithms import simplified_linear_combination
from conestrip.random_extended_cones import random_general_cone_extended, add_random_border_cones_extended, \
    random_inside_point_extended, random_border_point_extended, random_between_point_extended
from conestrip.utility import StopWatch, is_power_of_two
from conestrip.propositional_algorithms import propositional_conestrip_solution, is_in_propositional_cone_generator


def generate_cones(cone_size, generator_size, gamble_size, coordinate_bound, border_count):
    R = random_general_cone_extended(cone_size, gamble_size, generator_size, coordinate_bound)
    add_random_border_cones_extended(R, border_count, False)

    print('--- Generated cone ---')
    print(R)
    print()

    for index, r in enumerate(R.generators):
        if not r.parent:
            continue
        r_parent, _ = r.parent

        print('====================================================================================')
        print(f'Testing cone generator {index}\n')

        print('r_parent =\n', r_parent, '\n')
        print('r =\n', r, '\n')
        x1, lambda1 = random_inside_point_extended(r)
        x2, lambda2 = random_border_point_extended(r)
        x3, lambda3 = random_between_point_extended(r)
        print('x1 =', print_gamble(x1), 'lambda1 =', print_gamble(lambda1))
        print('x2 =', print_gamble(x2), 'lambda2 =', print_gamble(lambda2))
        print('x3 =', print_gamble(x3), 'lambda3 =', print_gamble(lambda3))
        print()
        assert x1 == linear_combination(lambda1, r.vertices)
        assert x2 == linear_combination(lambda2, r.vertices)
        assert x3 == simplified_linear_combination(lambda3, r_parent.vertices)

        watch = StopWatch()
        assert is_in_cone_generator_extended(r, x1)
        print(f'is_in_cone_generator(r, x1): {watch.seconds():.4f}s')

        watch.restart()
        assert is_in_cone_generator_border_extended(r, x2)
        print(f'is_in_cone_generator_border(r, x2): {watch.seconds():.4f}s')

        watch.restart()
        assert not is_in_cone_generator_extended(r, x3)
        print(f'not is_in_cone_generator(r, x3): {watch.seconds():.4f}s')

        watch.restart()
        assert not is_in_closed_cone_generator_extended(r, x3)
        print(f'not is_in_closed_cone_generator(r, x3): {watch.seconds():.4f}s')

        watch.restart()
        assert is_in_cone_generator_extended(r_parent, x3)
        print(f'is_in_cone_generator(r_parent, x3): {watch.seconds():.4f}s')

        watch.restart()
        assert is_in_general_cone_extended(R, x1, solver=solve_conestrip1)
        print(f'is_in_general_cone(R, x1, solver=solve_conestrip1): {watch.seconds():.4f}s')

        watch.restart()
        assert is_in_general_cone_extended(R, x1, solver=solve_conestrip2)
        print(f'is_in_general_cone(R, x1, solver=solve_conestrip2): {watch.seconds():.4f}s')

        watch.restart()
        assert is_in_general_cone_extended(R, x1, solver=solve_conestrip3)
        print(f'is_in_general_cone(R, x1, solver=solve_conestrip1): {watch.seconds():.4f}s')

        watch.restart()
        assert is_in_general_cone_extended(R, x1, solver=conestrip_algorithm)
        print(f'is_in_general_cone(R, x1, solver=conestrip_algorithm): {watch.seconds():.4f}s')

        watch.restart()
        assert is_in_general_cone_extended(R, x1, solver=conestrip_cdd_algorithm)
        print(f'is_in_general_cone(R, x1, solver=conestrip_cdd_algorithm): {watch.seconds():.4f}s')

        if is_power_of_two(gamble_size):
            watch.restart()
            assert is_in_propositional_cone_generator(r.to_cone_generator(), x1)
            print(f'is_in_propositional_cone_generator(r, x1): {watch.seconds():.4f}s')

            watch.restart()
            assert not is_in_propositional_cone_generator(r.to_cone_generator(), x3)
            print(f'not is_in_propositional_cone_generator(r, x3): {watch.seconds():.4f}s')

            watch.restart()
            assert is_in_propositional_cone_generator(r_parent.to_cone_generator(), x3)
            print(f'is_in_propositional_cone_generator(r_parent, x3): {watch.seconds():.4f}s')

            watch.restart()
            assert is_in_general_cone(R.to_general_cone(), x1, solver=propositional_conestrip_solution)
            print(f'is_in_general_cone(R, x1, solver=propositional_conestrip_solution): {watch.seconds():.4f}s')
        print()


def info(args):
    print('--- Settings ---')
    print(f'cone-size = {args.cone_size}')
    print(f'generator-size = {args.generator_size}')
    print(f'gamble-size = {args.gamble_size}')
    print(f'coordinate-bound = {args.coordinate_bound}')
    print(f'border-count = {args.border_count}')
    print(f'count = {args.count}')
    print()


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('--cone-size', type=int, default=1, help='the number of cone generators in the initial general cone')
    cmdline_parser.add_argument('--generator-size', type=int, default=3, help='the number of gambles in the cone generators of the initial general cone')
    cmdline_parser.add_argument('--gamble-size', type=int, default=3, help='the number of elements of the gambles in the initial general cone')
    cmdline_parser.add_argument('--coordinate-bound', type=int, default=10, help='the maximum absolute value of the coordinates')
    cmdline_parser.add_argument('--border-count', type=int, default=10, help='the number of border facets that is added to the initial general cone')
    cmdline_parser.add_argument('--count', type=int, default=1000, help='the number of times the experiment is repeated')
    args = cmdline_parser.parse_args()
    for _ in range(args.count):
        info(args)
        generate_cones(args.cone_size, args.generator_size, args.gamble_size, args.coordinate_bound, args.border_count)


if __name__ == '__main__':
    main()
