# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
from conestrip.cones import print_gamble, linear_combination
from conestrip.random_cones import add_random_border_cones, random_border_point, random_inside_point, random_general_cone
from conestrip.conestrip import is_in_general_cone, is_in_cone_generator, is_in_cone_generator_border, random_between_point, simplified_linear_combination


def generate_cones(cone_size, generator_size, gamble_size, coordinate_bound):
    R = random_general_cone(cone_size, gamble_size, generator_size, coordinate_bound)
    add_random_border_cones(R, 2, False)

    for r in R.generators:
        if not r.parent:
            continue
        r_parent, _ = r.parent

        print('====================================================================================')
        print('r_parent =\n', r_parent, '\n')
        print('r =\n', r, '\n')
        x1, lambda1 = random_inside_point(r)
        x2, lambda2 = random_border_point(r)
        x3, lambda3 = random_between_point(r)
        print('x1 =', print_gamble(x1), 'lambda1 =', print_gamble(lambda1))
        print('x2 =', print_gamble(x2), 'lambda2 =', print_gamble(lambda2))
        print('x3 =', print_gamble(x3), 'lambda3 =', print_gamble(lambda3))
        assert x1 == linear_combination(lambda1, r.vertices)
        assert x2 == linear_combination(lambda2, r.vertices)
        assert x3 == simplified_linear_combination(lambda3, r_parent.vertices)
        assert is_in_cone_generator(r, x1)
        assert is_in_cone_generator_border(r, x2)
        assert not is_in_cone_generator(r, x3)
        assert not is_in_cone_generator(r, x3, with_border=True)
        assert is_in_cone_generator(r_parent, x3)
        assert is_in_general_cone(R, x1)
        print()


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('--cone-size', type=int, default=1, help='the number of cone generators in the initial general cone')
    cmdline_parser.add_argument('--generator-size', type=int, default=3, help='the number of gambles in the cone generators of the initial general cone')
    cmdline_parser.add_argument('--gamble-size', type=int, default=3, help='the number of elements of the gambles in the initial general cone')
    cmdline_parser.add_argument('--coordinate-bound', type=int, default=10, help='the maximum absolute value of the coordinates')
    cmdline_parser.add_argument('--count', type=int, default=1000, help='the number of times the experiment is repeated')
    args = cmdline_parser.parse_args()
    for _ in range(args.count):
        generate_cones(args.cone_size, args.generator_size, args.gamble_size, args.coordinate_bound)


if __name__ == '__main__':
    main()
