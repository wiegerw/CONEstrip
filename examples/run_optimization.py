#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import random
from fractions import Fraction

from conestrip.cones import print_gambles, print_fractions
from conestrip.global_settings import GlobalSettings
from conestrip.optimization import generate_mass_function, make_lower_prevision_function1, incurs_sure_loss, \
    is_mass_function, print_lower_prevision_function, make_perturbation, lower_prevision_sum, \
    lower_prevision_clamped_sum, make_lower_prevision_function2
from conestrip.random_cones import random_gambles
from conestrip.utility import StopWatch


def info(args):
    print('--- Settings ---')
    print(f'seed = {args.seed}')
    print(f'gamble-size = {args.gamble_size}')
    print(f'k-size = {args.k_size}')
    print(f'coordinate-bound = {args.coordinate_bound}')
    print(f'epsilon-lower-prevision = {args.epsilon_lower_prevision}')
    print(f'count = {args.count}')
    print(f'verbose = {args.verbose}')
    print(f'pretty = {args.pretty}')
    print(f'print-smt = {args.print_smt}')
    print()


def run_testcase1(args):
    GlobalSettings.verbose = args.verbose
    Omega = list(range(args.gamble_size))
    K = random_gambles(args.k_size, args.gamble_size, args.coordinate_bound)
    p = generate_mass_function(Omega)
    assert is_mass_function(p)
    if args.epsilon_lower_prevision > 0:
        P_p = make_lower_prevision_function2(p, K, Fraction(args.epsilon_lower_prevision))
    else:
        P_p = make_lower_prevision_function1(p, K)
    print('--- testcase 1 ---')
    print(f'K = {print_gambles(K, args.pretty)}\np = {print_fractions(p, args.pretty)}\nP_p = {print_lower_prevision_function(P_p, args.pretty)}')
    watch = StopWatch()
    result = incurs_sure_loss(P_p, Omega, args.pretty)
    print(f'incurs_sure_loss(P_p, Omega): {result} {watch.seconds():.4f}s\n')
    assert(not result)


def run_testcase2(args):
    GlobalSettings.verbose = args.verbose
    Omega = list(range(args.gamble_size))
    K = random_gambles(args.k_size, args.gamble_size, args.coordinate_bound)
    p = generate_mass_function(Omega)
    assert is_mass_function(p)
    if args.epsilon_lower_prevision > 0:
        P_p = make_lower_prevision_function2(p, K, Fraction(args.epsilon_lower_prevision))
    else:
        P_p = make_lower_prevision_function1(p, K)
    print('--- testcase 2 ---')
    print(f'K = {print_gambles(K, args.pretty)}\np = {print_fractions(p, args.pretty)}\nP_p = {print_lower_prevision_function(P_p, args.pretty)}\n')
    for epsilon in [0.000001, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
        Q = make_perturbation(K, Fraction(epsilon))
        P = lower_prevision_clamped_sum(P_p, Q)
        print(f'epsilon = {epsilon}\nP = {print_lower_prevision_function(P, args.pretty)}')
        watch = StopWatch()
        result = incurs_sure_loss(P, Omega, args.pretty)
        print(f'incurs_sure_loss(P, Omega): {result} {watch.seconds():.4f}s\n')


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--seed", help="the seed of the random generator", type=int, default=0)
    cmdline_parser.add_argument('--gamble-size', type=int, default=3, help='the number of elements of the gambles in the initial general cone')
    cmdline_parser.add_argument('--k-size', type=int, default=3, help='the number of elements of the set of gambles K')
    cmdline_parser.add_argument('--coordinate-bound', type=int, default=10, help='the maximum absolute value of the coordinates')
    cmdline_parser.add_argument('--epsilon-lower-prevision', type=float, default=0, help='the epsilon value used for generating lower prevision functions')
    cmdline_parser.add_argument('--count', type=int, default=1000, help='the number of times the experiment is repeated')
    cmdline_parser.add_argument('--test', type=int, default=1, help='the test case (1 or 2)')
    cmdline_parser.add_argument('--pretty', help='print fractions as floats', action='store_true')
    cmdline_parser.add_argument('--print-smt', help='print info about the generated SMT problems', action='store_true')
    cmdline_parser.add_argument('--verbose', '-v', help='print verbose output', action='store_true')
    args = cmdline_parser.parse_args()
    if args.seed == 0:
        args.seed = random.randrange(0, 10000000000)
    info(args)
    for _ in range(args.count):
        if args.test == 1:
            run_testcase1(args)
        elif args.test == 2:
            run_testcase2(args)
        else:
            raise RuntimeError(f'Unknown test case {args.test}')


if __name__ == '__main__':
    main()
