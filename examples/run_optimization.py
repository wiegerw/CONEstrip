#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse

from conestrip.optimization import generate_mass_function, make_lower_prevision_function1, incurs_sure_loss, \
    is_mass_function
from conestrip.random_cones import random_gambles
from conestrip.utility import StopWatch


def info(args):
    print('--- Settings ---')
    print(f'gamble-size = {args.gamble_size}')
    print(f'k-size = {args.k_size}')
    print(f'coordinate-bound = {args.coordinate_bound}')
    print(f'count = {args.count}')
    print()


def run_testcase1(gamble_size: int, k_size: int, bound: int = 10):
    Omega = list(range(gamble_size))
    K = random_gambles(k_size, gamble_size, bound)
    p = generate_mass_function(Omega)
    assert is_mass_function(p)
    P_p = make_lower_prevision_function1(p, K)
    print('--- testcase 1 ---')
    print(f'K = {K}\np = {p}\nP_p = {P_p}')
    watch = StopWatch()
    assert(incurs_sure_loss(P_p, Omega))
    print(f'incurs_sure_loss(P_p, Omega): {watch.seconds():.4f}s\n')


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('--gamble-size', type=int, default=3, help='the number of elements of the gambles in the initial general cone')
    cmdline_parser.add_argument('--k-size', type=int, default=3, help='the number of elements of the set of gambles K')
    cmdline_parser.add_argument('--coordinate-bound', type=int, default=10, help='the maximum absolute value of the coordinates')
    cmdline_parser.add_argument('--count', type=int, default=1000, help='the number of times the experiment is repeated')
    args = cmdline_parser.parse_args()
    info(args)
    for _ in range(args.count):
        run_testcase1(args.gamble_size, args.k_size, args.coordinate_bound)


if __name__ == '__main__':
    main()
