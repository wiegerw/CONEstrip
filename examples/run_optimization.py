#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import random
from fractions import Fraction
from typing import List
import numpy as np
import xarray as xr

from conestrip.cones import print_gambles, print_fractions
from conestrip.global_settings import GlobalSettings
from conestrip.optimization import generate_mass_function, lower_prevision_function1, incurs_sure_loss, \
    is_mass_function, print_lower_prevision_function, generate_lower_prevision_perturbation, \
    lower_prevision_clamped_sum, lower_prevision_function2, is_coherent, generate_mass_functions
from conestrip.random_cones import random_gambles
from conestrip.utility import StopWatch


def info(args):
    print('--- Settings ---')
    print(f'seed = {args.seed}')
    print(f'omega-size = {args.omega_size}')
    print(f'k-size = {args.k_size}')
    print(f'coordinate-bound = {args.coordinate_bound}')
    print(f'error-magnitude = {args.error_magnitude}')
    print(f'repetitions = {args.repetitions}')
    print(f'verbose = {args.verbose}')
    print(f'pretty = {args.pretty}')
    print(f'print-smt = {args.print_smt}')
    print()


def make_default_epsilon_range() -> List[Fraction]:
    return [Fraction(f) for f in [0.000001, 0.00001, 0.0001, 0.001, 0.1, 1, 10]]


# Returns n values of the shape [..., 1/8, 1/4, 1/2, 1]
def make_epsilon_range(n: int) -> List[Fraction]:
    epsilon = Fraction(1)
    result = []
    for i in range(n):
        result.append(epsilon)
        epsilon /= 2
    return list(reversed(result))


def run_testcase1(args):
    GlobalSettings.verbose = args.verbose
    Omega = list(range(args.omega_size))
    K = random_gambles(args.k_size, args.omega_size, args.coordinate_bound)

    for _ in range(args.repetitions):
        p = generate_mass_function(Omega)
        error_magnitude = Fraction(args.error_magnitude)
        assert is_mass_function(p)
        if error_magnitude > 0:
            P_p = lower_prevision_function2(p, K, Fraction(error_magnitude))
        else:
            P_p = lower_prevision_function1(p, K)
        print('--- testcase 1 ---')
        print(f'K = {print_gambles(K, args.pretty)}\np = {print_fractions(p, args.pretty)}\nP_p = {print_lower_prevision_function(P_p, args.pretty)}')
        watch = StopWatch()
        result = incurs_sure_loss(P_p, Omega, args.pretty)
        print(f'incurs_sure_loss(P_p, Omega): {result} {watch.seconds():.4f}s\n')
        assert(not result)


def run_testcase2(args):
    GlobalSettings.verbose = args.verbose
    error_magnitude = Fraction(args.error_magnitude)
    Omega = list(range(args.omega_size))
    K = random_gambles(args.k_size, args.omega_size, args.coordinate_bound)

    for _ in range(args.repetitions):
        p = generate_mass_function(Omega)
        assert is_mass_function(p)
        if error_magnitude > 0:
            P_p = lower_prevision_function2(p, K, Fraction(error_magnitude))
        else:
            P_p = lower_prevision_function1(p, K)
        print('--- testcase 2 ---')
        print(f'K = {print_gambles(K, args.pretty)}\np = {print_fractions(p, args.pretty)}\nP_p = {print_lower_prevision_function(P_p, args.pretty)}\n')
        for epsilon in make_default_epsilon_range():
            Q = generate_lower_prevision_perturbation(K, Fraction(epsilon))
            P = lower_prevision_clamped_sum(P_p, Q)
            print(f'epsilon = {float(epsilon):7.4f}\nP = {print_lower_prevision_function(P, args.pretty)}')
            watch = StopWatch()
            result = incurs_sure_loss(P, Omega, args.pretty)
            print(f'incurs_sure_loss(P, Omega): {result} {watch.seconds():.4f}s\n')


def print_bool(x: bool) -> str:
    return 'T' if x else 'F'


def print_number_list(x: List[Fraction]) -> str:
    numbers = list(f'{xi:6.4f}' for xi in x)
    numbers = ', '.join(numbers)
    return f'[{numbers}]'


def run_testcase3(args):
    print('--- testcase 3 ---')
    GlobalSettings.verbose = args.verbose
    I, E, N = [int(s) for s in args.testcase3_dimensions.split(',')]
    V = 2  # the number of values per experiment
    Omega = list(range(args.omega_size))
    K = random_gambles(args.k_size, args.omega_size, args.coordinate_bound)

    p = generate_mass_functions(Omega)
    M = len(p)
    delta = [Fraction(i, I) for i in range(I)]  # the imprecision values
    epsilon = make_epsilon_range(E)  # the error magnitude values

    print(f'M, I, E, N = {M}, {I}, {E}, {N}')
    print(f'delta = {list(map(float, delta))}')
    print(f'epsilon = {list(map(float, epsilon))}')
    for m in range(M):
        print(f'probability mass function {m} = {list(map(float, p[m]))}')
    print('')

    pmf_coords = list(range(M))
    imprecision_coords = [float(d) for d in delta]  # N.B. list(map(float, delta)) doesn't work!
    errmag_coords = [float(e) for e in epsilon]
    repetitions_coords = list(range(N))
    values_coords = ['sureloss', 'coherence']
    gamble_coords = list(range(len(K)))
    outcomes_coords = list(range(len(Omega)))

    # create DataArray G containing the gambles in K
    G_data = np.empty((len(K), len(Omega)))
    G_dims = ['gambles', 'outcomes']
    G_coords = [gamble_coords, outcomes_coords]
    for i, f in enumerate(K):
        G_data[i] = [float(f_i) for f_i in f]
    G = xr.DataArray(G_data, G_coords, G_dims)

    # create DataArray A containing the probability mass functions
    A_data = np.empty((len(p), len(Omega)))
    A_dims = ['pmf', 'outcomes']
    A_coords = [pmf_coords, outcomes_coords]
    for i, p_i in enumerate(p):
        A_data[i] = [float(x) for x in p_i]
    A = xr.DataArray(A_data, A_coords, A_dims)

    # create DataArray Q containing the experimental results
    Q_data = np.empty((M, I, E, N, V), dtype=object)
    Q_dims = ['pmf', 'imprecision', 'errmag', 'repetitions', 'values']
    Q_coords = [pmf_coords, imprecision_coords, errmag_coords, repetitions_coords, values_coords]

    for m in range(M):
        for i in range(I):
            print(f'm, i = {m}, {i}')
            for e in range(E):
                for n in range(N):
                    P_delta = lower_prevision_function2(p[m], K, delta[i])
                    Q_epsilon = generate_lower_prevision_perturbation(K, epsilon[e])
                    P_m_i_e_n = lower_prevision_clamped_sum(P_delta, Q_epsilon)
                    sure_loss = incurs_sure_loss(P_m_i_e_n, Omega, args.pretty)
                    coherent = is_coherent(P_m_i_e_n, Omega, args.pretty)
                    Q_data[m, i, e, n] = [int(sure_loss), int(coherent)]

    Q = xr.DataArray(Q_data, Q_coords, Q_dims)

    Z_data_vars = {'Q': Q, 'A': A, 'G': G}
    Z_coords = {'pmf': pmf_coords, 'imprecision': imprecision_coords, 'errmag': errmag_coords, 'repetitions': repetitions_coords, 'values': values_coords, 'outcomes': outcomes_coords, 'gambles': gamble_coords}
    Z_dims = {'pmf': M, 'imprecision': I, 'errmag': E, 'repetitions': N, 'values': V, 'outcomes': len(Omega), 'gambles': len(K)}
    Z = xr.Dataset(Z_data_vars, Z_coords, Z_dims)

    print(f'saving data set to {args.output_filename}')
    Z.to_netcdf(args.output_filename)

def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--seed", help="the seed of the random generator", type=int, default=0)
    cmdline_parser.add_argument('--omega-size', type=int, default=3, help='the number of elements of event set Omega')
    cmdline_parser.add_argument('--k-size', type=int, default=3, help='the number of elements of the set of gambles K')
    cmdline_parser.add_argument('--coordinate-bound', type=int, default=1, help='the maximum absolute value of the coordinates')
    cmdline_parser.add_argument('--error-magnitude', type=str, default='0', help='the error magnitude value used for generating lower prevision functions')
    cmdline_parser.add_argument('--repetitions', type=int, default=1, help='the number of times an experiment is repeated')
    cmdline_parser.add_argument('--test', type=int, default=1, help='the test case (1 or 2)')
    cmdline_parser.add_argument('--pretty', help='print fractions as floats', action='store_true')
    cmdline_parser.add_argument('--print-smt', help='print info about the generated SMT problems', action='store_true')
    cmdline_parser.add_argument('--verbose', '-v', help='print verbose output', action='store_true')
    cmdline_parser.add_argument('--testcase3-dimensions', type=str, default='10,7,5', help='the dimensions I,E,N of test case 3 (a comma-separated list)')
    cmdline_parser.add_argument('--output-filename', type=str, help='a filename where output is stored')
    args = cmdline_parser.parse_args()
    if args.seed == 0:
        args.seed = random.randrange(0, 10000000000)
    info(args)

    if args.test == 1:
        run_testcase1(args)
    elif args.test == 2:
        run_testcase2(args)
    elif args.test == 3:
        run_testcase3(args)
    else:
        raise RuntimeError(f'Unknown test case {args.test}')


if __name__ == '__main__':
    main()
