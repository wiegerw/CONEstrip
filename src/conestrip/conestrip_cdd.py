# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)


import re
from collections import defaultdict
from typing import Any, List, Optional, Set, Tuple
from more_itertools import collapse
from more_itertools.recipes import flatten
from z3 import *
import cdd
from conestrip.cones import GeneralCone, Gamble, ConeGenerator, ConvexCombination, linear_combination
from conestrip.conestrip import conestrip_constraints, is_valid_conestrip_input


def conestrip_cdd_constraints(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], variables: Tuple[Any, Any, Any]):
    less_equal_constraints = []
    equal_constraints = []

    # variables
    lambda_, mu, sigma = variables

    n = len(f0)                      # the number of elements in a gamble
    n_lambda = len(lambda_)
    n_mu = len(list(flatten(mu)))
    n_sigma = 1
    N = n_lambda + n_mu + n_sigma    # the total number of variables

    # G = list(flatten(r.gambles for r in R0.generators))  # a flat list of all the gambles in R0
    # assert len(G) == n_mu
    G = [r.gambles for r in R0.generators]

    # Constraints are stored in the format [b -A] for constraints of the type Ax <= b
    # To make this easier we use [b A] for constraints of the type Ax + b >= 0

    # lambda_i >= 0
    for i in range(n):
        a = [Fraction(0)] * N
        a[i] = Fraction(1)
        b = Fraction(0)
        less_equal_constraints.append([b] + a)

    # lambda_i <= 1, hence -lambda_i + 1 >= 0
    for i in range(n):
        a = [Fraction(0)] * N
        a[i] = Fraction(-1)
        b = Fraction(1)
        less_equal_constraints.append([b] + a)

    # mu >= 0
    for i in range(n_mu):
        a = [Fraction(0)] * N
        a[n_lambda + i] = Fraction(1)
        b = Fraction(0)
        less_equal_constraints.append([b] + a)

    # sigma >= 1, hence sigma - 1 >= 0
    a = [Fraction(0)] * N
    a[-1] = Fraction(1)
    b = Fraction(-1)
    less_equal_constraints.append([b] + a)

    # sum(lambda_i) >= 1, hence sum(lambda_i) - 1 >= 0
    a = [Fraction(0)] * N
    for i in range(n_lambda):
        a[i] = Fraction(1)
    b = Fraction(-1)
    less_equal_constraints.append([b] + a)

    # lambda_d <= mu_d,g, hence -lambda_d + mu_d,g >= 0
    mu_index = n_lambda
    for d, G_d in enumerate(G):
        for G_dg in G_d:
            a = [Fraction(0)] * N
            a[d] = Fraction(-1)
            a[mu_index] = Fraction(1)
            mu_index += 1
            b = Fraction(0)
            less_equal_constraints.append([b] + a)

    # main constraints: sum d: mu_d g_d - f * sigma = 0
    for k in range(n):
        a = [Fraction(0)] * N
        index = n_lambda
        for G_d in G:
            for G_dg in G_d:
                a[index] = G_dg[k]
                index += 1
        a[-1] = -f0[k]
        b = Fraction(0)
        equal_constraints.append([b] + a)

    return less_equal_constraints, equal_constraints


def conestrip_cdd_solution(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    """
    An implementation of formula (4) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """
    assert is_valid_conestrip_input(R0, f0, Omega_Gamma, Omega_Delta)

    # variables
    lambda_variables = [f'lambda{d}' for d in range(len(R0))]
    mu_variables = [[f'mu{d}_{i}' for i in range(len(R0[d]))] for d in range(len(R0))]
    sigma = 'sigma'
    n_lambda = len(R0.generators)
    n_mu = [len(r.gambles) for r in R0.generators]

    if verbose:
        all_variables = lambda_variables + list(flatten(mu_variables)) + [sigma]
        print('variables:', ' '.join(all_variables))

    less_equal_constraints, equal_constraints = conestrip_cdd_constraints(R0, f0, Omega_Gamma, Omega_Delta, (lambda_variables, mu_variables, sigma))
    constraints = less_equal_constraints + equal_constraints
    n_less_equal = len(less_equal_constraints)
    n_equal = len(equal_constraints)
    mat = cdd.Matrix(constraints)
    mat.obj_type = cdd.LPObjType.MAX  # TODO: is it possible to just solve instead of minimize/maximize?
    mat.lin_set = { range(n_less_equal, n_less_equal + n_equal) }
    print(mat)

    lp = cdd.LinProg(mat)
    lp.solve()
    if lp.status == cdd.LPStatusType.OPTIMAL:
        x = lp.primal_solution
        lambda_ = x[:n_lambda]
        mu = []
        index = n_lambda
        for n in n_mu:
            mu.append(x[index:index+n])
            index = index + n
        sigma = x[index]
        return lambda_, mu, sigma


def conestrip_cdd(R: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    """
    An implementation of the CONEstrip algorithm in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    @param R:
    @param f0:
    @param Omega_Gamma:
    @param Omega_Delta:
    @return: A solution (lambda, mu, sigma) to the CONEstrip optimization problem (4), or None if no solution exists
    """

    while True:
        Lambda = conestrip_cdd_solution(R, f0, Omega_Gamma, Omega_Delta, verbose)
        if not Lambda:
            return None
        lambda_, mu, sigma = Lambda
        Q = [d for d, lambda_d in enumerate(lambda_) if lambda_d == 0]
        if all(x == 0 for x in collapse(mu[d] for d in Q)):
            return Lambda
        R = GeneralCone([R_d for d, R_d in enumerate(R) if d not in Q])
