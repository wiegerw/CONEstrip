# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)


import re
from collections import defaultdict
from typing import Any, List, Optional, Set, Tuple
from more_itertools.recipes import flatten
from z3 import *
import cdd
from conestrip.cones import GeneralCone, Gamble, ConeGenerator, ConvexCombination, linear_combination
from conestrip.conestrip import conestrip_constraints, is_valid_conestrip_input


def conestrip_cdd_constraints(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], variables: Tuple[Any, Any, Any]):
    less_equal_constraints = []
    equality_constraints = []

    # variables
    lambda_, mu, sigma = variables

    n = len(f0)                      # the number of elements in a gamble
    n_lambda = len(lambda_)
    n_mu = len(mu)
    N = len(lambda_ + mu + [sigma])  # the total number of variables

    G = list(flatten(r.gambles for r in R0.generators))  # a flat list of all the gambles in R0
    assert len(G) == len(list(flatten(mu)))

    # Constraints are stored in the format [b -A]

    # -lambda <= 0
    for i in range(n):
        constraint = [Fraction(0)] * N
        constraint[i] = Fraction(1)
        less_equal_constraints.append([Fraction(0)] + constraint)

    # lambda <= 1
    for i in range(n):
        constraint = [Fraction(0)] * N
        constraint[i] = Fraction(-1)
        less_equal_constraints.append([Fraction(1)] + constraint)

    # -mu <= 0
    for i in range(n_mu):
        constraint = [Fraction(0)] * N
        constraint[n_lambda + i] = Fraction(1)
        less_equal_constraints.append([Fraction(0)] + constraint)

    # -sigma <= 1
    constraint = [Fraction(0)] * N
    constraint[n_lambda + n_mu] = Fraction(1)
    less_equal_constraints.append([Fraction(1)] + constraint)

    # main constraints
    for j in range(n):
        constraint = [Fraction(0)] * N
        for i in range(n_mu):
            constraint[n_lambda + i] = G[i][j]
        equality_constraints.append([f0[j]] + constraint)

    # object function
    object_function = [Fraction(0)] * N
    for i in range(n_lambda):
        object_function[i] = Fraction(1)
    object_function = [Fraction(0)] + object_function

    return less_equal_constraints, equality_constraints, object_function


def conestrip_cdd_solution(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    """
    An implementation of formula (4) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """
    assert is_valid_conestrip_input(R0, f0, Omega_Gamma, Omega_Delta)

    # variables
    lambda_ = [f'lambda{d}' for d in range(len(R0))]
    mu = [[f'mu{d}_{i}' for i in range(len(R0[d]))] for d in range(len(R0))]
    sigma = 'sigma'

    less_equal_constraints, equality_constraints, object_function = conestrip_cdd_constraints(R0, f0, Omega_Gamma, Omega_Delta, (lambda_, mu, sigma))
    constraints = less_equal_constraints + equality_constraints
    mat = cdd.Matrix(constraints)
    mat.obj_type = cdd.LPObjType.MAX
    mat.obj_func = object_function
    print(mat)

    # # expressions
    # goal = simplify(sum(lambda_))

    variables = lambda_ + list(flatten(mu)) + [sigma]
    variables = [str(x) for x in variables]
    print('variables:', variables)

    # for c in constraints:
    #     parse_equation(c)

    # optimizer = Optimize()
    # optimizer.add(constraints)
    # optimizer.maximize(goal)
    # if optimizer.check() == sat:
    #     model = optimizer.model()
    #     lambda_solution = [model.evaluate(lambda_[d]) for d in range(len(R0))]
    #     mu_solution = [[model.evaluate(mu[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))]
    #     sigma_solution = model.evaluate(sigma)
    #     if verbose:
    #         print('--- solution ---')
    #         print('lambda =', lambda_solution)
    #         print('mu =', mu_solution)
    #         print('sigma =', sigma_solution)
    #         print('goal =', model.evaluate(goal))
    #     return lambda_solution, mu_solution, sigma_solution
    # else:
    #     return None


