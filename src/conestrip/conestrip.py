# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, Optional, Tuple
from more_itertools import collapse
from more_itertools.recipes import flatten
from z3 import *
from gambles import *
from utility import product, sum_rows


def conestrip_solutions(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    """
    An implementation of formula (4) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """

    # check that Omega_Gamma and Omega_Delta are a partition of { 0, ..., |f0|-1 }
    assert(set(Omega_Gamma).isdisjoint(set(Omega_Delta)))
    assert(set(list(range(len(f0)))) == set(Omega_Gamma) | set(Omega_Delta))
    # check that the gambles in R0 have the correct length
    assert(all(len(x) == len(f0) for x in flatten(R0)))

    # variables
    lambda_ = [Real(f'lambda{k}') for k in range(len(R0))]
    mu = [[Real(f'mu{k}_{i}') for i in range(len(R0[k]))] for k in range(len(R0))]
    sigma = Real('sigma')

    # constants
    g = [[[RealVal(R0[k][i][j]) for j in range(len(R0[k][i]))] for i in range(len(R0[k]))] for k in range(len(R0))]
    f = [RealVal(f0[j]) for j in range(len(f0))]

    # intermediate expressions
    h = sum_rows(list(sum_rows([product(mu[k][i], g[k][i]) for i in range(len(R0[k]))]) for k in range(len(R0))))
    goal = simplify(sum(lambda_))

    # 0 <= lambda <= 1
    lambda_constraints = [0 <= x for x in lambda_] + [x <= 1 for x in lambda_]

    # tau >= 1
    mu_constraints = [x >= 0 for x in collapse(mu)]

    # sigma >= 1
    sigma_constraints = [sigma >= 1]

    # main constraints
    constraints_1 = [goal >= 1]
    constraints_2 = [h[omega] <= sigma * f[omega] for omega in Omega_Gamma]
    constraints_3 = [h[omega] >= sigma * f[omega] for omega in Omega_Delta]
    constraints_4 = list(collapse([[lambda_[k] <= mu[k][i] for i in range(len(R0[k]))] for k in range(len(R0))]))

    if verbose:
        print('--- variables ---')
        print(lambda_)
        print(mu)
        print(sigma)
        print('--- constants ---')
        print('g =', g)
        print('f =', f)
        print('h =', h)
        print('--- constraints ---')
        print(lambda_constraints)
        print(mu_constraints)
        print(sigma_constraints)
        print(constraints_1)
        print(constraints_2)
        print(constraints_3)
        print(constraints_4)

    optimizer = Optimize()
    optimizer.add(lambda_constraints + mu_constraints + sigma_constraints + constraints_1 + constraints_2 + constraints_3 + constraints_4)
    optimizer.maximize(goal)
    if optimizer.check() == sat:
        model = optimizer.model()
        lambda_solution = [model.evaluate(lambda_[k]) for k in range(len(R0))]
        mu_solution = [[model.evaluate(mu[k][i]) for i in range(len(R0[k]))] for k in range(len(R0))]
        sigma_solution = model.evaluate(sigma)
        if verbose:
            print('--- solution ---')
            print('lambda =', lambda_solution)
            print('mu =', mu_solution)
            print('sigma =', sigma_solution)
        print('goal =', model.evaluate(goal))
        return lambda_solution, mu_solution, sigma_solution
    else:
        print("failed to solve")
        return None


def conestrip(R: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    while True:
        Lambda = conestrip_solutions(R, f0, Omega_Gamma, Omega_Delta)
        if not Lambda:
            return None
        lambda_, mu, _ = Lambda
        Q = [D for D, lambda_D in enumerate(lambda_) if lambda_D == 0]
        if all(x == 0 for x in collapse(mu[D] for D in Q)):
            return Lambda
        R = [R_D for D, R_D in enumerate(R) if D not in Q]


if __name__ == "__main__":
    R = parse_cone('''
      4 0 0
      0 5 0
      0 0 6

      1 0 1
      0 7 7

      1 2 3
      2 4 6
    ''')
    f = parse_gamble('2 5 8')
    Omega_Gamma = [0, 1]
    Omega_Delta = [2]

    print('==================')
    print('=== conestrip4 ===')
    print('==================')
    lambda_solution, mu_solution, sigma_solution = conestrip(R, f, Omega_Gamma, Omega_Delta, verbose=True)
    print('lambda =', lambda_solution)
    print('mu =', mu_solution)
    print('sigma =', sigma_solution)
