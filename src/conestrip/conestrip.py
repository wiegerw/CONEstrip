# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, List, Optional, Tuple
from more_itertools import collapse
from z3 import *
from cone import is_valid_conestrip_input
from gambles import GeneralCone, Gamble, parse_general_cone, parse_gamble
from utility import product, sum_rows


def conestrip_solutions(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    """
    An implementation of formula (4) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """
    assert is_valid_conestrip_input(R0, f0, Omega_Gamma, Omega_Delta)

    # variables
    lambda_ = [Real(f'lambda{d}') for d in range(len(R0))]
    mu = [[Real(f'mu{d}_{i}') for i in range(len(R0[d]))] for d in range(len(R0))]
    sigma = Real('sigma')

    # constants
    g = [[[RealVal(R0[d][i][j]) for j in range(len(R0[d][i]))] for i in range(len(R0[d]))] for d in range(len(R0))]
    f = [RealVal(f0[j]) for j in range(len(f0))]

    # intermediate expressions
    h = sum_rows(list(sum_rows([product(mu[d][i], g[d][i]) for i in range(len(R0[d]))]) for d in range(len(R0))))
    goal = simplify(sum(lambda_))

    # 0 <= lambda <= 1
    lambda_constraints = [0 <= x for x in lambda_] + [x <= 1 for x in lambda_]

    # mu >= 0
    mu_constraints = [x >= 0 for x in collapse(mu)]

    # sigma >= 1
    sigma_constraints = [sigma >= 1]

    # main constraints
    constraints_1 = [goal >= 1]
    constraints_2 = [h[omega] <= sigma * f[omega] for omega in Omega_Gamma]
    constraints_3 = [h[omega] >= sigma * f[omega] for omega in Omega_Delta]
    constraints_4 = list(collapse([[lambda_[d] <= mu[d][i] for i in range(len(R0[d]))] for d in range(len(R0))]))

    if verbose:
        print('--- variables ---')
        print(lambda_)
        print(mu)
        print(sigma)
        print('--- constants ---')
        print('g =', g)
        print('f =', f)
        print('--- intermediate expressions ---')
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
        lambda_solution = [model.evaluate(lambda_[d]) for d in range(len(R0))]
        mu_solution = [[model.evaluate(mu[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))]
        sigma_solution = model.evaluate(sigma)
        if verbose:
            print('--- solution ---')
            print('lambda =', lambda_solution)
            print('mu =', mu_solution)
            print('sigma =', sigma_solution)
            print('goal =', model.evaluate(goal))
        return lambda_solution, mu_solution, sigma_solution
    else:
        return None


def conestrip(R: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    while True:
        Lambda = conestrip_solutions(R, f0, Omega_Gamma, Omega_Delta)
        if not Lambda:
            return None
        lambda_, mu, _ = Lambda
        Q = [d for d, lambda_d in enumerate(lambda_) if lambda_d == 0]
        if all(x == 0 for x in collapse(mu[d] for d in Q)):
            return Lambda
        R = [R_d for d, R_d in enumerate(R) if d not in Q]


if __name__ == "__main__":
    R = parse_general_cone('''
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

    lambda_solution, mu_solution, sigma_solution = conestrip(R, f, Omega_Gamma, Omega_Delta, verbose=True)
    print('lambda =', lambda_solution)
    print('mu =', mu_solution)
    print('sigma =', sigma_solution)
