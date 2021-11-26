# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from more_itertools import collapse
from more_itertools.recipes import flatten
from typing import List
from z3 import *
from gambles import Cone, Gamble, parse_cone, parse_gamble
from utility import product, sum_rows


def is_valid_conestrip_input(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int]) -> bool:
    # check that Omega_Gamma and Omega_Delta are a partition of { 0, ..., |f0|-1 }
    # check that the gambles in R0 have the correct length
    return set(Omega_Gamma).isdisjoint(set(Omega_Delta)) and \
           set(list(range(len(f0)))) == set(Omega_Gamma) | set(Omega_Delta) and \
           all(len(x) == len(f0) for x in flatten(R0))


def conestrip1(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> bool:
    """
    An implementation of formula (1) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """
    assert is_valid_conestrip_input(R0, f0, Omega_Gamma, Omega_Delta)

    # variables
    lambda_ = [Real(f'lambda{d}') for d in range(len(R0))]
    nu = [[Real(f'nu{d}_{i}') for i in range(len(R0[d]))] for d in range(len(R0))]

    # constants
    g = [[[RealVal(R0[d][i][j]) for j in range(len(R0[d][i]))] for i in range(len(R0[d]))] for d in range(len(R0))]
    f = [RealVal(f0[j]) for j in range(len(f0))]

    # intermediate expressions
    h = sum_rows(list(product(lambda_[d], sum_rows([product(nu[d][i], g[d][i]) for i in range(len(R0[d]))])) for d in range(len(R0))))

    # 0 <= lambda <= 1
    lambda_constraints = [0 <= x for x in lambda_] + [x <= 1 for x in lambda_]

    # nu > 0
    nu_constraints = [0 < x for x in collapse(nu)]

    # main constraints
    constraints_1 = [simplify(sum(lambda_)) == 1]
    constraints_2 = [h[omega] <= f[omega] for omega in Omega_Gamma]
    constraints_3 = [h[omega] >= f[omega] for omega in Omega_Delta]

    if verbose:
        print('--- variables ---')
        print(lambda_)
        print(nu)
        print('--- constants ---')
        print('g =', g)
        print('f =', f)
        print('h =', h)
        print('--- constraints ---')
        print(lambda_constraints)
        print(nu_constraints)
        print(constraints_1)
        print(constraints_2)
        print(constraints_3)

    solver = Solver()
    solver.add(lambda_constraints + nu_constraints + constraints_1 + constraints_2 + constraints_3)
    if solver.check() == sat:
        model = solver.model()
        if verbose:
            print('--- solution ---')
            print('lambda =', [model.evaluate(lambda_[d]) for d in range(len(R0))])
            print('nu =', [[model.evaluate(nu[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))])
        return True
    else:
        return False


def conestrip2(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> bool:
    """
    An implementation of formula (2) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """
    assert is_valid_conestrip_input(R0, f0, Omega_Gamma, Omega_Delta)

    # variables
    lambda_ = [Real(f'lambda{d}') for d in range(len(R0))]
    tau = [[Real(f'tau{d}_{i}') for i in range(len(R0[d]))] for d in range(len(R0))]
    sigma = Real('sigma')

    # constants
    g = [[[RealVal(R0[d][i][j]) for j in range(len(R0[d][i]))] for i in range(len(R0[d]))] for d in range(len(R0))]
    f = [RealVal(f0[j]) for j in range(len(f0))]

    # intermediate expressions
    h = sum_rows(list(product(lambda_[d], sum_rows([product(tau[d][i], g[d][i]) for i in range(len(R0[d]))])) for d in range(len(R0))))

    # 0 <= lambda <= 1
    lambda_constraints = [0 <= x for x in lambda_] + [x <= 1 for x in lambda_]

    # tau >= 1
    tau_constraints = [x >= 1 for x in collapse(tau)]

    # sigma >= 1
    sigma_constraints = [sigma >= 1]

    # main constraints
    constraints_1 = [simplify(sum(lambda_)) >= 1]
    constraints_2 = [h[omega] <= sigma * f[omega] for omega in Omega_Gamma]
    constraints_3 = [h[omega] >= sigma * f[omega] for omega in Omega_Delta]

    if verbose:
        print('--- variables ---')
        print(lambda_)
        print(tau)
        print(sigma)
        print('--- constants ---')
        print('g =', g)
        print('f =', f)
        print('h =', h)
        print('--- constraints ---')
        print(lambda_constraints)
        print(tau_constraints)
        print(sigma_constraints)
        print(constraints_1)
        print(constraints_2)
        print(constraints_3)

    solver = Solver()
    solver.add(lambda_constraints + tau_constraints + sigma_constraints + constraints_1 + constraints_2 + constraints_3)
    if solver.check() == sat:
        model = solver.model()
        if verbose:
            print('--- solution ---')
            print('lambda =', [model.evaluate(lambda_[d]) for d in range(len(R0))])
            print('tau =', [[model.evaluate(tau[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))])
            print('sigma =', model.evaluate(sigma))
        return True
    else:
        return False


def conestrip3(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> bool:
    """
    An implementation of formula (3) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
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

    # 0 <= lambda <= 1
    lambda_constraints = [0 <= x for x in lambda_] + [x <= 1 for x in lambda_]

    # mu >= 0
    mu_constraints = [x >= 0 for x in collapse(mu)]

    # sigma >= 1
    sigma_constraints = [sigma >= 1]

    # main constraints
    constraints_1 = [simplify(sum(lambda_)) >= 1]
    constraints_2 = [h[omega] <= sigma * f[omega] for omega in Omega_Gamma]
    constraints_3 = [h[omega] >= sigma * f[omega] for omega in Omega_Delta]
    constraints_4 = list(collapse([[And(lambda_[d] <= mu[d][i], mu[d][i] <= lambda_[d] * mu[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))]))

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

    solver = Solver()
    solver.add(lambda_constraints + mu_constraints + sigma_constraints + constraints_1 + constraints_2 + constraints_3 + constraints_4)
    if solver.check() == sat:
        model = solver.model()
        if verbose:
            print('--- solution ---')
            print('lambda =', [model.evaluate(lambda_[d]) for d in range(len(R0))])
            print('mu =', [[model.evaluate(mu[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))])
            print('sigma =', model.evaluate(sigma))
        return True
    else:
        return False


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
    print('=== conestrip1 ===')
    print('==================')
    result1 = conestrip1(R, f, Omega_Gamma, Omega_Delta, verbose=True)

    print('==================')
    print('=== conestrip2 ===')
    print('==================')
    result2 = conestrip2(R, f, Omega_Gamma, Omega_Delta, verbose=True)

    print('==================')
    print('=== conestrip3 ===')
    print('==================')
    result3 = conestrip3(R, f, Omega_Gamma, Omega_Delta, verbose=True)

    print('results:', result1, result2, result3, '\n')
