# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, List, Optional, Set, Tuple
from more_itertools import collapse
from more_itertools.recipes import flatten
from z3 import *
from conestrip.cones import GeneralCone, Gamble
from conestrip.utility import product, sum_rows


def is_valid_conestrip_input(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int]) -> bool:
    # check that the union of Omega_Gamma and Omega_Delta equals { 0, ..., |f0|-1 }
    # check that the gambles in R0 have the correct length
    return set(list(range(len(f0)))) == set(Omega_Gamma) | set(Omega_Delta) and \
           all(len(x) == len(f0) for x in flatten(R0))


def omega_sets(Omega_Gamma: List[int], Omega_Delta: List[int]) -> Tuple[Set[int], Set[int], Set[int]]:
    less_equal = set(Omega_Gamma) - set(Omega_Delta)
    greater_equal = set(Omega_Delta) - set(Omega_Gamma)
    equal = set(Omega_Gamma) & set(Omega_Delta)
    return less_equal, greater_equal, equal


def conestrip1(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any]]:
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
    nu_constraints = [x > 0 for x in collapse(nu)]

    # main constraints
    less_equal, greater_equal, equal = omega_sets(Omega_Gamma, Omega_Delta)
    constraints_1 = [simplify(sum(lambda_)) == 1]
    constraints_2 = [h[omega] <= f[omega] for omega in less_equal]
    constraints_3 = [h[omega] >= f[omega] for omega in greater_equal]
    constraints_4 = [h[omega] == f[omega] for omega in equal]

    constraints = lambda_constraints + nu_constraints + constraints_1 + constraints_2 + constraints_3 + constraints_4

    if verbose:
        print('--- variables ---')
        print(lambda_)
        print(nu)
        print('--- constants ---')
        print('g =', g)
        print('f =', f)
        print('--- intermediate expressions ---')
        print('h =', h)
        print('--- constraints ---')
        print(lambda_constraints)
        print(nu_constraints)
        print(constraints_1)
        print(constraints_2)
        print(constraints_3)
        print(constraints_4)

    solver = Solver()
    solver.add(constraints)
    if solver.check() == sat:
        model = solver.model()
        lambda_solution = [model.evaluate(lambda_[d]) for d in range(len(R0))]
        nu_solution = [[model.evaluate(nu[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))]
        if verbose:
            print('--- solution ---')
            print('lambda =', lambda_solution)
            print('nu =', nu_solution)
        return lambda_solution, nu_solution
    else:
        return None


def conestrip2(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
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
    less_equal, greater_equal, equal = omega_sets(Omega_Gamma, Omega_Delta)
    constraints_1 = [simplify(sum(lambda_)) >= 1]
    constraints_2 = [h[omega] <= f[omega] for omega in less_equal]
    constraints_3 = [h[omega] >= f[omega] for omega in greater_equal]
    constraints_4 = [h[omega] == f[omega] for omega in equal]
    constraints = lambda_constraints + tau_constraints + sigma_constraints + constraints_1 + constraints_2 + constraints_3 + constraints_4

    if verbose:
        print('--- variables ---')
        print(lambda_)
        print(tau)
        print(sigma)
        print('--- constants ---')
        print('g =', g)
        print('f =', f)
        print('--- intermediate expressions ---')
        print('h =', h)
        print('--- constraints ---')
        print(lambda_constraints)
        print(tau_constraints)
        print(sigma_constraints)
        print(constraints_1)
        print(constraints_2)
        print(constraints_3)
        print(constraints_4)

    solver = Solver()
    solver.add(constraints)
    if solver.check() == sat:
        model = solver.model()
        lambda_solution = [model.evaluate(lambda_[d]) for d in range(len(R0))]
        tau_solution = [[model.evaluate(tau[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))]
        sigma_solution = model.evaluate(sigma)
        if verbose:
            print('--- solution ---')
            print('lambda =', lambda_solution)
            print('tau =', tau_solution)
            print('sigma =', sigma_solution)
        return lambda_solution, tau_solution, sigma_solution
    else:
        return None


def conestrip3(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
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
    less_equal, greater_equal, equal = omega_sets(Omega_Gamma, Omega_Delta)
    constraints_1 = [simplify(sum(lambda_)) >= 1]
    constraints_2 = [h[omega] <= f[omega] for omega in less_equal]
    constraints_3 = [h[omega] >= f[omega] for omega in greater_equal]
    constraints_4 = [h[omega] == f[omega] for omega in equal]
    constraints_5 = list(collapse([[And(lambda_[d] <= mu[d][i], mu[d][i] <= lambda_[d] * mu[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))]))
    constraints = lambda_constraints + mu_constraints + sigma_constraints + constraints_1 + constraints_2 + constraints_3 + constraints_4 + constraints_5

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
        print(constraints_5)

    solver = Solver()
    solver.add(constraints)
    if solver.check() == sat:
        model = solver.model()
        lambda_solution = [model.evaluate(lambda_[d]) for d in range(len(R0))]
        mu_solution = [[model.evaluate(mu[d][i]) for i in range(len(R0[d]))] for d in range(len(R0))]
        sigma_solution = model.evaluate(sigma)
        if verbose:
            print('--- solution ---')
            print('lambda =', lambda_solution)
            print('tau =', mu_solution)
            print('sigma =', sigma_solution)
        return lambda_solution, mu_solution, sigma_solution
    else:
        return None


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
    less_equal, greater_equal, equal = omega_sets(Omega_Gamma, Omega_Delta)
    constraints_1 = [goal >= 1]
    constraints_2 = [h[omega] <= f[omega] for omega in less_equal]
    constraints_3 = [h[omega] >= f[omega] for omega in greater_equal]
    constraints_4 = [h[omega] == f[omega] for omega in equal]
    constraints_5 = list(collapse([[lambda_[d] <= mu[d][i] for i in range(len(R0[d]))] for d in range(len(R0))]))
    constraints = lambda_constraints + mu_constraints + sigma_constraints + constraints_1 + constraints_2 + constraints_3 + constraints_4 + constraints_5

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
        print(constraints_5)

    optimizer = Optimize()
    optimizer.add(constraints)
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


def conestrip(R: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int]) -> Optional[Tuple[Any, Any, Any]]:
    """
    An implementation of the CONEstrip algorithm in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    @param R:
    @param f0:
    @param Omega_Gamma:
    @param Omega_Delta:
    @return: A solution (lambda, mu, sigma) to the CONEstrip optimization problem (4), or None if no solution exists
    """
    while True:
        Lambda = conestrip_solutions(R, f0, Omega_Gamma, Omega_Delta)
        if not Lambda:
            return None
        lambda_, mu, _ = Lambda
        Q = [d for d, lambda_d in enumerate(lambda_) if lambda_d == 0]
        if all(x == 0 for x in collapse(mu[d] for d in Q)):
            return Lambda
        R = [R_d for d, R_d in enumerate(R) if d not in Q]


def is_in_general_cone(cone: GeneralCone, g: Gamble) -> Any:
    n = len(g)
    Omega_Gamma = list(range(n))
    Omega_Delta = list(range(n))
    return conestrip1(cone, g, Omega_Gamma, Omega_Delta, verbose=True)
