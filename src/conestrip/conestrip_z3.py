# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, List, Optional, Tuple
from more_itertools import collapse
from more_itertools.recipes import flatten
from z3 import *

from conestrip.gamble_algorithms import is_positive_combination
from conestrip.cones import GeneralCone, Gamble, ConeGenerator
from conestrip.utility import product, sum_rows, is_solved


def is_valid_conestrip_input(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int]) -> bool:
    # check that the union of Omega_Gamma and Omega_Delta equals { 0, ..., |f0|-1 }
    # check that the gambles in R0 have the correct length
    return set(list(range(len(f0)))) == set(Omega_Gamma) | set(Omega_Delta) and \
           all(len(x) == len(f0) for x in flatten(R0))


def make_omega_constraints(f: Any, h: Any, Omega_Gamma: List[int], Omega_Delta: List[int]) -> Tuple[Any, Any, Any]:
    less_equal = set(Omega_Gamma) - set(Omega_Delta)
    greater_equal = set(Omega_Delta) - set(Omega_Gamma)
    equal = set(Omega_Gamma) & set(Omega_Delta)
    constraints_1 = [h[omega] <= f[omega] for omega in less_equal]
    constraints_2 = [h[omega] >= f[omega] for omega in greater_equal]
    constraints_3 = [h[omega] == f[omega] for omega in equal]
    return constraints_1, constraints_2, constraints_3


def make_omega_sigma_constraints(f: Any, h: Any, sigma: Any, Omega_Gamma: List[int], Omega_Delta: List[int]) -> Tuple[Any, Any, Any]:
    less_equal = set(Omega_Gamma) - set(Omega_Delta)
    greater_equal = set(Omega_Delta) - set(Omega_Gamma)
    equal = set(Omega_Gamma) & set(Omega_Delta)
    constraints_1 = [h[omega] <= sigma * f[omega] for omega in less_equal]
    constraints_2 = [h[omega] >= sigma * f[omega] for omega in greater_equal]
    constraints_3 = [h[omega] == sigma * f[omega] for omega in equal]
    return constraints_1, constraints_2, constraints_3


def conestrip1_constraints(R0: GeneralCone, f: List[Any], Omega_Gamma: List[int], Omega_Delta: List[int], variables: Tuple[Any, Any], verbose: bool = False) -> Tuple[List[Any], List[Any]]:
    # variables
    lambda_, nu = variables

    # constants
    g = [[[RealVal(R0[d][i][j]) for j in range(len(R0[d][i]))] for i in range(len(R0[d]))] for d in range(len(R0))]

    # if f contains elements of type ArithRef, then they are already in Z3 format
    if not isinstance(f[0], ArithRef):
        f = [RealVal(f[j]) for j in range(len(f))]

    # intermediate expressions
    h = sum_rows(list(product(lambda_[d], sum_rows([product(nu[d][i], g[d][i]) for i in range(len(R0[d]))])) for d in range(len(R0))))

    # 0 <= lambda <= 1
    lambda_constraints0 = [0 <= x for x in lambda_] + [x <= 1 for x in lambda_]

    # sum(lambda) == 1
    lambda_constraints1 = [simplify(sum(lambda_)) == 1]

    # nu > 0
    nu_constraints = [x > 0 for x in collapse(nu)]

    # omega constraints
    constraints_2, constraints_3, constraints_4 = make_omega_constraints(f, h, Omega_Gamma, Omega_Delta)

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
        print(lambda_constraints0)
        print(lambda_constraints1)
        print(nu_constraints)
        print(constraints_2)
        print(constraints_3)
        print(constraints_4)

    return lambda_constraints0 + lambda_constraints1 + nu_constraints, constraints_2 + constraints_3 + constraints_4


def solve_conestrip1(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Tuple[Any, Any]:
    """
    An implementation of formula (1) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """
    assert is_valid_conestrip_input(R0, f0, Omega_Gamma, Omega_Delta)

    # variables
    lambda_ = [Real(f'lambda{d}') for d in range(len(R0))]
    nu = [[Real(f'nu{d}_{i}') for i in range(len(R0[d]))] for d in range(len(R0))]

    constraints = list(flatten(conestrip1_constraints(R0, f0, Omega_Gamma, Omega_Delta, (lambda_, nu), verbose)))
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
        return None, None


def conestrip2_constraints(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], variables: Tuple[Any, Any, Any], verbose: bool = False) -> Tuple[List[Any], List[Any]]:
    # variables
    lambda_, tau, sigma = variables

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
    constraints_2, constraints_3, constraints_4 = make_omega_sigma_constraints(f, h, sigma, Omega_Gamma, Omega_Delta)
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

    return constraints


def solve_conestrip2(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    """
    An implementation of formula (2) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """
    assert is_valid_conestrip_input(R0, f0, Omega_Gamma, Omega_Delta)

    # variables
    lambda_ = [Real(f'lambda{d}') for d in range(len(R0))]
    tau = [[Real(f'tau{d}_{i}') for i in range(len(R0[d]))] for d in range(len(R0))]
    sigma = Real('sigma')

    constraints = conestrip2_constraints(R0, f0, Omega_Gamma, Omega_Delta, (lambda_, tau, sigma), verbose)
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
        return None, None, None


def conestrip3_constraints(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], variables: Tuple[Any, Any, Any], verbose: bool = False) -> Tuple[List[Any], List[Any]]:
    # variables
    lambda_, mu, sigma = variables

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
    constraints_2, constraints_3, constraints_4 = make_omega_sigma_constraints(f, h, sigma, Omega_Gamma, Omega_Delta)
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

    return constraints


def solve_conestrip3(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Tuple[Any, Any, Any]:
    """
    An implementation of formula (3) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """
    assert is_valid_conestrip_input(R0, f0, Omega_Gamma, Omega_Delta)

    # variables
    lambda_ = [Real(f'lambda{d}') for d in range(len(R0))]
    mu = [[Real(f'mu{d}_{i}') for i in range(len(R0[d]))] for d in range(len(R0))]
    sigma = Real('sigma')

    constraints = conestrip3_constraints(R0, f0, Omega_Gamma, Omega_Delta, (lambda_, mu, sigma), verbose)
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
        return None, None, None


def conestrip4_constraints(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], variables: Tuple[Any, Any, Any], verbose: bool = False) -> Tuple[List[Any], List[Any]]:
    # variables
    lambda_, mu, sigma = variables

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
    constraints_2, constraints_3, constraints_4 = make_omega_sigma_constraints(f, h, sigma, Omega_Gamma, Omega_Delta)
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

    return constraints


def solve_conestrip4(R0: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Tuple[Any, Any, Any]:
    """
    An implementation of formula (4) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """
    assert is_valid_conestrip_input(R0, f0, Omega_Gamma, Omega_Delta)

    # variables
    lambda_ = [Real(f'lambda{d}') for d in range(len(R0))]
    mu = [[Real(f'mu{d}_{i}') for i in range(len(R0[d]))] for d in range(len(R0))]
    sigma = Real('sigma')

    # expressions
    goal = simplify(sum(lambda_))

    constraints = conestrip4_constraints(R0, f0, Omega_Gamma, Omega_Delta, (lambda_, mu, sigma), verbose)
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
        return None, None, None


def conestrip_algorithm(R: GeneralCone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Tuple[Any, Any, Any]:
    """
    An implementation of the CONEstrip algorithm in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    @param R:
    @param f0:
    @param Omega_Gamma:
    @param Omega_Delta:
    @param verbose:
    @return: A solution (lambda, mu, sigma) to the CONEstrip optimization problem (4), or None if no solution exists
    """
    while True:
        lambda_, mu, sigma = solve_conestrip4(R, f0, Omega_Gamma, Omega_Delta, verbose)
        if not lambda_:
            return None, None, None
        if all(x == 0 for x in collapse(mu[d] for d, lambda_d in enumerate(lambda_) if lambda_d == 0)):
            return lambda_, mu, sigma
        R = [R_d for d, R_d in enumerate(R) if not lambda_[d] == 0]


def is_in_cone_generator(R: ConeGenerator, g: Gamble, verbose: bool = False) -> bool:
    n = len(g)
    Omega_Gamma = list(range(n))
    Omega_Delta = list(range(n))
    cone = [R]
    solution = solve_conestrip1(cone, g, Omega_Gamma, Omega_Delta, verbose=verbose)
    return is_solved(solution)


def is_in_closed_cone_generator(R: ConeGenerator, g: Gamble) -> bool:
    return is_positive_combination(g, R) is not None


def is_in_cone_generator_border(R: ConeGenerator, g: Gamble) -> bool:
    return not is_in_cone_generator(R, g) and is_in_closed_cone_generator(R, g)


def is_in_general_cone(cone: GeneralCone, g: Gamble, solver=solve_conestrip4) -> bool:
    n = len(g)
    Omega_Gamma = list(range(n))
    Omega_Delta = list(range(n))
    solution = solver(cone, g, Omega_Gamma, Omega_Delta, verbose=False)
    return is_solved(solution)
