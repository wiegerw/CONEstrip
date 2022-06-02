# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from fractions import Fraction
from more_itertools import collapse
from typing import Any, List, Optional, Set, Tuple
import z3
from conestrip.cones import GeneralCone, Gamble, ConeGenerator, ConvexCombination, linear_combination, parse_gamble
from conestrip.utility import sum_rows, product


PropositionalGamble = List[Fraction]
PropositionalConeGenerator = List[PropositionalGamble]
PropositionalGeneralCone = List[PropositionalConeGenerator]
PropositionalSentence = Any
PropositionalBasis = List[PropositionalSentence]
BooleanVariable = Any
BooleanGamble = List[Any]


def gamble_coefficients(g: Gamble, Phi: List[PropositionalSentence]) -> PropositionalGamble:
    """
    Get the coefficients of gamble g with respect to the basis Phi
    @param g: a gamble
    @param Phi: a sequence of basic functions
    """

    n = len(g)
    k = len(Phi)
    x = z3.Reals(' '.join(f'x{i}' for i in range(k)))

    solver = z3.Solver()
    for j in range(n):
        eqn = sum([Phi[i][j] * x[i] for i in range(k)]) == g[j]
        solver.add(eqn)

    solver.check()
    model = solver.model()
    return [model[xi] for xi in x]


def parse_boolean_gamble(text: str) -> Gamble:
    result = [Fraction(s) for s in text.strip().split()]
    assert all(f in [0,1] for f in result)
    return result


def parse_propositional_basis(text: str) -> List[PropositionalSentence]:
    gambles = list(map(parse_boolean_gamble, text.strip().split('\n')))
    return gambles


def solve_propositional_conestrip1(psi: PropositionalSentence, Phi: PropositionalBasis, C):
    k = len(Phi)
    solver = z3.Solver()
    solver.add(psi)
    solver.add(And(Phi[i] == C[i] for i in range(k)))
    if solver.check() == z3.sat:
        model = solver.model()
        return [model[c] for c in C]
    return None


def propositional_conestrip2_constraints(R: PropositionalGeneralCone, f0: PropositionalGamble, Gamma, Delta, variables: Tuple[Any, Any, Any, Any], verbose: bool = False) -> List[Any]:
    # variables
    lambda_, mu, sigma, kappa = variables

    # constants
    g = [[[z3.RealVal(R[d][i][j]) for j in range(len(R[d][i]))] for i in range(len(R[d]))] for d in range(len(R))]
    f = [z3.RealVal(f0[j]) for j in range(len(f0))]

    # intermediate expressions
    h = sum_rows(list(sum_rows([product(mu[d][i], g[d][i]) for i in range(len(R[d]))]) for d in range(len(R))))
    goal = z3.simplify(sum(lambda_))

    # 0 <= lambda <= 1
    lambda_constraints = [0 <= x for x in lambda_] + [x <= 1 for x in lambda_]

    # mu >= 0
    mu_constraints = [x >= 0 for x in collapse(mu)]

    # sigma >= 1
    sigma_constraints = [sigma >= 1]

    # main constraints
    constraints_1 = [goal >= 1]
    constraints_2 = [kappa[i] <= h[i] - sigma * f[i] for i in range(len(kappa))]
    constraints_3 = [kappa[i] >= h[i] - sigma * f[i] for i in range(len(kappa))]
    constraints_4 = list(collapse([[lambda_[d] <= mu[d][i] for i in range(len(R[d]))] for d in range(len(R))]))
    constraints = lambda_constraints + mu_constraints + sigma_constraints + constraints_1 + constraints_2 + constraints_3 + constraints_4

    if verbose:
        print('--- variables ---')
        print(lambda_)
        print(mu)
        print(sigma)
        print(kappa)
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

    return constraints


def solve_propositional_conestrip2(R: PropositionalGeneralCone, f0: PropositionalGamble, Gamma, Delta, Phi: PropositionalBasis, verbose: bool = False) -> Optional[Tuple[Any, Any, Any, Any]]:
    """
    An implementation of formula (8) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """

    # variables
    lambda_ = [z3.Real(f'lambda{d}') for d in range(len(R))]
    mu = [[z3.Real(f'mu{d}_{i}') for i in range(len(R[d]))] for d in range(len(R))]
    sigma = z3.Real('sigma')
    kappa = [z3.Real(f'kappa{i}') for i in range(len(Phi))]

    # expressions
    goal = z3.simplify(sum(lambda_))

    constraints = propositional_conestrip2_constraints(R, f0, Gamma, Delta, (lambda_, mu, sigma, kappa), verbose)
    optimizer = z3.Optimize()
    optimizer.add(constraints)
    optimizer.maximize(goal)
    if optimizer.check() == z3.sat:
        model = optimizer.model()
        lambda_solution = [model.evaluate(lambda_[d]) for d in range(len(R))]
        mu_solution = [[model.evaluate(mu[d][i]) for i in range(len(R[d]))] for d in range(len(R))]
        sigma_solution = model.evaluate(sigma)
        kappa_solution = [model.evaluate(kappa[d]) for d in range(len(kappa))]
        if verbose:
            print('--- solution ---')
            print('lambda =', lambda_solution)
            print('mu =', mu_solution)
            print('sigma =', sigma_solution)
            print('sigma =', kappa_solution)
            print('goal =', model.evaluate(goal))
        return lambda_solution, mu_solution, sigma_solution, kappa_solution
    else:
        return None, None, None, None


def solve_propositional_conestrip3(psi: PropositionalSentence, kappa: List[Fraction], B, C):
    k = len(Phi)
    optimizer = z3.Optimize()
    optimizer.add(psi)
    optimizer.add(And(Phi[i] == C[i] for i in range(k)))
    optimizer.maximize(sum(kappa[i] * C[i] for i in range(k)))
    if optimizer.check() == z3.sat:
        model = optimizer.model()
        return [model[c] for c in C]
    return None


def propositional_conestrip_algorithm(R: PropositionalGeneralCone, f0: PropositionalGamble, B, Phi: PropositionalBasis, psi: PropositionalSentence, psi_Gamma: PropositionalSentence, psi_Delta: PropositionalSentence, verbose: bool = False) -> Optional[Tuple[Any, Any, Any, Any]]:
    """
    An implementation of the Propositional CONEstrip algorithm in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    @param R:
    @param f0:
    @param B:
    @param Phi:
    @param psi:
    @param psi_Gamma:
    @param psi_Delta:
    @param verbose:
    @return: A solution (lambda, mu, sigma, kappa) to the propositional CONEstrip optimization problem, or None if no solution exists
    """
    k = len(Phi)
    C = z3.Bools(' '.join(f'c{i}' for i in range(k)))

    Gamma = []
    gamma = solve_propositional_conestrip1(z3.And(psi, psi_Gamma), Phi, C)
    if gamma:
        Gamma = [gamma]

    Delta = []
    delta = solve_propositional_conestrip1(z3.And(psi, psi_Delta), Phi, C)
    if delta:
        Gamma = [delta]

    while True:
        lambda_, mu, sigma, kappa = solve_propositional_conestrip2(R, f0, Gamma, Delta, Phi, verbose)
        if not lambda_:
            return None

        Q = [d for d, lambda_d in enumerate(lambda_) if lambda_d == 0]
        R = [R_d for d, R_d in enumerate(R) if lambda_[d] != 0]

        gamma = [0] * k
        delta = [0] * k

        if Gamma:
            gamma = solve_propositional_conestrip3(z3.And(psi, psi_Gamma), kappa, B, C)
            Gamma.append(gamma)

        if Delta:
            delta = solve_propositional_conestrip3(z3.And(psi, psi_Gamma), kappa, B, C)
            Delta.append(delta)

        if sum(kappa[i] * gamma[i] for i in range(k)) <= 0 and 0 <= sum(kappa[i] * delta[i] for i in range(k)) and all(x == 0 for x in collapse(mu[d] for d in Q)):
            return (lambda_, mu, sigma, kappa)
