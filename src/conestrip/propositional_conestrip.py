# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from fractions import Fraction
from more_itertools import collapse
from typing import Any, List, Optional, Tuple
import z3
from conestrip.propositional_cones import PropositionalBasis, PropositionalGamble, PropositionalSentence, PropositionalGeneralCone, BooleanVariable
from conestrip.utility import sum_rows, product


def solve_propositional_conestrip1(psi: PropositionalSentence,
                                   B: List[BooleanVariable],  # unused, since the solution for variables in B is omitted
                                   C: List[BooleanVariable],
                                   Phi: PropositionalBasis
                                  ):
    k = len(Phi)
    solver = z3.Solver()
    solver.add(psi)
    solver.add(z3.And([Phi[i] == C[i] for i in range(k)]))
    if solver.check() == z3.sat:
        model = solver.model()
        return [model[c] for c in C]
    return None


def propositional_conestrip2_constraints(R: PropositionalGeneralCone,
                                         f: PropositionalGamble,
                                         Gamma: List[Any],
                                         Delta: List[Any],
                                         Phi: PropositionalBasis,
                                         variables: Tuple[Any, Any, Any, Any],
                                         verbose: bool = False
                                        ) -> List[Any]:
    k = len(Phi)

    # variables
    lambda_, mu, sigma, kappa = variables

    # constants
    g = [[[z3.RealVal(R[d][i][j]) for j in range(len(R[d][i]))] for i in range(len(R[d]))] for d in range(len(R))]
    f_ = [z3.RealVal(f[j]) for j in range(len(f))]

    # intermediate expressions
    h = sum_rows(list(sum_rows([product(mu[d][i], g[d][i]) for i in range(len(R[d]))]) for d in range(len(R))))

    # 0 <= lambda <= 1
    lambda_constraints = [0 <= x for x in lambda_] + [x <= 1 for x in lambda_]

    # mu >= 0
    mu_constraints = [x >= 0 for x in collapse(mu)]

    # sigma >= 1
    sigma_constraints = [sigma >= 1]

    # main constraints
    constraints_1 = [sum(lambda_) >= 1]
    constraints_2 = list(collapse([[lambda_[d] <= mu[d][g] for g in range(len(R[d]))] for d in range(len(R))]))
    constraints_3 = [kappa[i] == h[i] - sigma * f_[i] for i in range(len(kappa))]
    constraints_4 = [z3.simplify(linear_combination(kappa, gamma)) <= 0 for gamma in Gamma]
    constraints_5 = [z3.simplify(linear_combination(kappa, delta)) >= 0 for delta in Delta]
    constraints = lambda_constraints + mu_constraints + sigma_constraints + constraints_1 + constraints_2 + constraints_3 + constraints_4 + constraints_5

    if verbose:
        print('--- variables ---')
        print(lambda_)
        print(mu)
        print(sigma)
        print(kappa)
        print('--- constants ---')
        print('g =', g)
        print('f =', f_)
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


# Evaluate a model value and convert it to Fraction
def model_value(model, x):
    y = model.evaluate(x)
    return Fraction(y.numerator_as_long(), y.denominator_as_long())


# C contains elements with type boolean
def linear_combination(kappa: List[Fraction], C: List[Any]):
    return sum(z3.If(C[i], kappa[i], z3.RealVal(0)) for i in range(len(kappa)))


def solve_propositional_conestrip2(R: PropositionalGeneralCone,
                                   f: PropositionalGamble,
                                   Gamma,
                                   Delta,
                                   Phi: PropositionalBasis,
                                   verbose: bool = False
                                  ) -> Optional[Tuple[Any, Any, Any, Any]]:
    """
    An implementation of formula (8) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """

    if verbose:
        print('=== propositional_conestrip2 ===')

    # variables
    lambda_ = [z3.Real(f'lambda{d}') for d in range(len(R))]
    mu = [[z3.Real(f'mu{d}_{i}') for i in range(len(R[d]))] for d in range(len(R))]
    sigma = z3.Real('sigma')
    kappa = [z3.Real(f'kappa{i}') for i in range(len(Phi))]

    # expressions
    goal = z3.simplify(sum(lambda_))

    constraints = propositional_conestrip2_constraints(R, f, Gamma, Delta, Phi, (lambda_, mu, sigma, kappa), verbose)
    optimizer = z3.Optimize()
    optimizer.add(constraints)
    optimizer.maximize(goal)
    if optimizer.check() == z3.sat:
        model = optimizer.model()
        lambda_solution = [model_value(model, lambda_[d]) for d in range(len(R))]
        mu_solution = [[model_value(model, mu[d][i]) for i in range(len(R[d]))] for d in range(len(R))]
        sigma_solution = model_value(model, sigma)
        kappa_solution = [model_value(model, kappa[d]) for d in range(len(kappa))]
        if verbose:
            print('--- solution ---')
            print(model)
        return lambda_solution, mu_solution, sigma_solution, kappa_solution
    else:
        return None, None, None, None


def solve_propositional_conestrip3(psi: PropositionalSentence,
                                   kappa: List[Fraction],
                                   B: List[BooleanVariable],  # unused, since the solution for variables in B is omitted
                                   C: List[BooleanVariable],
                                   Phi: PropositionalBasis
                                  ):
    k = len(Phi)
    optimizer = z3.Optimize()
    optimizer.add(psi)
    # optimizer.maximize(sum(kappa[i] * C[i] for i in range(k)))  # N.B. this doesn't work in Z3
    optimizer.maximize(linear_combination(kappa, C))
    if optimizer.check() == z3.sat:
        model = optimizer.model()
        return [bool(model[c]) for c in C]
    return None


def propositional_conestrip_algorithm(R: PropositionalGeneralCone,
                                      f: PropositionalGamble,
                                      B: List[BooleanVariable],
                                      Phi: PropositionalBasis,
                                      psi: PropositionalSentence,
                                      psi_Gamma: PropositionalSentence,
                                      psi_Delta: PropositionalSentence,
                                      verbose: bool = False
                                     ) -> Optional[Tuple[Any, Any, Any, Any]]:
    """
    An implementation of the Propositional CONEstrip algorithm in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    @param R:
    @param f:
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
    gamma = solve_propositional_conestrip1(z3.And(psi, psi_Gamma), B, C, Phi)
    if gamma:
        Gamma = [gamma]

    Delta = []
    delta = solve_propositional_conestrip1(z3.And(psi, psi_Delta), B, C, Phi)
    if delta:
        Delta = [delta]

    if verbose:
        print(f'R = {R}')
        print(f'f = {f}')
        print(f'psi && psi_Gamma = {z3.simplify(z3.And(psi, psi_Gamma))}')
        print(f'psi && psi_Delta = {z3.simplify(z3.And(psi, psi_Delta))}')
        print(f'B = {B}')
        print(f'Gamma = {Gamma}')
        print(f'Delta = {Delta}')

    while True:
        lambda_, mu, sigma, kappa = solve_propositional_conestrip2(R, f, Gamma, Delta, Phi, verbose=False)
        if not lambda_:
            return None

        R = [R_d for d, R_d in enumerate(R) if lambda_[d] != 0]

        gamma = [0] * k
        delta = [0] * k

        if Gamma:
            gamma = solve_propositional_conestrip3(z3.And(psi, psi_Gamma), kappa, B, C, Phi)
            Gamma.append(gamma)

        if Delta:
            delta = solve_propositional_conestrip3(z3.And(psi, psi_Delta), kappa, B, C, Phi)
            Delta.append(delta)

        if verbose:
            print('\n- iteration -')
            print(f'lambda = {lambda_}')
            print(f'mu = {mu}')
            print(f'sigma = {sigma}')
            print(f'kappa = {kappa}')
            print(f'R = {R}')
            if Gamma:
                print(f'gamma = {gamma}')
            print(f'Gamma = {Gamma}')
            if Delta:
                print(f'delta = {delta}')
            print(f'Delta = {Delta}')

        if sum(kappa[i] * gamma[i] for i in range(k)) <= 0 <= sum(kappa[i] * delta[i] for i in range(k)) \
                and all(x == 0 for x in collapse(mu[d] for d, lambda_d in enumerate(lambda_) if lambda_d == 0)):
            return lambda_, mu, sigma, kappa
