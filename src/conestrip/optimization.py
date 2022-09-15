# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, List, Tuple
from more_itertools import collapse
from more_itertools.recipes import flatten
from z3 import *

from conestrip.cones import GeneralCone, Gamble
from conestrip.utility import product, sum_rows, random_rationals_summing_to_one

AtomicEvent = int
Event = List[AtomicEvent]
PossibilitySpace = List[AtomicEvent]

MassFunction = List[Fraction]
LowerPrevisionFunction = List[Tuple[Gamble, Fraction]]
ConditionalLowerPrevisionFunction = List[Tuple[Gamble, Event]]


def make_one_omega(i: int, N: int) -> Gamble:
    result = [Fraction(0)] * N
    result[i] = Fraction(1)
    return result


def make_one_Omega(N: int) -> Gamble:
    return [Fraction(1)] * N


def make_minus_one_Omega(N: int) -> Gamble:
    return [-Fraction(1)] * N


def make_zero(N: int) -> Gamble:
    return [Fraction(0)] * N


def make_one(B: Event, N: int) -> Gamble:
    return [Fraction(i in B) for i in range(N)]


def make_minus_one(B: Event, N: int) -> Gamble:
    return [-Fraction(i in B) for i in range(N)]


def is_unit_gamble(g: Gamble) -> bool:
    return g.count(Fraction(1)) == 1 and g.count(Fraction(0)) == len(g) - 1


def generate_mass_function(Omega: PossibilitySpace) -> MassFunction:
    N = len(Omega)
    return random_rationals_summing_to_one(N)


def lower_prevision_set_A(p: MassFunction, K: List[Gamble], Omega: PossibilitySpace) -> List[Gamble]:
    def dot(f: Gamble, g: Gamble) -> Fraction:
        return sum(x * y for (x, y) in zip(f, g))

    def minus(f: Gamble, c: Fraction) -> Gamble:
        return [f_i - c for f_i in f]

    return [minus(h, dot(p, h)) for h in K]


def optimize_constraints(R: GeneralCone, f: List[Any], B: List[Tuple[Any, Any]], Omega: PossibilitySpace, variables: Tuple[Any, Any], verbose: bool = False) -> Tuple[List[Any], List[Any]]:
    # variables
    lambda_, nu = variables

    # constants
    g = [[[RealVal(R[d][i][j]) for j in range(len(R[d][i]))] for i in range(len(R[d]))] for d in range(len(R))]

    # if f contains elements of type ArithRef, then they are already in Z3 format
    if not isinstance(f[0], ArithRef):
        f = [RealVal(f[j]) for j in range(len(f))]

    # intermediate expressions
    h = sum_rows(list(product(lambda_[d], sum_rows([product(nu[d][i], g[d][i]) for i in range(len(R[d]))])) for d in range(len(R))))

    # 0 < lambda
    lambda_constraints = [0 < x for x in lambda_]

    # 0 <= nu
    nu_constraints = [0 <= x for x in collapse(nu)]

    constraints_1 = [h[omega] == f[omega] for omega in Omega]

    constraints_2 = []
    for b, c in range(len(B)):
        h_j = sum_rows(list(product(lambda_[d], sum_rows([product(nu[d][i], b[d][i]) for i in range(len(R[d]))])) for d in range(len(R))))
        h_j_constraints = [h_j[omega] == f[omega] for omega in Omega]
        constraints_2.extend(h_j_constraints)

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

    return lambda_constraints + nu_constraints, constraints_1 + constraints_2


def optimize_find(R: GeneralCone, f: Gamble, B: List[Tuple[Any, Any]], Omega: List[int], verbose: bool = False):
    # variables
    lambda_ = [Real(f'lambda{d}') for d in range(len(R))]
    nu = [[Real(f'nu{d}_{i}') for i in range(len(R[d]))] for d in range(len(R))]

    constraints = list(flatten(optimize_constraints(R, f, B, Omega, (lambda_, nu), verbose)))
    solver = Solver()
    solver.add(constraints)
    if solver.check() == sat:
        model = solver.model()
        lambda_solution = [model.evaluate(lambda_[d]) for d in range(len(R))]
        nu_solution = [[model.evaluate(nu[d][i]) for i in range(len(R[d]))] for d in range(len(R))]
        if verbose:
            print('--- solution ---')
            print('lambda =', lambda_solution)
            print('nu =', nu_solution)
        return lambda_solution, nu_solution
    else:
        return None, None


def optimize_maximize(R: GeneralCone, f: Gamble, a: Any, B: List[Tuple[Any, Any]], Omega: List[int], verbose: bool = False):
    # variables
    lambda_ = [Real(f'lambda{d}') for d in range(len(R))]
    nu = [[Real(f'nu{d}_{i}') for i in range(len(R[d]))] for d in range(len(R))]

    constraints = list(flatten(optimize_constraints(R, f, B, Omega, (lambda_, nu), verbose)))
    goal = None
    optimizer = Optimize()
    optimizer.add(constraints)
    optimizer.maximize(goal)
    if optimizer.check() == sat:
        model = optimizer.model()
        lambda_solution = [model.evaluate(lambda_[d]) for d in range(len(R))]
        nu_solution = [[model.evaluate(nu[d][i]) for i in range(len(R[d]))] for d in range(len(R))]
        if verbose:
            print('--- solution ---')
            print('lambda =', lambda_solution)
            print('nu =', nu_solution)
            print('goal =', model.evaluate(goal))
        return lambda_solution, nu_solution
    else:
        return None, None


def minus_constant(f: Gamble, c: Fraction) -> Gamble:
    return [x - c for x in f]


def make_lower_prevision_function1(p: MassFunction, K: List[Gamble]) -> LowerPrevisionFunction:
    def value(f: Gamble) -> Fraction:
        assert len(f) == len(p)
        return sum(p_i * f_i for (p_i, f_i) in zip(p, f))

    return [(f, value(f)) for f in K]


def make_lower_prevision_function2(p: MassFunction, K: List[Gamble], epsilon: Fraction) -> LowerPrevisionFunction:
    def value(f: Gamble) -> Fraction:
        assert len(f) == len(p)
        return (1 - epsilon) * sum(p_i * f_i for (p_i, f_i) in zip(p, f)) + epsilon * min(f)

    return [(f, value(f)) for f in K]


def lower_prevision_assessment(P: LowerPrevisionFunction):
    return [minus_constant(h, c) for (h, c) in P]


def incurs_sure_loss(R: GeneralCone, Omega: PossibilitySpace) -> bool:
    N = len(Omega)
    zero = make_zero(N)
    lambda_, mu = optimize_find(R, zero, [], Omega)
    return lambda_ is not None


def incurs_sure_loss1(A: List[Gamble], Omega: PossibilitySpace) -> bool:
    N = len(Omega)
    Omega = list(range(N))
    zero = make_zero(N)
    D = [make_one_omega(i, N) for i in range(N)] + [a for a in A if not a == zero and not is_unit_gamble(a)]
    R = [D]

    lambda_, mu = optimize_find(R, zero, [], Omega)
    return lambda_ is not None


def is_unconditional_natural_extension1(A: List[Gamble], Omega: PossibilitySpace) -> bool:
    N = len(Omega)
    Omega = list(range(N))
    zero = make_zero(N)
    one_Omega = make_one_Omega(N)
    minus_one_Omega = make_minus_one_Omega(N)

    R1 = [[g] for g in A]
    R2 = [[make_one_Omega(N)], [make_minus_one_Omega(N)], [make_zero(N)]]
    R3 = [[make_one_omega(i, N) for i in range(N)]]
    R = R1 + R2 + R3  # TODO: remove duplicates

    def a_value(D, g):
        if D == [one_Omega] and g == one_Omega:
            return 1
        elif D == [minus_one_Omega] and g == minus_one_Omega:
            return -1
        return 0

    a = [[a_value(D, g) for g in D] for D in R]

    lambda_, mu = optimize_maximize(R, zero, a, [], Omega)
    return lambda_ is not None


def incurs_partial_loss1(Q: ConditionalLowerPrevisionFunction, B_: List[Tuple[Gamble, List[int]]], C: Event) -> bool:
    N = len(Q)
    Omega = list(range(N))
    zero = make_zero(N)

    R1 = [[g, make_one(B, N)] for (g, B) in B_ if g != zero]
    R2 = [[make_one_omega(i, N) for i in range(N)]]
    R = R1 + R2  # TODO: remove duplicates

    lambda_, mu = optimize_find(R, zero, [], Omega)
    return lambda_ is not None


def is_conditional_natural_extension1(Q: ConditionalLowerPrevisionFunction, B_: List[Tuple[Gamble, List[int]]], C: Event) -> bool:
    N = len(Q)
    Omega = list(range(N))
    zero = make_zero(N)
    one_C = make_one(C, N)
    minus_one_C = make_minus_one(C, N)
    one_Omega = make_one_Omega(N)
    minus_one_Omega = make_minus_one_Omega(N)

    R1 = [[g, make_one(B, N)] for (g, B) in B_]
    R2 = [[one_C], [minus_one_C], [zero]]
    R3 = [[make_one_omega(i, N) for i in range(N)]]
    R = R1 + R2 + R3  # TODO: remove duplicates

    def a_value(D, g):
        if D == [one_Omega] and g == one_Omega:
            return 1
        elif D == [minus_one_Omega] and g == minus_one_Omega:
            return -1
        return 0

    a = [[a_value(D, g) for g in D] for D in R]

    lambda_, mu = optimize_maximize(R, zero, a, [], Omega)
    return lambda_ is not None
