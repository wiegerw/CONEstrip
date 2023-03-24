# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
from itertools import chain, combinations
from typing import Any, List, Tuple

from more_itertools import collapse
from more_itertools.recipes import flatten
from z3 import *

from conestrip.cones import GeneralCone, Gamble, print_gamble, print_general_cone, print_cone_generator
from conestrip.global_settings import GlobalSettings
from conestrip.utility import product, sum_rows, random_rationals_summing_to_one

AtomicEvent = int
Event = List[AtomicEvent]
PossibilitySpace = List[AtomicEvent]
MassFunction = List[Fraction]
LowerPrevisionFunction = List[Tuple[Gamble, Fraction]]
LowerPrevisionAssessment = List[Gamble]
ConditionalLowerPrevisionFunction = List[Tuple[Gamble, Event, Fraction]]
ConditionalLowerPrevisionAssessment = List[Tuple[Gamble, Event]]


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def print_lower_prevision_function(P: LowerPrevisionFunction, pretty=False) -> str:
    if pretty:
        items = [f'({print_gamble(g, pretty)}, {float(c)})' for (g, c) in P]
        return '[{}]'.format(', '.join(items))
    else:
        items = [f'({print_gamble(g)}, {c})' for (g, c) in P]
        return ', '.join(items)


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


def generate_mass_function(Omega: PossibilitySpace, number_of_zeroes: int = 0, decimals: int = 2) -> MassFunction:
    if number_of_zeroes == 0:
        N = len(Omega)
        return random_rationals_summing_to_one(N, 10 ** decimals)
    else:
        N = len(Omega)
        values = random_rationals_summing_to_one(N - number_of_zeroes, 10 ** decimals)
        zero_positions = random.sample(range(N), number_of_zeroes)
        result = [Fraction(0)] * N
        index = 0
        for i in range(N):
            if i not in zero_positions:
                result[i] = values[index]
                index += 1
        return result


# Generates a mass function on the subsets of Omega.
# The mass of the empty set is always 0.
# The subsets are assumed to be ordered according to powerset, which means
# that the empty set is always at the front.
def generate_subset_mass_function(Omega: PossibilitySpace, decimals: int = 2) -> MassFunction:
    N = 2 ** len(Omega)
    return [Fraction(0)] + random_rationals_summing_to_one(N - 1, 10 ** decimals)


# generates 1 mass function with 1 non-zero, 2 with 2 non-zeroes, etc.
def generate_mass_functions(Omega: PossibilitySpace, decimals: int = 2) -> List[MassFunction]:
    result = []
    N = len(Omega)
    for number_of_zeroes in range(N - 1, -1, -1):
        for _ in range(N - number_of_zeroes):
            result.append(generate_mass_function(Omega, number_of_zeroes, decimals))
    return result


def is_mass_function(p: MassFunction) -> Bool:
    return all(x >= 0 for x in p) and sum(p) == 1


def optimize_constraints(R: GeneralCone, f: List[Any], B: List[Tuple[Any, Any]], Omega: PossibilitySpace, variables: Any) -> Tuple[List[Any], List[Any]]:
    # variables
    mu = variables

    # constants
    g = [[[RealVal(R[d][i][j]) for j in range(len(R[d][i]))] for i in range(len(R[d]))] for d in range(len(R))]

    # if f contains elements of type ArithRef, then they are already in Z3 format
    if not isinstance(f[0], ArithRef):
        f = [RealVal(f[j]) for j in range(len(f))]

    # intermediate expressions
    h = sum_rows(list(sum_rows([product(mu[d][i], g[d][i]) for i in range(len(R[d]))]) for d in range(len(R))))

    # 0 <= mu && exists mu_D: (x != 0 for x in mu_D)
    mu_constraints = [0 <= x for x in collapse(mu)] + [Or([And([x != 0 for x in mu_D]) for mu_D in mu])]

    constraints_1 = [h[omega] == f[omega] for omega in Omega]

    constraints_2 = []
    for b, c in range(len(B)):
        h_j = sum_rows(list(sum_rows([product(mu[d][i], b[d][i]) for i in range(len(R[d]))]) for d in range(len(R))))
        h_j_constraints = [h_j[omega] == f[omega] for omega in Omega]
        constraints_2.extend(h_j_constraints)

    if GlobalSettings.print_smt:
        print('--- variables ---')
        print(mu)
        print('--- constants ---')
        print('g =', g)
        print('f =', f)
        print('--- intermediate expressions ---')
        print('h =', h)
        print('--- constraints ---')
        print(mu_constraints)
        print(constraints_1)

    return mu_constraints, constraints_1 + constraints_2


def optimize_find(R: GeneralCone, f: Gamble, B: List[Tuple[Any, Any]], Omega: List[int]) -> Any:
    # variables
    mu = [[Real(f'mu{d}_{i}') for i in range(len(R[d]))] for d in range(len(R))]

    constraints = list(flatten(optimize_constraints(R, f, B, Omega, mu)))
    solver = Solver()
    solver.add(constraints)
    if GlobalSettings.verbose:
        print('=== optimize_find ===')
        print_constraints('constraints:\n', constraints)
    if solver.check() == sat:
        model = solver.model()
        mu_solution = [[model.evaluate(mu[d][i]) for i in range(len(R[d]))] for d in range(len(R))]
        if GlobalSettings.verbose:
            print('SAT')
            print('mu =', mu_solution)
        return mu_solution
    else:
        if GlobalSettings.verbose:
            print('UNSAT')
        return None


def print_constraints(msg: str, constraints: List[Any]) -> None:
    print(msg)
    for constraint in constraints:
        print(constraint)
        print('')


def optimize_maximize_full(R: GeneralCone, f: Gamble, a: List[List[Fraction]], B: List[Tuple[Any, Any]], Omega: List[int]) -> Tuple[Any, Fraction]:
    # variables
    mu = [[Real(f'mu{d}_{i}') for i in range(len(R[d]))] for d in range(len(R))]

    constraints = list(flatten(optimize_constraints(R, f, B, Omega, mu)))
    goal = simplify(sum(sum(mu[d][g] * a[d][g] for g in range(len(mu[d]))) for d in range(len(mu))))
    optimizer = Optimize()
    optimizer.add(constraints)
    optimizer.maximize(goal)
    if GlobalSettings.verbose:
        print('=== optimize_maximize ===')
        print('goal:', goal)
        print_constraints('constraints:\n', constraints)
    if optimizer.check() == sat:
        model = optimizer.model()
        mu_solution = [[model.evaluate(mu[d][i]) for i in range(len(R[d]))] for d in range(len(R))]
        goal_solution = model.evaluate(goal)
        if GlobalSettings.verbose:
            print('SAT')
            print('mu =', mu_solution)
            print('goal =', model.evaluate(goal))
        return mu_solution, goal_solution
    else:
        if GlobalSettings.verbose:
            print('UNSAT')
        return None, Fraction(0)


def optimize_maximize(R: GeneralCone, f: Gamble, a: List[List[Fraction]], B: List[Tuple[Any, Any]], Omega: List[int]):
    mu, goal = optimize_maximize_full(R, f, a, B, Omega)
    return mu


def optimize_maximize_value(R: GeneralCone, f: Gamble, a: List[List[Fraction]], B: List[Tuple[Any, Any]], Omega: List[int]):
    mu, goal = optimize_maximize_full(R, f, a, B, Omega)
    return goal


def minus_constant(f: Gamble, c: Fraction) -> Gamble:
    return [x - c for x in f]


def linear_lower_prevision_function(p: MassFunction, K: List[Gamble]) -> LowerPrevisionFunction:
    def value(f: Gamble) -> Fraction:
        assert len(f) == len(p)
        return sum(p_i * f_i for (p_i, f_i) in zip(p, f))

    return [(f, value(f)) for f in K]


def linear_vacuous_lower_prevision_function(p: MassFunction, K: List[Gamble], delta: Fraction) -> LowerPrevisionFunction:
    def value(f: Gamble) -> Fraction:
        assert len(f) == len(p)
        return (1 - delta) * sum(p_i * f_i for (p_i, f_i) in zip(p, f)) + delta * min(f)

    return [(f, value(f)) for f in K]


# We assume that the subsets of Omega are ordered according to the powerset function
def belief_lower_prevision_function(p: MassFunction, K: List[Gamble]) -> LowerPrevisionFunction:
    def value(f: Gamble) -> Fraction:
        assert len(p) == 2 ** len(f)
        return sum((p[i] * min(S)) for (i, S) in enumerate(powerset(f)) if i > 0)  # N.B. skip the empty set!

    return [(f, value(f)) for f in K]


def lower_prevision_assessment(P: LowerPrevisionFunction) -> LowerPrevisionAssessment:
    return [minus_constant(h, c) for (h, c) in P]


def conditional_lower_prevision_assessment(P: ConditionalLowerPrevisionFunction, Omega: PossibilitySpace) -> ConditionalLowerPrevisionAssessment:
    def hadamard(f: Gamble, g: Gamble) -> Gamble:
        return [x * y for (x, y) in zip(f, g)]

    N = len(Omega)
    return [(hadamard(minus_constant(h, c), make_one(B, N)), B) for (h, B, c) in P]


def sure_loss_cone(A: LowerPrevisionAssessment, Omega: PossibilitySpace) -> GeneralCone:
    N = len(Omega)
    zero = make_zero(N)
    D = [make_one_omega(i, N) for i in range(N)] + [a for a in A if not a == zero and not is_unit_gamble(a)]
    R = [D]
    return R


def partial_loss_cone(B: ConditionalLowerPrevisionAssessment, Omega: PossibilitySpace) -> GeneralCone:
    N = len(Omega)
    zero = make_zero(N)

    R1 = [[g, make_one(B1, N)] for (g, B1) in B if g != zero]
    R2 = [[make_one_omega(i, N)] for i in range(N)]
    R = R1 + R2  # TODO: remove duplicates
    return R


def natural_extension_cone(A: LowerPrevisionAssessment, Omega: PossibilitySpace) -> GeneralCone:
    N = len(Omega)
    R1 = [[g] for g in A]
    R2 = [[make_one_Omega(N)], [make_minus_one_Omega(N)], [make_zero(N)]]
    R3 = [[make_one_omega(i, N)] for i in range(N)]
    R = R1 + R2 + R3  # TODO: remove duplicates
    return R


def natural_extension_objective(R: GeneralCone, Omega: PossibilitySpace) -> List[List[Fraction]]:
    N = len(Omega)
    one_Omega = make_one_Omega(N)
    minus_one_Omega = make_minus_one_Omega(N)

    def a_value(D, g):
        if D == [one_Omega] and g == one_Omega:
            return Fraction(1)
        elif D == [minus_one_Omega] and g == minus_one_Omega:
            return Fraction(-1)
        return Fraction(0)

    a = [[a_value(D, g) for g in D] for D in R]

    return a


def incurs_sure_loss_cone(R: GeneralCone, Omega: PossibilitySpace) -> bool:
    N = len(Omega)
    zero = make_zero(N)
    mu = optimize_find(R, zero, [], Omega)
    return mu is not None


def incurs_sure_loss(P: LowerPrevisionFunction, Omega: PossibilitySpace, pretty=False) -> bool:
    A = lower_prevision_assessment(P)
    R = sure_loss_cone(A, Omega)
    if GlobalSettings.verbose:
        print(f'incurs_sure_loss: R =\n{print_general_cone(R, pretty)}\n')
    return incurs_sure_loss_cone(R, Omega)


def natural_extension(A: List[Gamble], f: Gamble, Omega: PossibilitySpace, pretty=False) -> Fraction:
    R = natural_extension_cone(A, Omega)
    a = natural_extension_objective(R, Omega)
    if GlobalSettings.verbose:
        print(f'natural_extension: R =\n{print_general_cone(R, pretty)}\n')
        print(f'natural_extension: a =\n{print_cone_generator(a, pretty)}\n')
    return optimize_maximize_value(R, f, a, [], Omega)


def is_coherent(P: LowerPrevisionFunction, Omega: PossibilitySpace, pretty=False) -> bool:
    A = lower_prevision_assessment(P)
    for (f, P_f) in P:
        n_f = natural_extension(A, f, Omega, pretty)
        if not P_f == n_f:
            if GlobalSettings.verbose:
                print(f'the is_coherent check failed for: f=({print_gamble(f, pretty)})  P(f)={P_f}  natural_extension(f) = {n_f}')
            return False
    return True
    # return all(P_f == natural_extension(A, f, Omega, pretty) for (f, P_f) in P)


def incurs_partial_loss(P: ConditionalLowerPrevisionFunction, Omega: PossibilitySpace, pretty=False) -> bool:
    N = len(P)
    zero = make_zero(N)
    B = conditional_lower_prevision_assessment(P, Omega)
    R = partial_loss_cone(B, Omega)
    if GlobalSettings.verbose:
        print(f'incurs_partial_loss: R = {print_general_cone(R, pretty)}\n')
    mu = optimize_find(R, zero, [], Omega)
    return mu is not None


def conditional_natural_extension_cone(B: ConditionalLowerPrevisionAssessment, C: Event, Omega: PossibilitySpace) -> GeneralCone:
    N = len(Omega)
    zero = make_zero(N)
    one_C = make_one(C, N)
    minus_one_C = make_minus_one(C, N)

    R1 = [[g, make_one(B1, N)] for (g, B1) in B]
    R2 = [[one_C], [minus_one_C], [zero]]
    R3 = [[make_one_omega(i, N) for i in range(N)]]
    R = R1 + R2 + R3  # TODO: remove duplicates

    return R


def conditional_natural_extension(B: ConditionalLowerPrevisionAssessment, f: Gamble, C: Event, Omega: PossibilitySpace, pretty=False) -> Fraction:
    def hadamard(f: Gamble, g: Gamble) -> Gamble:
        return [x * y for (x, y) in zip(f, g)]

    N = len(Omega)
    R = conditional_natural_extension_cone(B, C, Omega)
    a = natural_extension_objective(R, Omega)
    if GlobalSettings.verbose:
        print(f'conditional_natural_extension: R = {print_general_cone(R, pretty)}\n')
    return optimize_maximize_value(R, hadamard(f, make_one(C, N)), a, [], Omega)


def generate_lower_prevision_perturbation(K: List[Gamble], epsilon: Fraction) -> LowerPrevisionFunction:
    result = []
    for f in K:
        delta = Fraction(random.uniform(0, 1))
        value = random.choice([-epsilon, epsilon]) * delta * (max(f) - min(f))
        result.append((f, value))
    return result


def scale_lower_prevision_function(P: LowerPrevisionFunction, c: Fraction) -> LowerPrevisionFunction:
    return [(f, c * value) for (f, value) in P]


def lower_prevision_sum(P: LowerPrevisionFunction, Q: LowerPrevisionFunction) -> LowerPrevisionFunction:
    def same_domain(P, Q):
        return len(P) == len(Q) and all(p[0] == q[0] for (p, q) in zip(P, Q))

    assert same_domain(P, Q)
    return [(p[0], p[1] + q[1]) for (p, q) in zip(P, Q)]


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def lower_prevision_clamped_sum(P: LowerPrevisionFunction, Q: LowerPrevisionFunction) -> LowerPrevisionFunction:
    def same_domain(P, Q):
        return len(P) == len(Q) and all(p[0] == q[0] for (p, q) in zip(P, Q))

    assert same_domain(P, Q)
    return [(f, clamp(value_f + value_g, min(f), max(f))) for ((f, value_f), (g, value_g)) in zip(P, Q)]
