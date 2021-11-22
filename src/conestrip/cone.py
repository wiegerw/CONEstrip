# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, Optional, Tuple
from more_itertools import collapse
from more_itertools.recipes import flatten
from z3 import *
from gambles import *
from utility import product, sum_rows


def conestrip1(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> bool:
    """
    An implementation of formula (1) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """

    # check that Omega_Gamma and Omega_Delta are a partition of { 0, ..., |f0|-1 }
    assert(set(Omega_Gamma).isdisjoint(set(Omega_Delta)))
    assert(set(list(range(len(f0)))) == set(Omega_Gamma) | set(Omega_Delta))
    # check that the gambles in R0 have the correct length
    assert(all(len(x) == len(f0) for x in flatten(R0)))

    # variables
    lambda_ = [Real(f'lambda{k}') for k in range(len(R0))]
    nu = [[Real(f'nu{k}_{i}') for i in range(len(R0[k]))] for k in range(len(R0))]

    # constants
    g = [[[RealVal(R0[k][i][j]) for j in range(len(R0[k][i]))] for i in range(len(R0[k]))] for k in range(len(R0))]
    f = [RealVal(f0[j]) for j in range(len(f0))]

    # intermediate expressions
    h = sum_rows(list(product(lambda_[k], sum_rows([product(nu[k][i], g[k][i]) for i in range(len(R0[k]))])) for k in range(len(R0))))

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
            print('lambda =', [model.evaluate(lambda_[k]) for k in range(len(R0))])
            print('nu =', [[model.evaluate(nu[k][i]) for i in range(len(R0[k]))] for k in range(len(R0))])
        return True
    else:
        return False


def conestrip2(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> bool:
    """
    An implementation of formula (2) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    """

    # check that Omega_Gamma and Omega_Delta are a partition of { 0, ..., |f0|-1 }
    assert(set(Omega_Gamma).isdisjoint(set(Omega_Delta)))
    assert(set(list(range(len(f0)))) == set(Omega_Gamma) | set(Omega_Delta))
    # check that the gambles in R0 have the correct length
    assert(all(len(x) == len(f0) for x in flatten(R0)))

    # variables
    lambda_ = [Real(f'lambda{k}') for k in range(len(R0))]
    tau = [[Real(f'tau{k}_{i}') for i in range(len(R0[k]))] for k in range(len(R0))]
    sigma = Real('sigma')

    # constants
    g = [[[RealVal(R0[k][i][j]) for j in range(len(R0[k][i]))] for i in range(len(R0[k]))] for k in range(len(R0))]
    f = [RealVal(f0[j]) for j in range(len(f0))]

    # intermediate expressions
    h = sum_rows(list(product(lambda_[k], sum_rows([product(tau[k][i], g[k][i]) for i in range(len(R0[k]))])) for k in range(len(R0))))

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
            print('lambda =', [model.evaluate(lambda_[k]) for k in range(len(R0))])
            print('tau =', [[model.evaluate(tau[k][i]) for i in range(len(R0[k]))] for k in range(len(R0))])
            print('sigma =', model.evaluate(sigma))
        return True
    else:
        return False


def conestrip3(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> bool:
    """
    An implementation of formula (3) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
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
    constraints_4 = list(collapse([[And(lambda_[k] <= mu[k][i], mu[k][i] <= lambda_[k] * mu[k][i]) for i in range(len(R0[k]))] for k in range(len(R0))]))

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
            print('lambda =', [model.evaluate(lambda_[k]) for k in range(len(R0))])
            print('mu =', [[model.evaluate(mu[k][i]) for i in range(len(R0[k]))] for k in range(len(R0))])
            print('sigma =', model.evaluate(sigma))
        return True
    else:
        return False


def conestrip4(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
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
    h = sum_rows(list(product(lambda_[k], sum_rows([product(mu[k][i], g[k][i]) for i in range(len(R0[k]))])) for k in range(len(R0))))
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
    optimizer.maximize(simplify(sum(lambda_)))
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


def conestrip4a(R0: Cone, f0: Gamble, Omega_Gamma: List[int], Omega_Delta: List[int], verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    """
    An implementation of formula (4) in 'A Propositional CONEstrip Algorithm', IPMU 2014.
    Instead of maximize, an optimal value is found using a loop.
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
    h = sum_rows(list(product(lambda_[k], sum_rows([product(mu[k][i], g[k][i]) for i in range(len(R0[k]))])) for k in range(len(R0))))
    goal = simplify(sum(lambda_))

    # 0 <= lambda <= 1
    lambda_constraints = [0 <= x for x in lambda_] + [x <= 1 for x in lambda_]

    # tau >= 1
    mu_constraints = [x >= 0 for x in collapse(mu)]

    # sigma >= 1
    sigma_constraints = [sigma >= 1]

    # main constraints
    constraints_2 = [h[omega] <= sigma * f[omega] for omega in Omega_Gamma]
    constraints_3 = [h[omega] >= sigma * f[omega] for omega in Omega_Delta]
    constraints_4 = list(collapse([[lambda_[k] <= mu[k][i] for i in range(len(R0[k]))] for k in range(len(R0))]))
    constraints = lambda_constraints + mu_constraints + sigma_constraints + constraints_2 + constraints_3 + constraints_4

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
        print(constraints_2)
        print(constraints_3)
        print(constraints_4)

    # Solve the problem with additional constraint 'goal >= value')
    def solve(value: float) -> Optional[Tuple[Any, Any, Any, Any]]:
        constraints_1 = [goal >= value]
        # print('solve', constraints_1)

        solver = Solver()
        solver.add(constraints + constraints_1)
        if solver.check() == sat:
            model = solver.model()
            lambda_solution = [model.evaluate(lambda_[k]) for k in range(len(R0))]
            mu_solution = [[model.evaluate(mu[k][i]) for i in range(len(R0[k]))] for k in range(len(R0))]
            sigma_solution = model.evaluate(sigma)
            goal_solution = model.evaluate(goal)
            # print('goal =', goal_solution)
            return goal_solution, lambda_solution, mu_solution, sigma_solution
        else:
            return None

    def as_float(value: Any) -> float:
        if isinstance(value, int):
            return float(value)
        assert isinstance(value, RatNumRef)
        return value.numerator_as_long() / value.denominator_as_long()

    min_value = RealVal(1)
    max_value = RealVal(len(lambda_))  # the maximum possible value of goal
    print(f'interval: [{min_value.as_decimal(50), max_value.as_decimal(50)}]')
    solution = solve(1)
    if not solution:
        return None
    min_value = solution[0]
    while as_float(min_value) < as_float(max_value):
        print(f'interval: [{min_value.as_decimal(50), max_value.as_decimal(50)}]')
        value = (min_value + max_value) / 2
        sol = solve(value)
        if sol:
            min_value = sol[0]
            solution = sol
        else:
            max_value = sol[0]

    if verbose:
        print('--- solution ---')
        print('goal =', min_value.as_decimal(50))
        print('lambda =', solution[1])
        print('mu =', solution[2])
        print('sigma =', solution[3])
    return solution[1], solution[2], solution[3],


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

    print('==================')
    print('=== conestrip4a ===')
    print('==================')
    lambda_solution, mu_solution, sigma_solution = conestrip4a(R, f, Omega_Gamma, Omega_Delta, verbose=True)
    # print('lambda =', lambda_solution)
    # print('mu =', mu_solution)
    # print('sigma =', sigma_solution)
