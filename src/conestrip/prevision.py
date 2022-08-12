# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from z3 import *
from conestrip.cones import *
from conestrip.utility import product, sum_rows


def calculate_lower_prevision(g0: List[Gamble], f0: Gamble, verbose: bool = False) -> Optional[float]:
    assert g0
    m = len(g0)
    n = len(g0[0])

    lambda_ = [Real(f'lambda{i}') for i in range(m)]
    f = [RealVal(f0[j]) for j in range(n)]
    g = [[RealVal(g0[i][j]) for j in range(n)] for i in range(m)]
    alpha = Real('alpha')

    # lambda >= 0
    lambda_constraints = [ 0 <= lambda_[i] for i in range(m)]

    # h = lambda_g * g
    h = sum_rows([product(lambda_[i], g[i]) for i in range(m)])

    # f - alpha >= lambda_g * g
    main_constraints = [f[j] - alpha >= h[j] for j in range(n)]

    if verbose:
        print('--- constraints ---')
        print(lambda_constraints)
        print(main_constraints)

    optimizer = Optimize()
    optimizer.add(lambda_constraints + main_constraints)
    optimizer.maximize(alpha)
    if optimizer.check() == sat:
        model = optimizer.model()
        if verbose:
            print('--- solution ---')
            print('alpha =', model.evaluate(alpha))
            print('lambda =', [model.evaluate(lambda_[i]) for i in range(m)])
            print('lambda * g =', [model.evaluate(h[i]) for i in range(m)])
        return model.evaluate(alpha)
    else:
        print("failed to solve")
        return None


def calculate_lower_prevision_with_slack(g0: List[Gamble], f0: Gamble, verbose: bool = False) -> Optional[float]:
    assert g0
    m = len(g0)
    n = len(g0[0])

    lambda_ = [Real(f'lambda{i}') for i in range(m)]
    mu = [Real(f'mu{j}') for j in range(n)]
    f = [RealVal(f0[j]) for j in range(n)]
    g = [[RealVal(g0[i][j]) for j in range(n)] for i in range(m)]
    alpha = Real('alpha')

    # lambda >= 0
    lambda_constraints = [0 <= lambda_[i] for i in range(m)]

    # mu >= 0
    mu_constraints = [0 <= mu[j] for j in range(n)]

    # h = lambda_g * g
    h = sum_rows([product(lambda_[i], g[i]) for i in range(m)])

    # lambda_g * g + mu + alpha = f
    main_constraints = [h[j] + mu[j] + alpha == f[j] for j in range(n)]

    if verbose:
        print('--- constraints ---')
        print(lambda_constraints)
        print(mu_constraints)
        print(main_constraints)

    optimizer = Optimize()
    optimizer.add(lambda_constraints + mu_constraints + main_constraints)
    optimizer.maximize(alpha)
    if optimizer.check() == sat:
        model = optimizer.model()
        if verbose:
            print('--- solution ---')
            print('alpha =', model.evaluate(alpha))
            print('lambda =', [model.evaluate(lambda_[i]) for i in range(m)])
            print('mu =', [model.evaluate(mu[j]) for j in range(n)])
            print('lambda * g =', [model.evaluate(h[i]) for i in range(m)])
        return model.evaluate(alpha)
    else:
        print("failed to solve")
        return None
