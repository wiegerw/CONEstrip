# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from z3 import *
from conestrip.cones import *
from conestrip.utility import product, sum_rows


def avoids_sure_loss(g0: List[Gamble], verbose: bool = False) -> bool:
    assert g0
    m = len(g0)
    n = len(g0[0])

    lambda_ = [Real(f'lambda{i}') for i in range(m)]
    g = [[RealVal(g0[i][j]) for j in range(n)] for i in range(m)]

    # lambda >= 0
    lambda_constraints = [0 <= lambda_[i] for i in range(m)]

    # h = lambda_g * g
    h = sum_rows([product(lambda_[i], g[i]) for i in range(m)])

    # lambda_g * g <= -1
    main_constraints = [h[j] <= -1 for j in range(n)]

    if verbose:
        print('--- constraints ---')
        print(lambda_constraints)
        print(main_constraints)

    solver = Solver()
    solver.add(lambda_constraints + main_constraints)
    if solver.check() == sat:
        model = solver.model()
        if verbose:
            print('--- solution ---')
            print('lambda =', [model.evaluate(lambda_[i]) for i in range(m)])
            print('lambda * g =', [model.evaluate(h[i]) for i in range(m)])
        return False
    else:
        print("failed to solve")
        return True


def avoids_sure_loss_with_slack(g0: List[Gamble], verbose: bool = False) -> bool:
    assert g0
    m = len(g0)
    n = len(g0[0])

    lambda_ = [Real(f'lambda{i}') for i in range(m)]
    mu = [Real(f'mu{j}') for j in range(n)]

    g = [[RealVal(g0[i][j]) for j in range(n)] for i in range(m)]

    # lambda >= 0
    lambda_constraints = [ 0 <= lambda_[i] for i in range(m)]
    if verbose:
        print(lambda_constraints)

    # mu >= 1
    mu_constraints = [ 1 <= mu[j] for j in range(n)]

    # h = lambda_g * g
    h = sum_rows([product(lambda_[i], g[i]) for i in range(m)])

    # lambda_g * g + mu == 0
    main_constraints = [h[j] + mu[j] == 0 for j in range(n)]

    if verbose:
        print('--- constraints ---')
        print(lambda_constraints)
        print(mu_constraints)
        print(main_constraints)

    solver = Solver()
    solver.add(lambda_constraints + mu_constraints + main_constraints)
    if solver.check() == sat:
        model = solver.model()
        if verbose:
            print('--- solution ---')
            print('lambda =', [model.evaluate(lambda_[i]) for i in range(m)])
            print('mu =', [model.evaluate(mu[j]) for j in range(n)])
            print('lambda * g =', [model.evaluate(h[i]) for i in range(m)])
        return False
    else:
        print("failed to solve")
        return True
