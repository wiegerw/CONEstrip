# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import inspect
import cdd
from conestrip.cones import parse_general_cone, parse_gamble, parse_cone_generator
from conestrip.conestrip import conestrip1_solution, conestrip2_solution, conestrip3_solution, conestrip
from conestrip.polyhedron import Polyhedron
from conestrip.prevision import calculate_lower_prevision, calculate_lower_prevision_with_slack
from conestrip.sure_loss import avoids_sure_loss, avoids_sure_loss_with_slack


# Example 1: consider a rectangle with corners (0,0), (3,0), (3,2) and (0,2). It can be defined using the equations:
#
# x1  >= 0
# x1  <= 3
# x2  >= 0
# x2  <= 2
# x2  <= 3    N.B. This equation is redundant!
#
def example1():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[0,  1,  0],
                      [3, -1,  0],
                      [0,  0,  1],
                      [2,  0, -1],
                      [3,  0, -1],
                     ],
                     number_type="fraction")
    mat.rep_type = cdd.RepType.INEQUALITY

    print('- matrix before canonicalization')
    print(mat, '\n')
    mat.canonicalize()
    print('- matrix after canonicalization')
    print(mat, '\n')

    print('==================================================')
    print('=               H-representation')
    print('==================================================')
    poly = Polyhedron(mat)
    poly.info()
    assert poly.is_H()

    print('==================================================')
    print('=               V-representation')
    print('==================================================')
    poly.to_V()
    assert poly.is_V()
    poly.info()

# triangle (2,1), (4,1) and (2,2)
# x1 >= 2
# x2 >= 1
# x1 + 2 x2 <= 6
#
# rewrite that into
#
# -x1 <= -2
# -x2 <= -1
# x1 + 2 x2 <= 6
def example2():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[-2,  1,  0],
                      [-1,  0,  1],
                      [ 6,  -1, -2]],
                     number_type="fraction")
    mat.rep_type = cdd.RepType.INEQUALITY

    print('--- matrix ---')
    print(mat)

    poly = Polyhedron(mat)
    poly.info()


def example3():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[-2,  1,  0],
                      [-1,  0,  1],
                      [ 6,  -1, -2]],
                     number_type="fraction")
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    generators = poly.get_generators()
    print(generators.__class__)


def example_conestrip1():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_general_cone('''
      4 0 0
      0 5 0
      0 0 6

      1 0 1
      0 7 7

      1 2 3
      2 4 6
    ''')
    f = parse_gamble('2 5 8')
    Omega_Gamma = [0, 1, 2]
    Omega_Delta = [0, 1, 2]
    result1 = conestrip1_solution(R, f, Omega_Gamma, Omega_Delta, verbose=True)
    result2 = conestrip2_solution(R, f, Omega_Gamma, Omega_Delta, verbose=True)
    result3 = conestrip3_solution(R, f, Omega_Gamma, Omega_Delta, verbose=True)

    print('result1:', result1)
    print('result2:', result2)
    print('result3:', result3)


def example_conestrip2():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_general_cone('''
      4 0 0
      0 5 0
      0 0 6

      1 0 1
      0 7 7

      1 2 3
      2 4 6
    ''')
    f = parse_gamble('2 5 8')
    Omega_Gamma = [0, 1, 2]
    Omega_Delta = [0, 1, 2]

    lambda_solution, mu_solution, sigma_solution = conestrip(R, f, Omega_Gamma, Omega_Delta)
    print('lambda =', lambda_solution)
    print('mu =', mu_solution)
    print('sigma =', sigma_solution)


def example_sure_loss():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    generator = parse_cone_generator('''
      1 0 -3
      -2 1 1
      1 -4 1
    ''')
    result1 = avoids_sure_loss(generator.gambles, verbose=True)
    result2 = avoids_sure_loss_with_slack(generator.gambles, verbose=True)
    print(result1, result2)


def example_prevision():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    generator = parse_cone_generator('''
      1 0
      0 1
    ''')
    f = parse_gamble('2 5')
    alpha1 = calculate_lower_prevision(generator.gambles, f, verbose=True)
    alpha2 = calculate_lower_prevision_with_slack(generator.gambles, f, verbose=True)
    print(alpha1, alpha2)


example1()
example2()
example3()
example_conestrip1()
example_conestrip2()
example_sure_loss()
