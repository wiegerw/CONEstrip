# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import inspect
import cdd
from conestrip.polyhedron import Polyhedron


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


example1()
example2()
example3()
