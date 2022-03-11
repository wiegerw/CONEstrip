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
from conestrip.conestrip_cdd import conestrip_cdd_solution


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
    result1 = conestrip_cdd_solution(R, f, Omega_Gamma, Omega_Delta)
    # print('result1:', result1)


example_conestrip1()
