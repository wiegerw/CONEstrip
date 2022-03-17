# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import inspect
from conestrip.cones import parse_general_cone, parse_gamble, GeneralCone
from conestrip.conestrip import conestrip1_solution, conestrip2_solution, conestrip3_solution, conestrip
from conestrip.conestrip_cdd import conestrip_cdd_solution


def check_conestrip(R: GeneralCone, f_text: str, expected_result=False, verbose=True):
    f = parse_gamble(f_text)
    n = len(f)
    Omega_Gamma = list(range(n))
    Omega_Delta = list(range(n))
    # result1 = conestrip1_solution(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
    # result2 = conestrip2_solution(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
    # result3 = conestrip3_solution(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
    result4 = conestrip(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
    result5 = conestrip_cdd_solution(R, f, Omega_Gamma, Omega_Delta, verbose=verbose)
    # print('result1', result1)
    # print('result2', result2)
    # print('result3', result3)
    print('result4', result4)
    print('result5', result5)


def example_conestrip1():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    text = '''
      1 2
      2 1

      1 2
    '''
    R = parse_general_cone(text)
    # check_conestrip(R, '1 1', True)
    # check_conestrip(R, '10 10', True)
    # check_conestrip(R, '1 2', True)
    # check_conestrip(R, '2 4', True)
    check_conestrip(R, '2 1', False)
    # check_conestrip(R, '4 2', False)
    # check_conestrip(R, '1 3', False)
    # check_conestrip(R, '3 1', False)

example_conestrip1()
