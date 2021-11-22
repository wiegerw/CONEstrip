# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, List
from z3 import *


def product(a: Any, x: List[Any]) -> List[Any]:
    return [a * x_i for x_i in x]


def sum_rows(A: List[List[Any]]) -> List[Any]:
    m = len(A)
    n = len(A[0])
    return [simplify(sum(A[i][j] for i in range(m))) for j in range(n)]


def inner_product(x: List[Any], y: List[Any]) -> Any:
    assert(len(x) == len(y)), str(x) + str(y)
    return sum(x[i] * y[i] for i in range(len(x)))


