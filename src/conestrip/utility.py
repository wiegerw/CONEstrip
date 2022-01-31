# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
from fractions import Fraction
from typing import Any, List
from z3 import simplify


def product(a: Any, x: List[Any]) -> List[Any]:
    return [a * x_i for x_i in x]


def sum_rows(A: List[List[Any]]) -> List[Any]:
    m = len(A)
    n = len(A[0])
    return [simplify(sum(A[i][j] for i in range(m))) for j in range(n)]


def inner_product(x: List[Any], y: List[Any]) -> Any:
    assert(len(x) == len(y)), str(x) + str(y)
    return sum(x[i] * y[i] for i in range(len(x)))


def pretty_print(x) -> str:
    if isinstance(x, list):
        if x and not isinstance(x[0], list):
            return f"[{', '.join(map(str, x))}]"
        return f"[{', '.join(map(pretty_print, x))}]"
    return str(x)


def random_floats_summing_to_one(n: int) -> List[float]:
    values = [random.random() for i in range(n)]
    s = sum(values)
    return [x / s for x in values]


def random_rationals_summing_to_one(n: int) -> List[Fraction]:
    values = random_floats_summing_to_one(n)
    v = [int(round(1000*x)) / 1000 for x in values]
    v = v[:-1]
    v.append(1 - sum(v))
    return [Fraction(vi) for vi in v]
