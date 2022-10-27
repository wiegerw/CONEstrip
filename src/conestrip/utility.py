# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import math
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
    return sum(xi * yi for xi, yi in zip(x, y))


def print_list(x: List[Any]) -> str:
    return f"[{', '.join(str(xi) for xi in x)}]"


def print_list_list(x: List[Any]) -> str:
    return f"[{', '.join(print_list(xi) for xi in x)}]"


def random_floats_summing_to_one(n: int) -> List[float]:
    values = [random.random() for _ in range(n)]
    s = sum(values)
    return [x / s for x in values]


def random_rationals_summing_to_one(n: int, N=1000) -> List[Fraction]:
    values = random_floats_summing_to_one(n)
    v = [Fraction(int(round(N * x)), N).limit_denominator() for x in values]
    v = v[:-1]
    v.append((Fraction(1) - sum(v)).limit_denominator())
    assert sum(v) == 1, f'random_rationals_summing_to_one: the sum of {v} equals {sum(v)}'
    return v


def random_nonzero_rationals_summing_to_one(n: int) -> List[Fraction]:
    values = random_floats_summing_to_one(n)
    v = [Fraction(max(1, round(1000*x)), 1000) for x in values]
    assert all(x > 0 for x in v)
    s = sum(v) - 1
    I = [i for i in range(n) if v[i] > s]
    i0 = random.choice(I)
    v[i0] -= s
    v = [Fraction(vi) for vi in v]
    assert sum(v) == 1
    return v


def remove_spaces(txt: str) -> str:
    lines = txt.strip().split('\n')
    return '\n'.join(line.strip() for line in lines)


class StopWatch(object):
    def __init__(self):
        import time
        self.start = time.perf_counter()

    def seconds(self):
        import time
        end = time.perf_counter()
        return end - self.start

    def restart(self):
        import time
        self.start = time.perf_counter()


def is_power_of_two(n):
    m = int(math.log2(n))
    return 2**m == n


def is_solved(solution):
    return None not in solution
