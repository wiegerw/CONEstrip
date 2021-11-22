# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from typing import List

Gamble = List[float]

Cone = List[List[Gamble]]


def parse_gamble(text: str) -> Gamble:
    return list(map(float, text.strip().split()))


def parse_gambles(text: str) -> List[Gamble]:
    return list(map(parse_gamble, text.strip().split('\n')))


def parse_cone(text: str) -> Cone:
    return list(map(parse_gambles, re.split(r'\n\s*\n', text.strip())))
