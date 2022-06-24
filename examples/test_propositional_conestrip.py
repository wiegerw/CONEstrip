#!/usr/bin/env python3

# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from conestrip.cones import parse_cone_generator, parse_gamble
from conestrip.conestrip import is_in_cone_generator
from conestrip.propositional_algorithms import is_in_propositional_cone_generator


def test1():
    r = parse_cone_generator('-5 9')
    x3 = parse_gamble('-2 8')

    assert not is_in_cone_generator(r, x3)
    assert not is_in_propositional_cone_generator(r, x3, verbose=False)


def test2():
    r = parse_cone_generator('3 -4')
    x3 = parse_gamble('13 -14')

    assert not is_in_cone_generator(r, x3)
    assert not is_in_propositional_cone_generator(r, x3, verbose=True)


if __name__ == '__main__':
    test1()
    test2()
