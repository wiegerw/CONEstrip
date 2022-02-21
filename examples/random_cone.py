# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from z3 import simplify
from conestrip.cones import print_gamble, linear_combination
from conestrip.random_cones import add_random_border_cones, random_border_point, random_inside_point, random_general_cone
from conestrip.conestrip import is_in_general_cone, is_in_cone_generator, is_in_cone_generator_border, random_between_point

for _ in range(1000):
    cone_size = 1
    dimension = 3
    generator_size = 3
    bound = 10
    R = random_general_cone(cone_size, dimension, generator_size, bound)
    add_random_border_cones(R, 2, False)

    for r in R.generators:
        if not r.parent:
            continue
        r_parent, _ = r.parent

        print('====================================================================================')
        print('r_parent =\n', r_parent, '\n')
        print('r =\n', r, '\n')
        x1, lambda1 = random_inside_point(r)
        x2, lambda2 = random_border_point(r)
        x3, lambda3 = random_between_point(r)
        print('x1 =', print_gamble(x1), 'lambda =', print_gamble(lambda1))
        print('x2 =', print_gamble(x2), 'lambda =', print_gamble(lambda2))
        print('x3 =', print_gamble(x3), 'lambda =', print_gamble(lambda3))
        assert x1 == linear_combination(lambda1, r.vertices)
        assert x2 == linear_combination(lambda2, r.vertices)
        assert x3 == [simplify(x) for x in linear_combination(lambda3, r_parent.vertices)]
        assert is_in_cone_generator(r, x1)
        assert is_in_cone_generator_border(r, x2)
        assert not is_in_cone_generator(r, x3)
        assert not is_in_cone_generator(r, x3, with_border=True)
        assert is_in_cone_generator(r_parent, x3)
        assert is_in_general_cone(R, x1)
        print()
