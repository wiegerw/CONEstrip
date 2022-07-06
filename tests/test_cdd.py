#!/usr/bin/env python3

# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
import cdd


class Test(TestCase):
    def test_solve1(self):
        # x1 <= 3
        # x2 = 2
        # x1 + 2 x2 <= 6
        mat = cdd.Matrix([[3, -1, 0],
                          [2, 0, -1],
                          [6, -1, -2]],
                         number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
        mat.obj_type = cdd.LPObjType.MIN
        mat.lin_set = {1}
        lp = cdd.LinProg(mat)
        lp.solve()
        self.assertEqual(lp.status, cdd.LPStatusType.OPTIMAL)
        self.assertEqual((2, 2), lp.primal_solution)  # N.B. This is not the only solution

    def test_solve2(self):
        # x1 <= 1
        # x1 + x2 = 2
        # 3 x1 + 2 x2 <= 5
        mat = cdd.Matrix([[1, -1, 0],
                          [2, -1, -1],
                          [5, -3, -2]],
                         number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
        mat.obj_type = cdd.LPObjType.MAX
        mat.lin_set = {1}
        lp = cdd.LinProg(mat)
        lp.solve()
        self.assertEqual(lp.status, cdd.LPStatusType.OPTIMAL)
        self.assertEqual((1, 1), lp.primal_solution)  # N.B. This is not the only solution


if __name__ == '__main__':
    import unittest
    unittest.main()
