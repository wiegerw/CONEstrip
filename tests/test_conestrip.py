# Copyright 2021 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from conestrip.cone import parse_cone

class Test(TestCase):
    def stub(self):
        R = parse_cone('''
          1 0 0
          0 1 0
          0 0 1
          1 0 1
          0 1 1
          1 1 0
        ''')

        for g in R[0]:
            print(g)


        self.assertTrue(True)


if __name__ == '__main__':
    import unittest
    unittest.main()
