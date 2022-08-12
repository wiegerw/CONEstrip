# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import inspect
from typing import List
from conestrip.cones import Gamble, gambles_to_polyhedron
from conestrip.extended_cones import parse_extended_general_cone
from conestrip.random_extended_cones import add_random_border_cone_extended, add_random_border_cones_extended


def example1():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_extended_general_cone('''
      4 0 0
      0 5 0
      0 0 6
      1 1 1
    ''')

    poly = gambles_to_polyhedron(R.generators[0].gambles)
    vertices = poly.vertices()

    # converts indices to points
    def make_face(indices: List[int]) -> List[Gamble]:
        return [vertices[i] for i in indices]

    faces = [make_face(face) for face in poly.face_vertex_adjacencies()]

    print(faces)


def example2():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_extended_general_cone('''
      4 0 0
      0 5 0
      0 0 6
      1 1 1
    ''')

    generator = R.generators[0]
    print(f'generator:\n{generator}')
    generator1 = add_random_border_cone_extended(generator)
    print(f'generator1:\n{generator1}')


def example3():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_extended_general_cone('''
      4 0 0
      0 5 0
      0 0 6
      1 1 1
    ''')

    add_random_border_cones_extended(R, 5)
    print(R)


if __name__ == '__main__':
    example1()
    example2()
    example3()
