# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import inspect
from typing import List
from conestrip.cones import parse_general_cone, Gamble, gambles_to_polyhedron, find_generator_dependencies
from conestrip.random_cones import add_random_border_cone, add_random_border_cones


def example1():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_general_cone('''
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

    R = parse_general_cone('''
      4 0 0
      0 5 0
      0 0 6
      1 1 1
    ''')

    generator = R.generators[0]
    print(f'generator:\n{generator}')
    generator1 = add_random_border_cone(generator)
    print(f'generator1:\n{generator1}')


def example3():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_general_cone('''
      4 0 0
      0 5 0
      0 0 6
      1 1 1
    ''')

    add_random_border_cones(R, 5)
    print(R)

    dependencies = find_generator_dependencies(R)
    print(dependencies)


example1()
example2()
example3()
