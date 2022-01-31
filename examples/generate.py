import inspect
from typing import List
from conestrip.gambles import parse_cone, Gamble
from conestrip.utility import pretty_print
from conestrip.make_test_cases import gambles_to_polyhedron, random_border_cone, add_random_border_cones


def example1():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_cone('''
      4 0 0
      0 5 0
      0 0 6
      1 1 1
    ''')

    poly = gambles_to_polyhedron(R[0])
    # poly.info()

    vertices = poly.vertices()

    # converts indices to points
    def make_face(indices: List[int]) -> List[Gamble]:
        return [vertices[i] for i in indices]

    faces = [make_face(face) for face in poly.face_vertex_adjacencies()]

    print(faces)


def example2():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_cone('''
      4 0 0
      0 5 0
      0 0 6
      1 1 1
    ''')

    cone = random_border_cone(R[0])
    pretty_print([cone])


def example3():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    R = parse_cone('''
      4 0 0
      0 5 0
      0 0 6
      1 1 1
    ''')

    add_random_border_cones(R, 5)
    print(pretty_print(R))

example1()
example2()
example3()
