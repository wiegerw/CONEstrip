import inspect
from typing import List
import cdd
from conestrip.polyhedron import Polyhedron
from conestrip.gambles import parse_cone, Gamble


def gambles_to_polyhedron(gambles: List[Gamble]) -> Polyhedron:
    n = len(gambles[0])
    origin = [0.0] * n
    A = [origin] + gambles
    A = [[1.0] + x for x in A]
    # print('A=', A)
    mat = cdd.Matrix(A)
    mat.rep_type = cdd.RepType.GENERATOR
    mat.canonicalize()
    poly = Polyhedron(mat)
    poly.to_V()
    return poly


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

    faces = [make_face(face) for face in poly.face_vertex_adjacencies() if 0 in face]

    print(faces)


example1()
