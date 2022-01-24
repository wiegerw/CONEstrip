import cdd


# prints some information about a polyhedron
def info(poly):
    print('- list the inequalities')
    print([list(x) for x in poly.get_inequalities()])

    print('- list the generators')
    print([list(x) for x in poly.get_generators()])

    print('- for each face, list adjacent faces')
    print([list(x) for x in poly.get_adjacency()])

    print('- for each face, list adjacent vertices')
    print([list(x) for x in poly.get_incidence()])

    print('- for each vertex, list adjacent vertices')
    print([list(x) for x in poly.get_input_adjacency()])

    print('- for each vertex, list adjacent faces')
    print([list(x) for x in poly.get_input_incidence()])


# Example 1: consider a rectangle with corners (0,0), (3,0), (3,2) and (0,2). It can be defined using the equations:
#
# x1  >= 0
# x1  <= 3
# x2  >= 0
# x2  <= 2
# x2  <= 3    N.B. This equation is redundant!
#
def example():
    mat = cdd.Matrix([[0,  1,  0],
                      [3, -1,  0],
                      [0,  0,  1],
                      [2,  0, -1],
                     ],
                     number_type="fraction")
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    info(poly)


example()
