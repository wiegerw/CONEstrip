import inspect
import cdd

#------------------------------------------------------------------------#
#                     test_adjacency_list.py
#------------------------------------------------------------------------#

def test_make_vertex_adjacency_list(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    # The following lines test that poly.get_adjacency_list()
    # returns the correct adjacencies.

    # We start with the H-representation for a cube
    mat = cdd.Matrix([[1, 1, 0 ,0],
                      [1, 0, 1, 0],
                      [1, 0, 0, 1],
                      [1, -1, 0, 0],
                      [1, 0, -1, 0],
                      [1, 0, 0, -1]],
                     number_type=number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)

    poly = cdd.Polyhedron(mat)
    adjacency_list = poly.get_adjacency()

    # Family size should equal the number of vertices of the cube (8)
    assert len(adjacency_list) == 8

    # All the vertices of the cube should be connected by three other vertices
    assert [len(adj) for adj in adjacency_list] == [3]*8

    # The vertices must be numbered consistently
    # The first vertex is adjacent to the second, fourth and eighth
    # (note the conversion to a pythonic numbering system)
    adjacencies = [[1, 3, 7],
                   [0, 2, 6],
                   [1, 3, 4],
                   [0, 2, 5],
                   [2, 5, 6],
                   [3, 4, 7],
                   [1, 4, 7],
                   [0, 5, 6]]
    for i in range(8):
        assert list(adjacency_list[i]) == adjacencies[i]


def test_make_facet_adjacency_list(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    # This matrix is the same as in vtest_vo.ine
    mat = cdd.Matrix([[0, 0, 0, 1],
                      [5, -4, -2, 1],
                      [5, -2, -4, 1],
                      [16, -8, 0, 1],
                      [16, 0, -8, 1],
                      [32, -8, -8, 1]], number_type=number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)

    poly = cdd.Polyhedron(mat)

    adjacencies = [[1, 2, 3, 4, 6],
                   [0, 2, 3, 5],
                   [0, 1, 4, 5],
                   [0, 1, 5, 6],
                   [0, 2, 5, 6],
                   [1, 2, 3, 4, 6],
                   [0, 3, 4, 5]]

    adjacency_list = poly.get_input_adjacency()
    for i in range(7):
        assert list(adjacency_list[i]) == adjacencies[i]


#------------------------------------------------------------------------#
#                     test_incidence.py
#------------------------------------------------------------------------#

def test_vertex_incidence_cube(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    # The following lines test that poly.get_vertex_incidence()
    # returns the correct incidences.

    # We start with the H-representation for a cube
    mat = cdd.Matrix([[1, 1, 0 ,0],
                      [1, 0, 1, 0],
                      [1, 0, 0, 1],
                      [1, -1, 0, 0],
                      [1, 0, -1, 0],
                      [1, 0, 0, -1]],
                     number_type=number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)

    poly = cdd.Polyhedron(mat)
    incidence = poly.get_incidence()

    # Family size should equal the number of vertices of the cube (8)
    assert len(incidence) == 8

    # All the vertices of the cube should mark the incidence of 3 facets
    assert [len(inc) for inc in incidence] == [3]*8

    # The vertices must be numbered consistently
    # The first vertex is adjacent to the second, fourth and eighth
    # (note the conversion to a pythonic numbering system)
    incidence_list = [[1, 2, 3],
                      [1, 3, 5],
                      [3, 4, 5],
                      [2, 3, 4],
                      [0, 4, 5],
                      [0, 2, 4],
                      [0, 1, 5],
                      [0, 1, 2]]
    for i in range(8):
        assert sorted(list(incidence[i])) == incidence_list[i]


def test_vertex_incidence_vtest_vo(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    # This matrix is the same as in vtest_vo.ine
    mat = cdd.Matrix([[0, 0, 0, 1],
                      [5, -4, -2, 1],
                      [5, -2, -4, 1],
                      [16, -8, 0, 1],
                      [16, 0, -8, 1],
                      [32, -8, -8, 1]], number_type=number_type)

    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)

    poly = cdd.Polyhedron(mat)

    incidence_list = [[0, 4, 6],
                      [0, 2, 4],
                      [0, 1, 2],
                      [0, 1, 3],
                      [0, 3, 6],
                      [1, 2, 5],
                      [1, 3, 5],
                      [3, 5, 6],
                      [4, 5, 6],
                      [2, 4, 5]]

    incidence = poly.get_incidence()
    for i in range(10):
        assert sorted(list(incidence[i])) == incidence_list[i]


def test_facet_incidence_cube(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')

    # We start with the H-representation for a cube
    mat = cdd.Matrix([[1, 1, 0 ,0],
                      [1, 0, 1, 0],
                      [1, 0, 0, 1],
                      [1, -1, 0, 0],
                      [1, 0, -1, 0],
                      [1, 0, 0, -1]],
                     number_type=number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)

    poly = cdd.Polyhedron(mat)
    incidence = poly.get_input_incidence()

    # Family size should equal the number of facets of the cube (6), plus 1 (the empty infinite ray)
    assert len(incidence) == 7

    # All the facets of the cube should have 4 vertices.
    # The polyhedron is closed, so the last set should be empty
    assert [len(inc) for inc in incidence] == [4, 4, 4, 4, 4, 4, 0]

    # The vertices must be numbered consistently
    # The first vertex is adjacent to the second, fourth and eighth
    # (note the conversion to a pythonic numbering system)
    incidence_list = [[4, 5, 6, 7],
                      [0, 1, 6, 7],
                      [0, 3, 5, 7],
                      [0, 1, 2, 3],
                      [2, 3, 4, 5],
                      [1, 2, 4, 6],
                      []]
    for i in range(7):
        assert sorted(list(incidence[i])) == incidence_list[i]


def test_facet_incidence_vtest_vo(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    # This matrix is the same as in vtest_vo.ine
    mat = cdd.Matrix([[0, 0, 0, 1],
                      [5, -4, -2, 1],
                      [5, -2, -4, 1],
                      [16, -8, 0, 1],
                      [16, 0, -8, 1],
                      [32, -8, -8, 1]], number_type=number_type)

    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)

    poly = cdd.Polyhedron(mat)

    incidence_list = [[0, 1, 2, 3, 4],
                      [2, 3, 5, 6],
                      [1, 2, 5, 9],
                      [3, 4, 6, 7],
                      [0, 1, 8, 9],
                      [5, 6, 7, 8, 9],
                      [0, 4, 7, 8]]

    incidence = poly.get_input_incidence()
    for i in range(7):
        assert sorted(list(incidence[i])) == incidence_list[i]


#------------------------------------------------------------------------#
#                     test_linprog.py
#------------------------------------------------------------------------#

def test_lp2(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([['4/3',-2,-1],['2/3',0,-1],[0,1,0],[0,0,1]],
                     number_type=number_type)
    mat.obj_type = cdd.LPObjType.MAX
    mat.obj_func = (0,3,4)
    lp = cdd.LinProg(mat)
    lp.solve()
    assert lp.status == cdd.LPStatusType.OPTIMAL
    assert lp.obj_value == cdd.Fraction(11, 3)
    assert lp.primal_solution == (cdd.Fraction(1, 3), cdd.Fraction(2, 3))
    assert lp.dual_solution == (cdd.Fraction(3, 2), cdd.Fraction(5, 2))


def test_another(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[1,-1,-1,-1],[-1,1,1,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]], number_type=number_type)
    mat.obj_type = cdd.LPObjType.MIN
    mat.obj_func = (0,1,2,3)
    print(mat)

    lp = cdd.LinProg(mat)
    lp.solve()
    assert lp.obj_value == 1
    mat.obj_func = (0,-1,-2,-3)
    lp = cdd.LinProg(mat)
    lp.solve()
    assert lp.obj_value == -3
    mat.obj_func = (0,'1.12','1.2','1.3')
    lp = cdd.LinProg(mat)
    lp.solve()
    assert lp.obj_value == cdd.Fraction(28, 25)
    assert lp.primal_solution == (1, 0, 0)


#------------------------------------------------------------------------#
#                     test_polyhedron.py
#------------------------------------------------------------------------#

def test_sampleh1(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[2,-1,-1,0],[0,1,0,0],[0,0,1,0]],
                     number_type=number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)

    poly = cdd.Polyhedron(mat)

    ext = poly.get_generators()
    print(ext)

    assert ext.rep_type == cdd.RepType.GENERATOR
    assert list(ext) == [(1, 0, 0, 0), (1, 2, 0, 0), (1, 0, 2, 0), (0, 0, 0, 1)]
    # note: first row is 0, so fourth row is 3
    assert list(ext.lin_set) == [3]


def test_testcdd2(number_type):
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[7,-3,-0],[7,0,-3],[1,1,0],[1,0,1]], number_type=number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)

    assert list(mat) == [(7,-3,-0),(7,0,-3),(1,1,0),(1,0,1)]

    gen = cdd.Polyhedron(mat).get_generators()
    print(gen)

    assert gen.rep_type == cdd.RepType.GENERATOR
    assert list(gen) == [
         (1, cdd.Fraction(7, 3), -1),
         (1, -1, -1,),
         (1, -1, cdd.Fraction(7, 3)),
         (1, cdd.Fraction(7, 3), cdd.Fraction(7, 3))]

    # add an equality and an inequality
    mat.extend([[7, 1, -3]], linear=True)
    mat.extend([[7, -3, 1]])
    assert list(mat) == [(7,-3,-0),(7,0,-3),(1,1,0),(1,0,1),(7,1,-3),(7,-3,1)]
    assert list(mat.lin_set) == [4]
    gen2 = cdd.Polyhedron(mat).get_generators()
    assert gen2.rep_type == cdd.RepType.GENERATOR
    assert list(gen2) == [(1, -1, 2), (1, 0, cdd.Fraction(7, 3))]


#------------------------------------------------------------------------#
#                     documentation examples (Solving Linear Programs)
#------------------------------------------------------------------------#

def doc_example1():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([['4/3',-2,-1],['2/3',0,-1],[0,1,0],[0,0,1]], number_type='fraction')
    mat.obj_type = cdd.LPObjType.MAX
    mat.obj_func = (0,3,4)
    print(mat)
    print(mat.obj_func)
    lp = cdd.LinProg(mat)
    lp.solve()
    assert lp.status == cdd.LPStatusType.OPTIMAL
    print(lp.obj_value)
    print(" ".join("{0}".format(val) for val in lp.primal_solution))
    print(" ".join("{0}".format(val) for val in lp.dual_solution))


def doc_example2():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[2,-1,-1,0],[0,1,0,0],[0,0,1,0]], number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY
    print(mat)

    poly = cdd.Polyhedron(mat)
    print(poly)

    ext = poly.get_generators()
    print(ext)

    print(list(ext.lin_set)) # note: first row is 0, so fourth row is 3 [3]


def doc_example3():
    import cdd
    # We start with the H-representation for a square
    # 0 <= 1 + x1 (face 0)
    # 0 <= 1 + x2 (face 1)
    # 0 <= 1 - x1 (face 2)
    # 0 <= 1 - x2 (face 3)
    mat = cdd.Matrix([[1, 1, 0], [1, 0, 1], [1, -1, 0], [1, 0, -1]])
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    # The V-representation can be printed in the usual way:
    gen = poly.get_generators()
    print(gen)
    # V-representation
    # begin
    #  4 3 rational
    #  1 1 -1
    #  1 1 1
    #  1 -1 1
    #  1 -1 -1
    # end

    # graphical depiction of vertices and faces:
    #
    #   2---(3)---1
    #   |         |
    #   |         |
    #  (0)       (2)
    #   |         |
    #   |         |
    #   3---(1)---0
    #
    # vertex 0 is adjacent to vertices 1 and 3
    # vertex 1 is adjacent to vertices 0 and 2
    # vertex 2 is adjacent to vertices 1 and 3
    # vertex 3 is adjacent to vertices 0 and 2
    print([list(x) for x in poly.get_adjacency()])
    ## [[1, 3], [0, 2], [1, 3], [0, 2]]

    # vertex 0 is the intersection of faces (1) and (2)
    # vertex 1 is the intersection of faces (2) and (3)
    # vertex 2 is the intersection of faces (0) and (3)
    # vertex 3 is the intersection of faces (0) and (1)
    print([list(x) for x in poly.get_incidence()])
    ## [[1, 2], [2, 3], [0, 3], [0, 1]]

    # face (0) is adjacent to faces (1) and (3)
    # face (1) is adjacent to faces (0) and (2)
    # face (2) is adjacent to faces (1) and (3)
    # face (3) is adjacent to faces (0) and (2)
    print([list(x) for x in poly.get_input_adjacency()])
    ## v[[1, 3], [0, 2], [1, 3], [0, 2], []]

    # face (0) intersects with vertices 2 and 3
    # face (1) intersects with vertices 0 and 3
    # face (2) intersects with vertices 0 and 1
    # face (3) intersects with vertices 1 and 2
    print([list(x) for x in poly.get_input_incidence()])
    ## [[2, 3], [0, 3], [0, 1], [1, 2], []]

    # add a vertex, and construct new polyhedron
    gen.extend([[1, 0, 2]])
    vpoly = cdd.Polyhedron(gen)
    print(vpoly.get_inequalities())
    # H-representation
    # begin
    #  5 3 rational
    #  1 0 1
    #  2 1 -1
    #  1 1 0
    #  2 -1 -1
    #  1 -1 0
    # end

    # so now we have:
    # 0 <= 1 + x2
    # 0 <= 2 + x1 - x2
    # 0 <= 1 + x1
    # 0 <= 2 - x1 - x2
    # 0 <= 1 - x1
    #
    # graphical depiction of vertices and faces:
    #
    #        4
    #       / \
    #      /   \
    #    (1)   (3)
    #    /       \
    #   2         1
    #   |         |
    #   |         |
    #  (2)       (4)
    #   |         |
    #   |         |
    #   3---(0)---0
    #
    # for each face, list adjacent faces
    print([list(x) for x in vpoly.get_adjacency()])
    ## [[2, 4], [2, 3], [0, 1], [1, 4], [0, 3]]

    # for each face, list adjacent vertices
    print([list(x) for x in vpoly.get_incidence()])
    ## [[0, 3], [2, 4], [2, 3], [1, 4], [0, 1]]

    # for each vertex, list adjacent vertices
    print([list(x) for x in vpoly.get_input_adjacency()])
    ## [[1, 3], [0, 4], [3, 4], [0, 2], [1, 2]]

    # for each vertex, list adjacent faces
    print([list(x) for x in vpoly.get_input_incidence()])
    ## [[0, 4], [3, 4], [1, 2], [0, 2], [1, 3]]

test_make_vertex_adjacency_list("fraction")
test_make_facet_adjacency_list("fraction")
test_vertex_incidence_cube("fraction")
test_vertex_incidence_vtest_vo("fraction")
test_facet_incidence_cube("fraction")
test_facet_incidence_vtest_vo("fraction")
test_lp2("fraction")
test_another("fraction")
test_sampleh1("fraction")
test_testcdd2("fraction")
doc_example1()
doc_example2()
doc_example3()
