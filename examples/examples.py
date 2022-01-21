import cdd
import inspect

# Example 1: consider a rectangle with corners (0,0), (3,0), (3,2) and (0,2). It can be defined using the equations:
#
# -x1 <= 0
# x1  <= 3
# -x2 <= 0
# x2  <= 2
#
# The corresponding cdd matrix is therefore:
#
# 0  1  0
# 3 -1  0
# 0  0  1
# 2  0 -1
#
def example1():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[0,  1,  0],
                      [3, -1,  0],
                      [0,  0,  1],
                      [2,  0, -1]],
                     number_type="fraction")

    poly = cdd.Polyhedron(mat)

    generators = poly.get_generators()     # the V-representation
    # For a polyhedron described as P = conv(v_1, …, v_n) + nonneg(r_1, …, r_s), the V-representation matrix is [t V]
    #where t is the column vector with n ones followed by s zeroes, and V is the stacked matrix of n vertex row vectors
    # on top of s ray row vectors.

    inequalities = poly.get_inequalities() # the H-representation

    adjacency_list = poly.get_adjacency()

    print(mat)
    print('--- generators ---')
    print(generators)
    print('--- inequalities ---')
    print(inequalities)
    print('--- adjacancy list ---')
    print(adjacency_list)


# triangle (2,1), (4,1) and (2,2)
# x1 >= 2
# x2 >= 1
# x1 + 2 x2 <= 6
#
# rewrite that into
#
# -x1 <= -2
# -x2 <= -1
# x1 + 2 x2 <= 6
def example2():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[-2,  1,  0],
                      [-1,  0,  1],
                      [ 6,  -1, -2]],
                     number_type="fraction")
    mat.rep_type = cdd.RepType.INEQUALITY

    print('--- matrix ---')
    print(mat)

    poly = cdd.Polyhedron(mat)

    generators = poly.get_generators()     # the V-representation (but it is printed as H-representation!)
    # For a polyhedron described as P = conv(v_1, …, v_n) + nonneg(r_1, …, r_s), the V-representation matrix is [t V]
    #where t is the column vector with n ones followed by s zeroes, and V is the stacked matrix of n vertex row vectors
    # on top of s ray row vectors.
    print('--- generators (contains the vertices) ---')
    print(generators)

    inequalities = poly.get_inequalities() # the H-representation
    print('--- inequalities (contains the equations) ---')
    print(inequalities)

    adjacency_list = poly.get_adjacency()
    print('--- adjacancy list ---')
    print(adjacency_list)


# Example 1: consider a rectangle with corners (0,0), (3,0), (3,2) and (0,2). It can be defined using the equations:
#
# -x1 <= 0
# x1  <= 3
# -x2 <= 0
# x2  <= 2
#
# We add one redundant equation:
#
# x2 <= 3
#
# The corresponding cdd matrix is therefore:
#
# 0  1  0
# 3 -1  0
# 0  0  1
# 2  0 -1
# 3  0 -1
#
def example3():
    print(f'--- {inspect.currentframe().f_code.co_name} ---')
    mat = cdd.Matrix([[0,  1,  0],
                      [3, -1,  0],
                      [0,  0,  1],
                      [2,  0, -1],
                      [3,  0, -1],
                     ],
                     number_type="fraction")

    poly = cdd.Polyhedron(mat)

    generators = poly.get_generators()     # the V-representation
    # For a polyhedron described as P = conv(v_1, …, v_n) + nonneg(r_1, …, r_s), the V-representation matrix is [t V]
    #where t is the column vector with n ones followed by s zeroes, and V is the stacked matrix of n vertex row vectors
    # on top of s ray row vectors.
    print('--- generators (contains the vertices) ---')
    print(generators)

    inequalities = poly.get_inequalities() # the H-representation
    print('--- inequalities (contains the equations) ---')
    print(inequalities)

    adjacency_list = poly.get_adjacency()
    print('--- adjacancy list ---')
    print(adjacency_list)


#example1()
example3()