import cdd


class InEquality(object):
    def __init__(self, v):
        self.v = v

    def __str__(self):
        # prints a * x[i]
        def term(a, i):
            if a == 0:
                return ''
            if a == 1:
                return f'x{i}'
            if a == -1:
                return f'-x{i}'
            return f'{a}*x{i}'

        v = self.v
        n = len(v)
        b = -v[0]
        x = filter(None, [term(v[i], i) for i in range(1, n)])
        lhs = ' + '.join(x)
        return f'{lhs} = {b}'


class Polyhedron(cdd.Polyhedron):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    def is_H(self):
        return self.rep_type == cdd.RepType.INEQUALITY

    def is_V(self):
        return self.rep_type == cdd.RepType.GENERATOR

    def to_H(self):
        self.rep_type = cdd.RepType.INEQUALITY

    def to_V(self):
        self.rep_type = cdd.RepType.GENERATOR

    def generators(self):
        return [list(x) for x in self.get_generators()]

    def inequalities(self):
        return [list(x) for x in self.get_inequalities()]

    def vertices(self):
        return [list(x[1:]) for x in self.get_generators()]

    def faces(self):
        return [InEquality(x) for x in self.get_inequalities() if x]

    # for each vertex, list adjacent vertices
    def vertex_vertex_adjacencies(self):
        if self.is_H():
            return [list(x) for x in self.get_adjacency()]
        else:
            return [list(x) for x in self.get_input_adjacency()]

    # for each face, list adjacent faces
    def face_face_adjacencies(self):
        if self.is_H():
            return [list(x) for x in self.get_input_adjacency()]
        else:
            return [list(x) for x in self.get_adjacency()]

    # for each face, list adjacent vertices
    def face_vertex_adjacencies(self):
        if self.is_H():
            return [list(x) for x in self.get_input_incidence()]
        else:
            return [list(x) for x in self.get_incidence()]

    # for each vertex, list adjacent faces
    def vertex_face_adjacencies(self):
        if self.is_H():
            return [list(x) for x in self.get_incidence()]
        else:
            return [list(x) for x in self.get_input_incidence()]

    # prints some information about the polyhedron
    def info(self):
        print('- list the vertices')
        print(self.vertices())

        print('- list the faces')
        print(list(map(str, self.faces())))

        print('- list the generators')
        print(self.generators())

        print('- list the inequalities')
        print(self.inequalities())

        print('- for each face, list adjacent faces')
        print(self.face_face_adjacencies())

        print('- for each face, list adjacent vertices')
        print(self.face_vertex_adjacencies())

        print('- for each vertex, list adjacent vertices')
        print(self.vertex_vertex_adjacencies())

        print('- for each vertex, list adjacent faces')
        print(self.vertex_face_adjacencies())
