import cdd


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

    def vertices(self):
        return [list(x) for x in self.get_generators()]

    def faces(self):
        return [list(x) for x in self.get_inequalities()]

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
        print('- list the vertices (generators)')
        print(self.vertices())

        print('- list the faces (inequalities)')
        print(self.faces())

        print('- for each face, list adjacent faces')
        print(self.face_face_adjacencies())

        print('- for each face, list adjacent vertices')
        print(self.face_vertex_adjacencies())

        print('- for each vertex, list adjacent vertices')
        print(self.vertex_vertex_adjacencies())

        print('- for each vertex, list adjacent faces')
        print(self.vertex_face_adjacencies())
