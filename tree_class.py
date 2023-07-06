class Tree():
    '''
    Object for representation of a context tree. Implements method to add
    new nodes that adds all siblings to tree as well and a method to prune
    the tree, turning an intermediate node into a leaf. Note leaves are
    the contexts, and can be accessed easily by taking the nodes which are
    not keys in the self.edges attribute.

    Attributes:
    s: iterable
        Set of possible states of process.
    nodes: set
        All nodes, including root (""), intermediate and leaves (contexts).
    edges: dict
        Dict {node: list(children)} for root and all intermediate nodes.
    '''
    def __init__(self, s):
        self.s = s
        self.nodes = {""}
        self.edges = dict()

    def add_node(self, node): # adds node and siblings as leaves in the tree
        parent = node[1:]
        assert parent in self.nodes

        siblings = [b+parent for b in self.s]
        self.nodes.update(siblings)
        self.edges[parent] = siblings

    def prune_at_node(self, node): # turns node into a leaf
        try:
            children = self.edges[node]

            for child in children:
                if child not in self.edges.keys(): # child is leaf:
                    self.nodes.discard(child) # remove from nodes
                else: # child is intermediate node:
                    self.prune_at_node(child) # remove descendants first
                                              # through recursion at child
                    self.nodes.discard(child) # then remove child

            self.edges.pop(node) # node no longer has children
            # note node stays in self.nodes (as it still is a leaf)

        except KeyError: # node not in edges: node is already a leaf
            pass


