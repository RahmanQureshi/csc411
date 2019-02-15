import numpy as np
from collections import deque
from collections import defaultdict
from toposort import toposort_flatten

# All VJP functions accept v, the derivative up to the "current" node,
# the parent node to which we are propogating, and all the inputs/parents of the "current" node
def sin_vjp(v, parent, parents):
    return np.multiply(v, np.cos(parent.value))

def mult_vjp(v, parent, parents):
    if parent == parents[0]:
        return v*parents[1].value
    return v*parents[0].value

class Node:

    def __init__(self, value, vjps, parents):
        """
        parents are the inputs used to compute the value. each parent must have an associated vjp.
        """
        self.value = value
        self.vjps = vjps
        self.parents = parents
        self.v = 0 

    def __str__(self):
        return self.value.__str__()

    def __mul__(self, other):
        # TODO: this is not good, we are effectively duplicating information
        if other.__class__ != Node:
            other = Node(other, None, None)
        return Node(self.value*other.value, [mult_vjp, mult_vjp], [self, other])

    def __rmul__(self, other):
        if other.__class__ != Node:
            other = Node(other, None, None)
        return Node(self.value*other.value, [mult_vjp, mult_vjp], [self, other])

    def prop(self):
        for parent, vjp in zip(self.parents, self.vjps):
            parent.v += vjp(self.v, parent, self.parents)

def sin(x):
    if x.__class__ == Node:
        return Node(np.sin(x.value), [sin_vjp], [x])
    else:
        return Node(np.sin(x), [sin_vjp], [x])

def sinsin(x):
    return sin(sin(x))

def sinsinsin(x):
    return sin(sin(sin(x)))

def sinsquared(x):
    return sin(x)*sin(x)

def threetimessin(x):
    return 3*sin(x)

def buildGraph(end_node):
    G = defaultdict(set)
    visited = set()
    queue = deque([end_node])
    while len(queue) > 0:
        current = queue.popleft()
        visited.add(current)
        if current.parents == None:
            continue
        for parent in current.parents:
            G[current].add(parent)
            if parent not in visited:
                queue.append(parent)
    return G

def backprop(end_node, start_node):
    end_node.v = np.ones(len(end_node.value)) # deriv wrt itself is 1
    G = buildGraph(end_node)
    for node in reversed(toposort_flatten(G)):
        if node.parents == None:
            continue
        node.prop()
    return start_node.v

# returns gradient of fct evaluated at input
def grad(fct, x):
    start_node = Node(x, None, None)
    end_node = fct(start_node)
    return backprop(end_node, start_node)

if __name__ == "__main__":
    gradVal = grad(threetimessin, [0.2])
    print(gradVal)