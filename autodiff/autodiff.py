import numpy as np

class Node:

    def __init__(self, value, vjps, parents):
        """
        parents are the inputs used to compute the value. each parent must have an associated vjp.
        """
        self.value = value
        self.vjps = vjps
        self.parents = parents
        self.v = None # derivative of output with respect to this node. Needs to be set by a parent.

    def __str__(self):
        return self.value.__str__()

    def prop(self):
        for parent, vjp in zip(self.parents, self.vjps):
            parent.v = vjp(self.v, parent.value)

def sin_vjp(v, x):
    return np.multiply(v, np.cos(x))

def sin(x):
    if x.__class__ == Node:
        return Node(np.sin(x.value), [sin_vjp], [x])
    else:
        return Node(np.sin(x), [sin_vjp], [x])

def sinsin(x):
    return sin(sin(x))

def backprop(end_node):
    end_node.v = np.ones(len(end_node.value)) # deriv wrt itself is 1
    current_node = end_node
    while True:
        if current_node.parents == None:
            break
        current_node.prop()
        current_node = current_node.parents[0] # for now, assuming one parent (i.e. a chain)
    return current_node.v

# returns gradient of fct evaluated at input
def grad(fct, x):
    start_node = Node(x, None, None)
    end_node = fct(start_node)
    return backprop(end_node)

if __name__ == "__main__":
    gradVal = grad(sinsin, [0.2])
    print(gradVal)