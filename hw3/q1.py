import numpy as np

def computeY(X, w, b):
    return np.dot(X, w) + b

def computeHuberLoss(y, t, delta):
    a = y-t
    return (abs(a) <= delta)*(0.5*a**2) \
        +  (abs(a) > delta)*(delta*(abs(a)-0.5*delta))

def huberCost(X, t, delta, w, b):
    N = len(X)
    y = computeY(X, w, b)
    L = computeHuberLoss(y, t, delta)
    return (1./N)*sum(L)

def gradient_descent(X, t, lr, num_iter, delta):
    """
    Gradient descent full batch mode.

    Parameters:
        X - matrix, each row is a training sample
        t - labels 
        lr - learning rate
        num_iter - number of iterations
        delta - Huber loss parameter
    """
    N = len(X) # num examples
    D = len(X[0]) # num features
    # initialize weights and bias
    w = np.random.rand(D, 1)
    b = np.random.rand(1,1)
    costs = [0 for i in range(0, num_iter)] # also returns costs, so you can check them
    for i in range(0, num_iter):
        y = computeY(X, w, b)
        a = y - t
        # compute derivatives of individual losses
        # this is a column vector nx1
        dLdy = (abs(a)<delta)* a \
             + (a>delta)     * delta \
             + (a<-delta)    * -1*delta
        dFdy = (1./N)*dLdy.transpose()
        dydw = X # nxd
        dFdw = np.dot(dFdy, dydw) # 1xd
        w = w - lr*dFdw.transpose()
        
        # For clarity but unneeded. We could just sum dFdy
        dydb = np.array([1 for j in range(0,N)]).reshape(N,1)
        dFdb = np.dot(dFdy, dydb)
        b = b - lr*dFdb
        costs[i] = huberCost(X, t, delta, w, b)[0]
    return (w,b,costs)


def testHuberLoss():
    y = np.array(list(range(-10, 11)))
    t = np.array([0 for i in y])
    delta = 3
    L = computeHuberLoss(y, t, delta)
    assert L[0] == delta*(abs(y[0]-t[0])-0.5*delta)
    assert L[8] == 0.5*(y[8]-t[8])**2

def testGradientDescent():
    N = 100
    D = 3
    X = 100*np.random.rand(N,D) 
    t = np.array([0 for i in range(0, N)]).reshape(N,1)
    alpha = 0.00001 # interestingly, if alpha=0.01, causes oscillation
    num_iter = 500
    delta = 30
    w,b,costs = gradient_descent(X, t, alpha, num_iter, delta)
    assert all(np.diff(costs) < 0) # assert costs always goes down
    print("Found w=")
    print(w)
    print("Found b=")
    print(b)
    print("Final cost is %f" % huberCost(X, t, delta, w, b))
    

if __name__ == "__main__":
    testHuberLoss()
    testGradientDescent()