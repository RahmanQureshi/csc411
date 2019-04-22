#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.pyplot import *


def value_iteration(P, r, gamma = 0., policy = None, iter_no = None):
    '''Value iteration for finite state and action spaces'''
    
    actions_no, states_no = np.shape(P)[0:2]
    
    # A simple heuristic to set the number of iterations
    iter_no = min(1000, int(20./(1-gamma)))

    Q = np.zeros((states_no,actions_no))
    Q_new = Q

    if policy is None:
        Bellman_optimality = True
    else:
        Bellman_optimality = False

    for m in range(iter_no):
        Q = Q_new
        for s in range(states_no):
            for a in range(actions_no):
                
                value_next = 0
                
                # This loop could be optimized using numpy
                for s_next in range(states_no):                    
                    if Bellman_optimality:
                        value_next += P[a,s,s_next] * max(Q[s_next]) #                         
                    else:
                        value_next += P[a,s,s_next] * Q[s_next,policy[s_next]]                         
                    
                Q_new[s,a] = r[s,a] + gamma* value_next
          
    return Q






P = np.array( [ [[0, 1.], [0., 1.0]], 
     [ [1.,0 ], [1.0,0.0 ] ] ] )

r = np.array( [ [-1., 1.0], [0.0, 5.]] )
gamma = 0.9


pol_1 = np.array([0, 0]) # Saving policy
pol_2 = np.array([1,1]) # Spending policy

Q = value_iteration(P,r, gamma) #, policy=pol_2)
print 'Q:', Q
print 'Optimal policy:', np.argmax(Q,axis = 1)


# Varying discount factor (gamma) and computing the optimal action-value and value function, and policy
gamma_set = np.linspace(0,0.99,100)
Q_res = []
V_res = []
pi_res = []
for gamma in gamma_set:
    Q = value_iteration(P,r, gamma) #, policy=pol_2)
    V = np.max(Q,axis = 1)
    pi_greedy = np.argmax(Q,axis = 1)

    Q_res.append(Q*(1. - gamma))
    V_res.append(V*(1. - gamma))
    pi_res.append(pi_greedy)

subplot(1,2,1)
plot(gamma_set, np.array(pi_res)[:,0],'o')
xlabel('$\gamma$')
ylabel('Policy at state $s_1$')
ylim((-0.05,1.05))
subplot(1,2,2)
plot(gamma_set, np.array(pi_res)[:,1],'o')
ylim((-0.05,1.05))
xlabel('$\gamma$')
ylabel('Policy at state $s_2$')
figure()
plot(gamma_set, V_res)
legend(['State 1', 'State 2'])
xlabel('$\gamma$')
ylabel('(Normalized) optimal value function')
show()