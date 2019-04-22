from collections import defaultdict
import numpy as np
from matplotlib.pyplot import *

# state enumeration
class States:
    LOW_BANK_ACCOUNT = 0
    HIGH_BANK_ACCOUNT = 1

# action enumeration
class Actions:
    SAVE = 0
    SPEND = 1


# map from state, action to list of tuples (reward, probability of reward)
reward_pmf = {
    States.LOW_BANK_ACCOUNT: {
        Actions.SAVE: [(-1, 1)],
        Actions.SPEND: [(+1, 1)]
    },
    States.HIGH_BANK_ACCOUNT: {
        Actions.SAVE: [(0, 1)],
        Actions.SPEND: [(5, 1)]
    }
}

# map from state, action to list of tuples (next state, probability of next state)
transition_pmf = {
    States.LOW_BANK_ACCOUNT: {
        Actions.SAVE: [(States.HIGH_BANK_ACCOUNT, 1)],
        Actions.SPEND: [(States.LOW_BANK_ACCOUNT, 1)]
    },
    States.HIGH_BANK_ACCOUNT: {
        Actions.SAVE: [(States.HIGH_BANK_ACCOUNT, 1)],
        Actions.SPEND: [(States.LOW_BANK_ACCOUNT, 1)]
    }
}

states = [States.LOW_BANK_ACCOUNT, States.HIGH_BANK_ACCOUNT]
actions = [Actions.SAVE, Actions.SPEND]

def optimalAction(Q, s):
    """Return the optimal action to take given the Q function and the state.
    """
    best_action = None
    best_value = None
    for (action, value) in Q[s].items():
        if best_action == None:
            best_action = action
            best_value = value
        if value > best_value:
            best_action = action
            best_value = value
    return best_action

def maxQ(Q, s):
    """ Return the best value of the Q function given some state.
    """
    optimal_a = optimalAction(Q, s)
    return Q[s][optimal_a]

def value_iteration(transition_pmf, reward_pmf, states, actions, gamma=0.9, num_iter = 1000, policy=None):
    """ Finds the optimal Q function using value iteration.
        If given a policy, finds the Q function associated with that policy.
        Policy is a map from state to action.
    """

    # Q(state, action) with default values of 0
    Q = defaultdict(lambda: defaultdict(int))
    for i in xrange(0, num_iter):
        Qtemp = Q
        for state in states:
            for action in actions:
                if policy == None:
                    expected_current_reward = sum([r*p for r,p in reward_pmf[state][action]])
                else:
                    expected_current_reward = sum([r*p for r,p in reward_pmf[state][policy[state]]])
                discounted_future_reward = gamma * sum([p*maxQ(Q, s) for s,p in transition_pmf[state][action]])
                Qtemp[state][action] = expected_current_reward + discounted_future_reward
        Q = Qtemp
    return Q

if __name__ == "__main__":
    pol_1 = {States.LOW_BANK_ACCOUNT: Actions.SAVE, States.HIGH_BANK_ACCOUNT: Actions.SAVE} # Saving policy
    pol_2 = {States.LOW_BANK_ACCOUNT: Actions.SPEND, States.HIGH_BANK_ACCOUNT: Actions.SPEND} # Spending policy

    gamma = 0.9
    num_iter = 1000
    Q = value_iteration(transition_pmf, reward_pmf, states, actions, gamma, num_iter)
    Q = [[Q[s][a] for a in actions] for s in states] # convert it to the form as per the solution
    print(Q)
    print 'Optimal policy:', np.argmax(Q,axis = 1)


    # Varying discount factor (gamma) and computing the optimal action-value and value function, and policy
    gamma_set = np.linspace(0,0.99,100)
    Q_res = []
    V_res = []
    pi_res = []
    for gamma in gamma_set:
        Q = value_iteration(transition_pmf, reward_pmf, states, actions, gamma, num_iter) #, policy=pol_2)
        Q = np.array([np.array([Q[s][a] for a in actions]) for s in states]) # convert it to the form as per the solution
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