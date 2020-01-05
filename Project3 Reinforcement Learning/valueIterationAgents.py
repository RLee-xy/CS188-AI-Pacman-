# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        iteration = 0
        while iteration < self.iterations:
          values = util.Counter()
          for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
              maxv = float('-inf')
              if len(self.mdp.getPossibleActions(state)) == 0:
                iteration += 1
                continue
              for act in self.mdp.getPossibleActions():
                sumv = self.computeQValueFromValues(state,act)
                maxv = max(maxv,sumv)
              values[state] = maxv
          self.values = values.copy() 
          iteration += 1


        "*** YOUR CODE HERE ***"


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if not action:
          return 0
        sumv = 0
        for trans in self.mdp.getTransitionStatesAndProbs(state,act):
          sumv += trans[1] * (self.mdp.getReward(state,act,trans[0]) + self.discount*self.values[trans[0]])
        return sumv
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
          return None
        
        acts = self.mdp.getPossibleActions(state)
        maxv = float('-inf')
        bestact = None

        for act in acts:
          value = self.computeQValueFromValues(state,act)
          if maxv < value:
            bestact = act
            maxv = value
        return bestact


        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        self.values = util.Counter()
        i = 0
        states = self.mdp.getStates()
        
        while i < self.iterations:
          state = states[i%len(states)]
          if not self.mdp.isTerminal(state):
              maxv = float('-inf')
              if len(self.mdp.getPossibleActions(state)) == 0:
                i += 1
                continue
              for act in self.mdp.getPossibleActions():
                sumv = self.computeQValueFromValues(state,act)
                maxv = max(maxv,sumv)
              
              self.values[state] = maxv 
          i += 1


        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        self.values = util.Counter()
        predecessors = {}
        sweepQ = util.PriorityQueue()
        # add predecessprs
        for state in self.mdp.getStates():
          predecessors[state] = set()
          if self.mdp.isTerminal(state):
            continue
          for act in self.mdp.getPossibleActions(state):
            for trans in self.mdp.getTransitionStatesAndProbs(state,act):
              if trans[1] != 0:
                predecessors[trans[0]].add(state)
        # calculate difference for states and get priorityQueue1
        for state in self.mdp.getStates():
          if self.mdp.isTerminal(state):
            continue
          min_diff = float('inf')
          for act in self.mdp.getPossibleActions(state):
            diff = abs(self.getValue(state) - self.computeQValueFromValues(state,act))
            if diff < min_diff:
              min_diff = diff
          sweepQ.push(state, -min_diff)
        # iterate value for self.values
        i = 0
        while i < self.iterations:
          if sweepQ.isEmpty():
            break
          state = sweepQ.pop()
          maxv = float('-inf')
          if not self.mdp.isTerminal(state):
            for act in self.mdp.getPossibleActions(state): 
              value = self.computeQValueFromValues(state,act)
              maxv = max(maxv,value)
            self.value[state] = maxv
          
        # calculate difference for states' predecessors and get priorityQueue2
            for node in predecessors[state]:
              min_diff = float('inf')
              for act in self.mdp.getPossibleActions(node):
                diff = abs(self.getValue(state) - self.computeQValueFromValues(state,act))
                if diff < min_diff:
                  min_diff = diff
              if min_diff > self.theta:
                sweepQ.update(node, -min_diff)
          i += 1

        "*** YOUR CODE HERE ***"

