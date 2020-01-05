# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Is win or lose?
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')
        dis_ghost = 0
        score = successorGameState.getScore()
        
        # Is ghost close?
        for ghost in newGhostStates:
            if ghost.scaredTimer <= 1:
                dis_ghost = util.manhattanDistance(newPos, ghost.getPosition())
                if dis_ghost < 5:
                    score -= 10*(5 - dis_ghost)
        
        # Is there a food in successorGameState?
        food = newFood.asList();
        food_left = len(food)
        if food_left < len(currentGameState.getFood().asList()):
            score += 50
        
        # adjust the score according to the distance from newPos to the closest food
        dis_food = float('inf')
        
        for f in food:
            dis_food = min(dis_food, util.manhattanDistance(newPos, f))

        score -= dis_food

        return score


        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = 1 
        agent_index = 0
        action = self.maxi(gameState, agent_index, depth)
        return action
        util.raiseNotDefined()
        
        
    def mini(self, gameState, agentnum, dep, num_ghost):
        min_val = float('inf') 
        act = ''
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(agentnum):
            successor = gameState.generateSuccessor(agentnum, action)
            if agentnum == num_ghost:
                if dep < self.depth:
                    min_val = min(min_val, self.maxi(successor, 0, dep + 1))
                else:
                    min_val = min(min_val, self.evaluationFunction(successor))
            else:
                min_val = min(min_val, self.mini(successor, agentnum + 1, dep, num_ghost))
     

        return min_val



    def maxi(self, gameState, agentnum, dep):
        max_val = float('-inf')
        act = 'STOP'
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        num_ghost = gameState.getNumAgents() - 1
        for action in gameState.getLegalActions(agentnum):
            successor = gameState.generateSuccessor(agentnum, action)
            val = self.mini(successor, agentnum + 1, dep,num_ghost)
            if val > max_val:
                max_val = val
                act = action

        if dep == 1:
            return act
        else:
            return max_val


        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 1 
        agent_index = 0
        alpha = float('-inf')
        beta = float('inf')
        action = self.maxi(gameState, agent_index, depth, alpha, beta)
        return action
        util.raiseNotDefined()


    def mini(self, gameState, agentnum, dep, num_ghost, alpha,beta):
        min_val = float('inf') 
        act = ''
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(agentnum):
            successor = gameState.generateSuccessor(agentnum, action)
            if agentnum == num_ghost:
                if dep < self.depth:
                    val = min(min_val, self.maxi(successor, 0, dep + 1,alpha,beta))
                else:
                    val = min(min_val, self.evaluationFunction(successor))
            else:
                val = min(min_val, self.mini(successor, agentnum + 1, dep, num_ghost,alpha,beta))
            if val < min_val:
                min_val = val
            if min_val < alpha:
                return min_val
            beta = min(beta, min_val)

        return min_val



    def maxi(self, gameState, agentnum, dep, alpha, beta):
        max_val = float('-inf')
        act = 'STOP'
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        num_ghost = gameState.getNumAgents() - 1
        for action in gameState.getLegalActions(agentnum):
            successor = gameState.generateSuccessor(agentnum, action)
            val = self.mini(successor, agentnum + 1, dep,num_ghost,alpha, beta)
            if val > max_val:
                max_val = val
                act = action

            if max_val > beta:
                return max_val
            alpha = max(alpha, max_val)

        if dep == 1:
            return act
        else:
            return max_val

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        depth = 1 
        agent_index = 0
        action = self.maxi(gameState, agent_index, depth)
        return action
        util.raiseNotDefined()

    def expect(self, gameState, agentnum, dep, num_ghost):
        expect_val = 0
        act = ''
        act_num = 0
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(agentnum):
            act_num += 1
            successor = gameState.generateSuccessor(agentnum, action)
            if agentnum == num_ghost:
                if dep < self.depth:
                    expect_val += self.maxi(successor, 0, dep + 1)
                else:
                    expect_val += self.evaluationFunction(successor)
            else:
                expect_val += self.expect(successor, agentnum + 1, dep, num_ghost)
        expect_val = expect_val / act_num

        return expect_val



    def maxi(self, gameState, agentnum, dep):
        max_val = float('-inf')
        act = 'STOP'
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        num_ghost = gameState.getNumAgents() - 1
        for action in gameState.getLegalActions(agentnum):
            successor = gameState.generateSuccessor(agentnum, action)
            val = self.expect(successor, agentnum + 1, dep,num_ghost)
            if val > max_val:
                max_val = val
                act = action

        if dep == 1:
            return act
        else:
            return max_val

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pac_pos = currentGameState.getPacmanPosition()

    food_left = currentGameState.getFood().asList()

    ghosts = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    if currentGameState.isWin():
            return float('inf')
    if currentGameState.isLose():
            return float('-inf')

    for ghost in ghosts:
        if ghost.scaredTimer < 2:
            dis_ghost = util.manhattanDistance(pac_pos, ghost.getPosition())
            if dis_ghost < 5:
                score -= 2*(5 - dis_ghost)
            

    closestfood = float('inf')
    for food in food_left:
        
        dis_food = util.manhattanDistance(pac_pos, food)
        closestfood = min(closestfood, dis_food)
        score += 1 / dis_food

    return score - closestfood - currentGameState.getNumFood()
   
    

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
