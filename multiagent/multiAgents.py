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


import random, util

from util import manhattanDistance
from game import Directions
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
     #   print('successorGameState', successorGameState)
        newPos = successorGameState.getPacmanPosition()
    #    print('newPos', newPos)
        newFood = successorGameState.getFood()
      #  print('newFood', newFood)
        newGhostStates = successorGameState.getGhostStates()
     #   print('newGhostStates', newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #    print('newScaredTimes', newScaredTimes)
        "*** YOUR CODE HERE ***"
        
        foodList = newFood.asList()
   #     print('foodlist', foodList)
        ghostPositions = []
        for ghostState in newGhostStates:
            ghostPositions.append(ghostState.getPosition())

        foodWeight = 10
        scaredGhostWeight = 50

        evaluationScore = successorGameState.getScore()

        # distance of closest food
        closestFoodDotDist = float('inf')
        newClosestFoodDotDist = 0
        for food in foodList:
          #  print(manhattanDistance(newPos, food))
            newClosestFoodDotDist = manhattanDistance(newPos, food)
            if newClosestFoodDotDist < closestFoodDotDist:
                closestFoodDotDist = newClosestFoodDotDist

        #distance of closest ghost
        ghostDist = float('inf')
        minGhostDist = 0
        for ghostPosition in ghostPositions:
            minGhostDist = manhattanDistance(newPos, ghostPosition)
            if minGhostDist < ghostDist:
                ghostDist = minGhostDist

        evaluationScore = successorGameState.getScore()  # Initialize with the current score

        evaluationScore += foodWeight / (closestFoodDotDist + 1)
        if ghostDist < 2:
            if newScaredTimes[0] > 0:  # encouraging agent to be close to scared ghost and eating scared ghost
                evaluationScore += scaredGhostWeight
            else:  # Penalize pacman agent for being too close to non-scared ghost
                evaluationScore -= 10

        return evaluationScore

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
    # access important info about current state using functions from GameState class defined in pacman.py
    # write functions to recursively calculate values for maxNode and minNode in search tree
    # when calculating the value consider the variables current state, whether node is MaxNode or minNode and current depth of node

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
        """

        pacman_legal_actions = gameState.getLegalActions(0)
        max_value = float('-inf')
        max_action  = None #one to be returned at the end.

        for action in pacman_legal_actions:   #get the max value from all of it's successors.
            action_value = self.Min_Value(gameState.generateSuccessor(0, action), 1, 0)
            if ((action_value) > max_value ): #take the max of all the children.
                max_value = action_value
                max_action = action

        return max_action #Returns the final action .

    def Max_Value (self, gameState, depth):
        """For the Max Player here Pacman"""

        if ((depth == self.depth)  or (len(gameState.getLegalActions(0)) == 0)):
            return self.evaluationFunction(gameState)
        
        v = float('-inf')
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(0, action)
            score = self.Min_Value(successor_state, 1, depth)
            if score > v:
                v = score
        return v

    def Min_Value (self, gameState, agentIndex, depth):
        """ For the MIN Players or Agents  """

        if (len(gameState.getLegalActions(agentIndex)) == 0): #No Legal actions.
            return self.evaluationFunction(gameState)

        if agentIndex < gameState.getNumAgents() - 1:
            v = float('inf')
            legal_actions = gameState.getLegalActions(agentIndex)
            for action in legal_actions:
                successor_state = gameState.generateSuccessor(agentIndex, action)
                v = min(v, self.Min_Value(successor_state, agentIndex + 1, depth))
            return v
        else:  # the last ghost
            v = float('inf')
            legal_actions = gameState.getLegalActions(agentIndex)
            for action in legal_actions:
                successor_state = gameState.generateSuccessor(agentIndex, action)
                v = min(v, self.Max_Value(successor_state, depth + 1))
            return v
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_action = Directions.STOP

        def max_value(gameState, depth, alpha, beta):
            depth += 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            v = float('-inf')
            for action in gameState.getLegalActions(0):
                successor_state = gameState.generateSuccessor(0, action)
                v = max(v, min_value(successor_state, 1, depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        def min_value(gameState, agentIndex, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('inf')

            for action in gameState.getLegalActions(agentIndex):
                successor_state = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    v = min(v, max_value(successor_state, depth,alpha, beta))
                else:
                    v = min(v, min_value(successor_state, agentIndex + 1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        
        for action in gameState.getLegalActions(0):
            successor_state = gameState.generateSuccessor(0, action)
            score = min_value(successor_state, 1, 0, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        return best_action


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
        def max_value(gameState, depth):
            depth += 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                v = max(v, expect_value(successor, depth, 1))
            return v
        
        def expect_value(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            totalexpectedvalue = 0
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = max_value(successor,depth)
                else:
                    expectedvalue = expect_value(successor,depth,agentIndex+1)
                totalexpectedvalue = totalexpectedvalue + expectedvalue
            if len(actions) == 0:
                return 0
            return float(totalexpectedvalue)/float(len(actions))
        
        actions = gameState.getLegalActions(0)
        currentScore = float('-inf')
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            score = expect_value(nextState,0,1)
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction



def betterEvaluationFunction(currentGameState):
    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()

    "*** YOUR CODE HERE ***"
       # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()

    minFoodist = float('inf')
    for food in newFood:
        minFoodist = min(minFoodist, manhattanDistance(newPos, food))

    ghostDist = 0
    for ghost in currentGameState.getGhostPositions():
        ghostDist = manhattanDistance(newPos, ghost)
        if (ghostDist < 2):
            return -float('inf')

    foodLeft = currentGameState.getNumFood()
    capsLeft = len(currentGameState.getCapsules())

    foodLeftMultiplier = 950050
    capsLeftMultiplier = 10000
    foodDistMultiplier = 950

    additionalFactors = 0
    if currentGameState.isLose():
        additionalFactors -= 50000
    elif currentGameState.isWin():
        additionalFactors += 50000

    return 1.0/(foodLeft + 1) * foodLeftMultiplier + ghostDist + \
           1.0/(minFoodist + 1) * foodDistMultiplier + \
           1.0/(capsLeft + 1) * capsLeftMultiplier + additionalFactors

# Abbreviation
better = betterEvaluationFunction