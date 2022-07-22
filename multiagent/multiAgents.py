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


from typing import Counter
from util import chooseFromDistribution, manhattanDistance
from game import Directions
import random, util

import math
import sys

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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodLeft = childGameState.getNumFood()

        # to find the nearest ghost
        ghostLocation = childGameState.getGhostPositions()
        ghostPosit = ghostLocation[0]
        ghostDistance=0
        ghostCount=0
        for ghostPosit in ghostLocation:
            ghostCount+=1
            if ghostDistance == 0:
                ghostDistance = manhattanDistance(ghostPosit, newPos)
            elif ghostDistance > manhattanDistance(ghostPosit, newPos):
                ghostDistance = manhattanDistance(ghostPosit, newPos)

        # CHASE MODE to eat ghosts
        if newScaredTimes[0]>0:
            if newScaredTimes[0]==40:
                return sys.maxsize # to eat the cherry
            return 1/(ghostDistance+1) # to eat the ghost
        
        # every food distance
        maxFoodDistance = 0 
        #closest cherry or food dot
        minMANH = 0 
        cherries = childGameState.getCapsules()
        if (cherries):
            for cherry in cherries:
                if minMANH ==0:
                    minMANH = manhattanDistance(newPos, cherry)
                elif minMANH > manhattanDistance(newPos, cherry):
                    minMANH = manhattanDistance(cherry, newPos)
        for food in newFood.asList():
            if minMANH ==0:
                minMANH = manhattanDistance(newPos, food)
            elif minMANH > manhattanDistance(newPos, food):
                minMANH = manhattanDistance(newPos, food)
            maxFoodDistance += manhattanDistance(newPos, food)

        # copysign returns the magnitude of the first argument with the sign of the second.
        # this way i achieve a kind of step function, by using in the second argument the logarithm of the distance to the nearest ghost
        return math.copysign(ghostCount/(minMANH+1) + ghostCount/(maxFoodDistance+1) +100000/(foodLeft+1)+(0.0045*((ghostCount**2))*math.log(1+ghostDistance)), (math.log(((ghostDistance+1)/3))))
       

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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # to return the Pac-Man move call maxi
        return self.maxi(gameState, gameState.getNumAgents()*self.depth, 0)[0]

    def minimax(self, gameState, depth, player):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            # if its in depth 0 or winning state
            return self.evaluationFunction(gameState) 
        if player==0:
            return self.maxi(gameState, depth,player)[1]
        else:
            return self.mini(gameState, depth, player)[1]

    def mini(self, gameState, depth, player):
        mini = ("placeholder", float(sys.maxsize)) 
        for action in gameState.getLegalActions(player):
            actionToConsider = (action, self.minimax(gameState.getNextState(player, action), depth-1, (player+1)%gameState.getNumAgents()))
            if actionToConsider[1] < mini[1]:
                mini = actionToConsider
        return mini 

    def maxi(self, gameState, depth, player):
        maxi = ("placeholder", -float(sys.maxsize))
        for action in gameState.getLegalActions(player):
            actionToConsider = (action, self.minimax(gameState.getNextState(player, action), depth-1, (player+1)%gameState.getNumAgents()))
            if actionToConsider[1] > maxi[1]:
                maxi = actionToConsider
        return maxi 


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alfa = -float(sys.maxsize)
        beta = float(sys.maxsize)
        player = 0 # pacman 
        return self.maxValue(gameState, (gameState.getNumAgents()*self.depth), alfa, beta, player)[0]

    # same as minimax but with pruning
    def alfaBeta(self, gameState, depth, alfa, beta, player): 
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if player==0:
            return self.maxValue(gameState, depth, alfa, beta, player)[1]
        else:
            return self.minValue(gameState, depth, alfa, beta, player)[1]

    def minValue(self, gameState, depth, alfa, beta, player):
        mini = ("placeholder", float(sys.maxsize))
        for action in gameState.getLegalActions(player):
            actionToConsider = (action, self.alfaBeta(gameState.getNextState(player, action), depth-1, alfa, beta, (player+1)%gameState.getNumAgents()))
            if actionToConsider[1] < mini[1]:
                mini = actionToConsider
            if mini[1] < alfa:
                return mini
            elif (beta > mini[1]):
                beta = mini[1]
        return mini 

    def maxValue(self, gameState, depth, alfa, beta, player):
        maxi = ("placeholder", -float(sys.maxsize))
        for action in gameState.getLegalActions(player):
            actionToConsider = (action, self.alfaBeta(gameState.getNextState(player, action), depth-1, alfa, beta, (player+1)%gameState.getNumAgents()))
            if actionToConsider[1] > maxi[1]:
                maxi = actionToConsider
            if maxi[1] > beta:
                return maxi
            elif (alfa < maxi[1]):
                alfa = maxi[1]
        return maxi 


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
        # as previously return expectimax
        return self.expectiMax(gameState, (gameState.getNumAgents()*self.depth), "placeholder", 0)[0]
    
    def expectiMax(self, gameState, depth, action, player):
        if depth == 0 or gameState.isLose() or gameState.isWin(): 
            exp = (action, self.evaluationFunction(gameState)) 
            return exp 
        if player==0: # for pac-man
            maxMaxMaxSuperMax = ("placeholder", -(float(sys.maxsize)))
            for actionPlayer in gameState.getLegalActions(player):
                if depth == (gameState.getNumAgents()*self.depth):
                    consider = actionPlayer
                else:
                    consider = action
                actionToConsider = self.expectiMax(gameState.getNextState(player, actionPlayer), depth-1, consider, (player+1)%gameState.getNumAgents())
                if actionToConsider[1] > maxMaxMaxSuperMax[1]:
                    maxMaxMaxSuperMax = actionToConsider
            return maxMaxMaxSuperMax
        else:
            kasperGhost = 0
            for actionGhost in gameState.getLegalActions(player): 
                actionToConsider = self.expectiMax(gameState.getNextState(player, actionGhost), depth-1, action, (player+1)%gameState.getNumAgents())
                kasperGhost += (actionToConsider[1] * 1.0/len(gameState.getLegalActions(player)))
            ghosty = (action, kasperGhost)
            return ghosty


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    newPos = currentGameState.getPacmanPosition()

    # to find nearest ghost
    ghostLocation = currentGameState.getGhostPositions()
    ghostPosit = ghostLocation[0]
    ghostDistance=0
    ghostCount=0
    for ghostPosit in ghostLocation:
        ghostCount+=1
        if ghostDistance == 0:
            ghostDistance = manhattanDistance(ghostPosit, newPos)
        elif ghostDistance > manhattanDistance(ghostPosit, newPos):
            ghostDistance = manhattanDistance(ghostPosit, newPos)

    # CHASE MODE
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    if newScaredTimes[0]>0: 
        return sys.maxsize/(ghostDistance+1) # so that Pac-Man is more "aggressive" in his moves
        
    # distance of all foods
    maxFoodDistance = 0
    cherries = len(currentGameState.getCapsules())
    foodLeft = currentGameState.getNumFood()  
    newFood = currentGameState.getFood()

     # nearest food dot
    minMANH = 0
    for food in newFood.asList():
        if minMANH ==0:
            minMANH = manhattanDistance(newPos, food)
        elif minMANH > manhattanDistance(newPos, food):
            minMANH = manhattanDistance(newPos, food)
        maxFoodDistance += manhattanDistance(newPos, food) # maxfooddistance so that he goes towards "clusters" of food

    # copysign returns the magnitude of the first argument with the sign of the second.
    # this way i achieve a kind of step function, by using in the second argument the logarithm of the distance to the nearest ghost
    # +1 is there so that it can work when it is zero
    return math.copysign(10*(ghostCount/(maxFoodDistance+1)) + 1000*(ghostCount/(minMANH+1))+10000/(cherries+1) +10000000/(foodLeft+1)+(0.0045*((ghostCount**2))*math.log(1+ghostDistance)), (math.log(((ghostDistance+1)/2))))

# Abbreviation
better = betterEvaluationFunction
