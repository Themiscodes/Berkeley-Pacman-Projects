# Berkeley Pac-Man Projects

This is my solution to the Pac-Man assignments for UC Berkeley's Artificial Intelligence course, CS 188 of Spring 2021. The purpose of these projects was to learn foundational AI concepts, such as informed state-space search, probabilistic inference, and reinforcement learning. These concepts underly real-world application areas such as natural language processing, computer vision, and robotics.

## Project I - Search

![s2](https://user-images.githubusercontent.com/73662635/180445539-35d72dda-b1d3-4564-8927-999ccfa3911d.png)

1. In [depthFirstSearch](search/search.py#L103) I implemented a graph search version of the DFS algorithm, which avoids expanding any already visited states. It returns a list of actions (the path) that reaches the goal.
```
$ python pacman.py -l bigMaze -z .5 -p SearchAgent
```

2. In [breadthFirstSearch](search/search.py#L134) is my implementation of the BFS algorithm. It is similar to the DFS, with the difference of using a queue instead of a stack.
```
$ python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
```

3. In [aStarSearch](search/search.py#L167) I used a Priority Queue. In [evaluationFunction](search/search.py#L164) I calculate the cost of a state plus the heuristic cost, using the heuristic function that it takes as an argument.

```
$ python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

The three implementations described above use the following Graph Search algorithm:

```
frontier = {startNode}
expanded = {}
while frontier is not empty:
    node = frontier.pop()
    if isGoal(node):
        return path_to_node
    if node not in expanded:
        expanded.add(node)
        for each child of node's children:
            frontier.push(child)
return failed
```

4.  In [CornersProblem](search/searchAgents.py#L289) I defined a state representation for the search problem of finding the shortest path through the maze that touches all four corner. I decided here to keep a dictionary, heuristicInfo, for the information that is needed by the heuristic.

```
$ python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
```

5. In [cornersHeuristic](search/searchAgents.py#L415) I implemented a heuristic for the CornersProblem. I chose to use the euclidean distance to the corners that still contain a food dot, but taking into account the walls towards that corner. In heuristicInfo I save the corners reached in a specific state, to penalize a state accordingly and reduce the number of expansions.

```
$ python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```

> Heuristics take search states and return numbers that estimate the cost to a nearest goal. To be admissible, the heuristic values must be lower bounds on the actual shortest path cost to the nearest goal. To be consistent, it must additionally hold that if an action has cost c, then taking that action can only cause a drop in heuristic of at most c.

6. In [FoodSearchProblem](search/searchAgents.py#L484) I defined the search problem of finding the path that collects all the Pac-Man food of a layout in as few steps as possible. In [foodHeuristic](search/searchAgents.py#L562) I implemented the heuristic for this problem, which is similar to the cornersHeuristic. Only this time, I use the euclidean distance to the nearest food dot to Pac-Man, that has at the same time the less obstacles (walls).

```
$ python pacman.py -l mediumSearch -p AStarFoodSearchAgent
```

7. Finally, in [ClosestDotSearchAgent](search/searchAgents.py#L630) I implemented an agent that greedily eats the closest dot. It uses the BFS to locate the closest food dot to Pac-Man and returns that path.

```
$ python pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5
```

Below each implementation described above I have an example of execution to test the specific function. In order to perform all the test cases run:
```
$ python autograder.py
```

## Project II - Multi-Agent Search

![p4](https://user-images.githubusercontent.com/73662635/180445504-03628120-a51d-4bc4-880b-8d7ce593546f.png)

1. In [ReflexAgent](multiagent/multiAgents.py#L25) I implement an agent that considers both food and ghost locations. To achieve that in my [evaluationFunction](multiagent/multiAgents.py#L54) I used the distance of the nearest ghost, food, as well as total distance of every food, so that "clusters" of food are favored. Then, I experimented till I found the optimal weights for these below:
```
math.copysign(ghostCount/(minMANH+1) 
+ ghostCount/(maxFoodDistance+1) 
+ 100000/(foodLeft+1)
+ (0.0045*((ghostCount**2))*math.log(1+ghostDistance)), 
(math.log(((ghostDistance+1)/3))))
```
- I wanted to recreate a kind of step function, in that the values are negative when a ghost is in close proximity. To achieve that I used the copy-sign function which returns the magnitude of the first argument, with the sign of the second argument. This way, by having as a second argument the logarithm of the distance of the nearest ghost + 1 divided by 3, as soon as Pac-Man is within 2 moves of a ghost it becomes negative. Finally, in order to follow a more "aggressive" strategy I incentivize Pac-Man by returning high values to eat the cherry and then the ghosts. 

```
$ python pacman.py --frameTime 0 -p ReflexAgent -k 1
```

2. In [MinimaxAgent](multiagent/multiAgents.py#L149) is my implementation of an agent using the mini-max algorithm for any number of ghosts. Here, I wrote the minimax, mini and maxi functions, which are called recursively. A single level of the search is considered to be one Pac-Man move and all the ghosts’ responses, so depth 2 search involves Pac-Man and each ghost moving twice.

```
$ python autograder.py -q q2
```

3. In [AlphaBetaAgent](multiagent/multiAgents.py#L206) I implement an agent that uses alpha-beta pruning to more efficiently explore the minimax tree. I utilized the functions I wrote for mini-max adding the pruning of a smaller value from alpha and a larger value from beta respectively, to stop exploring a certain branch.

```
$ python autograder.py -q q3
```

4. In [ExpectimaxAgent](multiagent/multiAgents.py#L253) I implement expectimax, which is useful for modeling probabilistic behavior of agents who may make suboptimal choices, where minimax and alpha-beta assume that the adversary makes optimal decisions. In [expectiMax](multiagent/multiAgents.py#L268) I defined the function which recursively calls itself and in the case of the adversary nodes uses the probability. This way, if Pac-Man  perceives that he could be trapped, but might escape to grab a few more pieces of food, he’ll at least try.

```
$ python autograder.py -q q4
```

5. In [betterEvaluationFunction](multiagent/multiAgents.py#L292) I followed a similar approach to the evaluationFunction I implemented for the reflex agent, but evaluating states rather than actions. I also take into account aside from the distances and food clusters, the remaining cherries as well.
```
math.copysign(10*(ghostCount/(maxFoodDistance+1)) 
+ 1000*(ghostCount/(minMANH+1))
+ 10000/(cherries+1) 
+ 10000000/(foodLeft+1)
+ (0.0045*((ghostCount**2))*math.log(1+ghostDistance)), 
(math.log(((ghostDistance+1)/2))))
```
- The weights, as it can be seen above, are adjusted accordingly for this agent. I again used the same trick with the copy-sign, as well as the "chase mode" to incentivize Pac-Man to eat the cherry and hunt the ghosts, so that the final score he achieves is higher.
```
$ python autograder.py -q q5
```

## Notes

The Pac-Man projects are written in pure Python 3.6 and do not depend on any packages external to a standard Python distribution.

The Syllabus for this course can be found in [CS 188 Spring 2021](https://inst.eecs.berkeley.edu/~cs188/sp21/).
