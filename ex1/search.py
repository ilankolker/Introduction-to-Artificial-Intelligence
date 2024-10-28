"""
In search.py, you will implement generic search algorithms
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    start_state = problem.get_start_state()
    expanded = set()
    fringe = util.Stack()
    fringe.push((start_state, []))
    current_state = None
    # looping
    while not fringe.isEmpty():
        current_state = fringe.pop()
        if problem.is_goal_state(current_state[0]):
            break
        else:
            successors = problem.get_successors(current_state[0])
            if current_state[0] not in expanded:
                expanded.add(current_state[0])
                for successor in successors:
                    if successor[0] not in expanded:
                        # added if statement 27.05
                        if problem.is_goal_state(successor[0]):
                            return current_state[1] + [successor[1]]
                        fringe.push((successor[0],
                                     current_state[1] + [successor[1]]))

    return current_state[1]


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    start_state = problem.get_start_state()
    expanded = set()
    fringe = util.Queue()
    fringe.push((start_state, [], 0))
    current_state = None
    expanded.add(start_state)
    # looping
    while not fringe.isEmpty():
        current_state = fringe.pop()
        if problem.is_goal_state(current_state[0]):
            break
        else:
            successors = problem.get_successors(current_state[0])
            for successor in successors:
                if successor[0] not in expanded:
                    # added if statement 27.05
                    if problem.is_goal_state(successor[0]):
                        return current_state[1] + [successor[1]]
                    fringe.push((successor[0],
                                 current_state[1] + [successor[1]]))
                    expanded.add(successor[0])

    return current_state[1]

def triple_priority_func(item):
    # Assuming the item is a tuple and the price is at index 1
    return item[2]

def quad_priority_func(item):
    # Assuming the item is a tuple and the price is at index 1
    return item[3]

def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    start_state = problem.get_start_state()
    fringe = util.PriorityQueueWithFunction(triple_priority_func)
    fringe.push(Triple(start_state, [], 0))
    # expanded = [start_state]
    expanded = set()
    current_state = None
    # looping
    while not fringe.isEmpty():
        current_state = fringe.pop()
        if problem.is_goal_state(current_state[0]):
            break
        else:
            if current_state[0] not in expanded:
                expanded.add(current_state[0])
                successors = problem.get_successors(current_state[0])
                for successor in successors:
                    if successor[0] not in expanded:
                        fringe.push(Triple(successor[0],
                                     current_state[1] + [successor[1]],
                                     current_state[2] + successor[2]))

    return current_state[1]


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    start_state = problem.get_start_state()
    fringe = util.PriorityQueueWithFunction(quad_priority_func)
    fringe.push(Quad(start_state, [], 0, heuristic(start_state, problem)))
    expanded = set()
    current_state = None
    # looping
    while not fringe.isEmpty():
        current_state = fringe.pop()
        if problem.is_goal_state(current_state[0]):
            break
        else:
            if current_state[0] not in expanded:
                expanded.add(current_state[0])
                successors = problem.get_successors(current_state[0])
                for successor in successors:
                    fringe.push(Quad(successor[0],
                                 current_state[1] + [successor[1]],
                                 current_state[2] + successor[2],
                                 current_state[2] + successor[2] + heuristic(successor[0], problem)))

    return current_state[1]


class Triple:
    def __init__(self, state, path, price):  # Gets state, path to goal, price so far
        self.triple = state, path, price

    def __lt__(self, other):
        return self[3] < other[3]

    def __getitem__(self, index):
        return self.triple[index]

class Quad:
    def __init__(self, state, path, price, predicted_price):  # Gets state, path to goal, price so far
        self.triple = state, path, price, predicted_price

    def __lt__(self, other):
        return True

    def __getitem__(self, index):
        return self.triple[index]

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
