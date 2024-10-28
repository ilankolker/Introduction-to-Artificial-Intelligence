import math

import numpy as np

from board import Board
from search import SearchProblem, ucs
import util


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        h = self.board.board_h - 1
        w = self.board.board_w - 1
        if (state.get_position(0, 0) != -1 and
                state.get_position(0, w) != -1 and
                state.get_position(h, 0) != -1 and
                state.get_position(h, w) != -1):
            return True
        return False

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        sum_of_moves = 0
        for move in actions:
            sum_of_moves += move.piece.get_num_tiles()
        return sum_of_moves


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    h = state.board_h - 1
    w = state.board_w - 1
    uncovered_corners_set = set()
    if state.state[0][0] == -1:
        uncovered_corners_set.add((0, 0))
        if state.state[1][1] == -1:
            uncovered_corners_set.add((1, 1))
    if state.state[0][state.board_w - 1] == -1:
        uncovered_corners_set.add((0, state.board_w - 1))
        if state.state[1][state.board_w - 2] == -1:
            uncovered_corners_set.add((1, state.board_w - 2))
    if state.state[state.board_h - 1][0] == -1:
        uncovered_corners_set.add((state.board_h - 1, 0))
        if state.state[state.board_h - 2][1] == -1:
            uncovered_corners_set.add((state.board_h - 2, 1))
    if state.state[state.board_h - 1][state.board_w - 1] == -1:
        uncovered_corners_set.add((state.board_h - 1, state.board_w - 1))
        if state.state[state.board_h - 2][state.board_w - 2] == -1:
            uncovered_corners_set.add((state.board_h - 2, state.board_w - 2))
    return len(uncovered_corners_set)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        counter = 0
        for x_target, y_target in self.targets:
            if state.get_position(y_target, x_target) == 0:
                counter += 1
        if counter == len(self.targets):
            return True
        return False

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = \
            self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        sum_of_moves = 0
        for move in actions:
            sum_of_moves += move.piece.get_num_tiles()
        return sum_of_moves


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    uncovered_targets = get_uncovered_targets(problem.targets, state)
    if len(uncovered_targets) == 0:
        return 0
    res = len(uncovered_targets)
    if is_board_empty(state):
        starting_point = np.where(state.connected[0] == True)
        starting_point = list(zip(starting_point[0], starting_point[1]))[0]
        res += get_closest_chebyshev(uncovered_targets, starting_point)
    else:
        closest_distance_from_targets = set()
        for target in uncovered_targets:
            closest_distance = get_closest_filled_cell(target, state)
            closest_distance_from_targets.add(closest_distance)
        res += min(closest_distance_from_targets)
    return res - 1


def is_board_empty(state):
    for i in range(state.board_h):
        for j in range(state.board_w):
            if state.state[i][j] == 0:
                return False
    return True

def get_uncovered_targets(targets, state):
    uncovered_targets = set()
    for target in targets:
        x, y = target
        if state.state[x][y] == -1:
            uncovered_targets.add(target)
    return uncovered_targets

def get_closest_filled_cell(target, state):
    min_distance = float('inf')
    x, y = target
    for i in range(state.board_h):
        for j in range(state.board_w):
            if state.state[i, j] == 0:
                distance = max(abs(x - i), abs(y - j))
                if distance < min_distance:
                    min_distance = distance

    return min_distance


def get_closest_chebyshev(targets, start_point):
    min_distance = float('inf')
    x_start, y_start = start_point
    for target in targets:
        x, y = target
        distance = max(abs(x - x_start), abs(y - y_start))
        if distance < min_distance:
            min_distance = distance

    return min_distance


