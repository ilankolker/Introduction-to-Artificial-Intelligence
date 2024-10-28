import math

import numpy as np
import abc
import util
from game import Agent, Action



class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        "*** YOUR CODE HERE ***"
        best = -1
        number_of_corners = 4
        for _ in range(number_of_corners):
            current = 0
            # Check rows
            for row in range(len(board)):
                for col in range(len(board[0]) - 1):
                    if board[row][col] >= board[row][col + 1]:
                        current += 1
                    if board[row][col] <= board[row][col + 1]:
                        current += 1
            # Check columns
            for col in range(len(board)):
                for row in range(len(board[0]) - 1):
                    if board[row][col] >= board[row + 1][col]:
                        current += 1
                    if board[row][col] <= board[row + 1][col]:
                        current += 1
            if current > best:
                best = current

        return best

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_minmax_action(self, game_state, depth, agent_index):
        actions = game_state.get_legal_actions(agent_index)
        if depth == 0 or len(actions) == 0:
            return self.evaluation_function(game_state), Action.STOP

        if agent_index == 0:  # our agent, maximizer
            final_action = Action.STOP
            max_eval = -math.inf
            for action in game_state.get_legal_actions(0):
                successor = game_state.generate_successor(agent_index, action)
                eval, _ = self.get_minmax_action(successor, depth, 1)
                if eval > max_eval:
                    max_eval = eval
                    final_action = action
            return max_eval, final_action

        if agent_index == 1:  # our agent, minimizer
            final_action = Action.STOP
            min_eval = math.inf
            for action in game_state.get_legal_actions(1):
                successor = game_state.generate_successor(agent_index, action)
                eval, _ = self.get_minmax_action(successor, depth - 1, 0)
                if eval < min_eval:
                    min_eval = eval
                    final_action = action
            return min_eval, final_action

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        return self.get_minmax_action(game_state, self.depth, 0)[1]




class AlphaBetaAgent(MultiAgentSearchAgent):

    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def get_alpha_beta(self, game_state, depth, alpha, beta, agent_index):
        actions = game_state.get_legal_actions(agent_index)
        if depth == 0 or len(actions) == 0:
            return self.evaluation_function(game_state), Action.STOP

        if agent_index == 0:  # our agent, maximizer
            final_action = Action.STOP
            max_eval = -math.inf
            for action in game_state.get_legal_actions(0):
                successor = game_state.generate_successor(agent_index, action)
                eval, _ = self.get_alpha_beta(successor, depth, alpha, beta, 1)
                if eval > max_eval:
                    max_eval = eval
                    final_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, final_action

        if agent_index == 1:  # our agent, minimizer
            final_action = Action.STOP
            min_eval = math.inf
            for action in game_state.get_legal_actions(1):
                successor = game_state.generate_successor(agent_index, action)
                eval, _ = self.get_alpha_beta(successor, depth - 1, alpha, beta, 0)
                if eval < min_eval:
                    min_eval = eval
                    final_action = action
                beta = min(eval, beta)
                if beta <= alpha:
                    break
            return min_eval, final_action

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        return self.get_alpha_beta(game_state, self.depth, -math.inf, math.inf, 0)[1]



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """
    def get_expectimax_action(self, game_state, depth, agent_index):
        actions = game_state.get_legal_actions(agent_index)
        if depth == 0 or len(actions) == 0:
            return self.evaluation_function(game_state), Action.STOP

        if agent_index == 0:  # our agent, maximizer
            final_action = Action.STOP
            max_eval = -math.inf
            for action in game_state.get_legal_actions(agent_index):
                successor = game_state.generate_successor(agent_index, action)
                eval, _ = self.get_expectimax_action(successor, depth, 1)
                if eval > max_eval:
                    max_eval = eval
                    final_action = action
            return max_eval, final_action

        if agent_index == 1:  # our agent, minimizer
            final_action = Action.STOP
            response = 0
            num_of_actions = len(game_state.get_legal_actions(agent_index))
            for action in game_state.get_legal_actions(agent_index):
                successor = game_state.generate_successor(agent_index, action)
                eval, _ = self.get_expectimax_action(successor, depth - 1, 0)
                response += eval / num_of_actions
            return response, final_action

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        return self.get_expectimax_action(game_state, self.depth, 0)[1]


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: This function evaluates the current game state by calculating
    a weighted sum of five heuristic scores: monotonicity (board consistency),
    smoothness (difference between adjacent tiles), free tiles (empty spaces),
    max tile in corner (the highest tile placement), and merging potential
    (adjacent tiles with the same value). These scores guide the AI to favor
    more promising game states.
    """
    "*** YOUR CODE HERE ***"
    return score_monotonicity(current_game_state.board) * 1.0\
        + score_smoothness(current_game_state.board) * 0.1\
        + score_free_tiles(current_game_state.board) * 2.7 \
        + score_max_tile_in_corner(current_game_state.board) * 1.0 \
        + score_merging_potential(current_game_state.board) * 0.5


def score_monotonicity(current_game_state):
    """
        Calculates the monotonicity score of the game state in four
         directions (original and three rotations).

        Parameters:
        - current_game_state (list of list): 2D list representing the current
         state of the game.

        Returns:
        - int: Maximum monotonicity score among all directions.
        """
    scores = []
    number_of_corners = 4
    for _ in range(number_of_corners):
        current_score = 0
        for row in range(len(current_game_state)):
            for col in range(len(current_game_state[0]) - 1):
                if current_game_state[row][col] >= current_game_state[row][col + 1]:
                    current_score += 1
        for col in range(len(current_game_state)):
            for row in range(len(current_game_state[0]) - 1):
                if current_game_state[row][col] >= current_game_state[row + 1][col]:
                    current_score += 1
        scores.append(current_score)
        current_game_state = rotate_90_clockwise(current_game_state)
    return max(scores)


def score_smoothness(current_game_state):
    """
       Calculates the smoothness score of the game state,
       which measures the difference between adjacent tile values.

       Parameters:
       - current_game_state (list of list): 2D list representing the
        current state of the game.

       Returns:
       - int: Smoothness score of the game state.
       """
    smoothness = 0
    for row in range(len(current_game_state)):
        for col in range(len(current_game_state[0]) - 1):
            curr_position_val = current_game_state[row][col]
            right_position_val = current_game_state[row][col + 1]
            smoothness -= abs(curr_position_val - right_position_val)
    for col in range(len(current_game_state)):
        for row in range(len(current_game_state[0]) - 1):
            curr_position_val = current_game_state[row][col]
            down_position_val = current_game_state[row + 1][col]
            smoothness -= abs(curr_position_val - down_position_val)
    return smoothness


def score_free_tiles(current_game_state):
    """
        Counts the number of empty tiles in the game state.

        Parameters:
        - current_game_state (list of list): 2D list representing the current
         state of the game.

        Returns:
        - int: Number of free tiles in the game state.
        """
    return np.sum(current_game_state == 0)


def score_max_tile_in_corner(current_game_state):
    """
        Determines the maximum tile value present in the corners of
         the game board.

        Parameters:
        - current_game_state (list of list): 2D list representing the
         current state of the game.

        Returns:
        - int: Value of the largest tile in the corners (top-left, top-right,
         bottom-left, bottom-right)
               of the game board, or 0 if the maximum tile is not in any corner.
        """
    max_tile_of_board = max(max(row) for row in current_game_state)
    corners = [current_game_state[0][0], current_game_state[0][3],
               current_game_state[3][0], current_game_state[3][3]]
    return max_tile_of_board if max_tile_of_board in corners else 0


def score_merging_potential(current_game_state):
    """
        Computes the number of potential merges (pairs of adjacent tiles with
         the same value) in the game state.

        Parameters:
        - current_game_state (list of list): 2D list representing the current
         state of the game.

        Returns:
        - int: Number of potential merges in the game state.
        """
    merging_potential = 0
    for row in range(len(current_game_state)):
        for col in range(len(current_game_state[0]) - 1):
            if current_game_state[row][col] == current_game_state[row][col + 1]:
                merging_potential += 1
    for col in range(len(current_game_state)):
        for row in range(len(current_game_state[0]) - 1):
            if current_game_state[row][col] == current_game_state[row + 1][col]:
                merging_potential += 1
    return merging_potential


def rotate_90_clockwise(current_game_state):
    """
        Rotates the 4x4 game state matrix 90 degrees clockwise.

        Parameters:
        - current_game_state (list of list): 2D list representing
         the current state of the game.

        Returns:
        - list of list: Rotated 2D list of the game state.
        """
    n = 4  # matrix is 4 x 4
    rotated_state = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            rotated_state[j][n - 1 - i] = current_game_state[i][j]

    return rotated_state


# Abbreviation
better = better_evaluation_function
