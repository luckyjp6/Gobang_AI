# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import pygame
import copy
from pygame import gfxdraw
import itertools

from mcts_for_train import MCTS_Train


# Constants
BOARD_BROWN = (199, 105, 42)
BOARD_WIDTH = 1000
BOARD_BORDER = 75
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
STONE_RADIUS = 22
DOT_RADIUS = 4

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
    

    def xy_to_colrow(self, x, y):
        """Convert x,y coordinates to column and row number
        Args:
            x (float): x position
            y (float): y position
            size (int): size of grid
        Returns:
            Tuple[int, int]: column and row numbers of intersection
        """
        inc = (BOARD_WIDTH - 2 * BOARD_BORDER) / (self.width - 1)
        x_dist = x - BOARD_BORDER
        y_dist = y - BOARD_BORDER
        col = int(round(x_dist / inc))
        row = int(round(y_dist / inc))
        return col, row


    def colrow_to_xy(self, col, row):
        """Convert column and row numbers to x,y coordinates
        Args:
            col (int): column number (horizontal position)
            row (int): row number (vertical position)
            size (int): size of grid
        Returns:
            Tuple[float, float]: x,y coordinates of intersection
        """
        inc = (BOARD_WIDTH - 2 * BOARD_BORDER) / (self.width - 1)
        x = int(BOARD_BORDER + col * inc)
        y = int(BOARD_BORDER + row * inc)
        return x, y

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]
    
    def check(self, player, down, space = 1):
        states = self.states
        #print("down: ", down, "space: ", space)
        up = down + 6*space
        new_down = down - 3*space
        if new_down < 0: new_down = down%space
        #print("new_down: ", new_down)
        Count = 0
        C = []
        End = []

        for i in range(new_down, up, space):
            if states.get(i, -1) != player: 
                if (Count >= 3):
                    End.append(i)
                    C.append(Count)
                Count = 0
            else: Count += 1

        if len(C) == 1: return C[0], End[0]
        return -1, -1    
    
    def judge(self, right, left, enemy, action, end, num):
        """
        return value:
            0: unuseful (like xoooox or xooox or less then 3 in a row)
            1: live-3
            2: die-4
            3: live-4
        """
        if (num == -1): return 0
        if not (end > action) and (end-num <= action): return 0
        if (right == enemy and left == enemy): return 0

        # for 4 in a row
        if num == 4:
            # new action is included
            if (right == enemy or left == enemy): return 2
            else: return 3
        elif num == 3:
            if (right == enemy or left == enemy): return 0
            else: return 1
    
    def die_4_live_3_eval(self, player, enemy, new_action):
        states = copy.deepcopy(self.states)
        width = self.width
        height = self.height
        now_h = new_action // width
        now_w = new_action % width

        right = 0
        left = 0
        score = 0.0
        standar_score = [0.0, 5.0, 1.0, 1000.0]
        """
            0: unuseful (like xoooox or xooox or less then 3 in a row)
            1: live-3
            2: die-4
            3: live-4
        """

        # row
        num, end = self.check(player, new_action)
        if (num != -1):
            if end-num-1 // width != now_h: left = enemy
            else: left = states.get(end-num-1, -1)
            if end // width != now_h: right = enemy
            else: right = states.get(end, -1)

        g = self.judge(right, left, enemy, new_action, end, num)
        score += standar_score[g]

        # col
        num, end = self.check(player, new_action, width)
        if (num != -1):
            if ((end - (num+1)*width) % width != now_w): left = enemy
            else: left = states.get(end - (num+1)*width, -1)
            if end % width != now_w: right = enemy
            else: right = states.get(end, -1)

        g = self.judge(right, left, enemy, new_action, end, num)
        score += standar_score[g]
        
        # right-up to left-down 
        num, end = self.check(player, new_action, width+1)
        if (num != -1):
            left = end - (num+1)*(width+1)
            if ( (left // width - now_h) != (left % width - now_w)): left = enemy
            else: left = states.get(left, -1)
            if ( (end // width - now_h) != (end % width - now_w)): right = enemy
            else: right = states.get(end, -1)

        g = self.judge(right, left, enemy, new_action, end, num)
        score += standar_score[g]

        # left-up to right-down 
        num, end = self.check(player, new_action, width-1)
        if (num != -1):
            left = end - (num+1)*(width-1)
            if ( (left // width - now_h) != -(left % width - now_w)): left = enemy
            else: left = states.get(left, -1)
            if ( (end // width - now_h) != -(end % width - now_w)): right = enemy
            else: right = states.get(end, -1)

        g = self.judge(right, left, enemy, new_action, end, num)
        score += standar_score[g]

        return score
        
    def die_4_live_3(self, new_action):
        player = self.get_current_player()
        enemy = 1
        if player == 1: enemy = 2
        Attack_score = self.die_4_live_3_eval(player, enemy, new_action)
        Defense_score = -self.die_4_live_3_eval(enemy, player, new_action)
        return Attack_score + Defense_score

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


def make_grid(size):
        """Return list of (start_point, end_point pairs) defining gridlines
        Args:
            size (int): size of grid
        Returns:
            Tuple[List[Tuple[float, float]]]: start and end points for gridlines
        """
        start_points, end_points = [], []

        # vertical start points (constant y)
        xs = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
        ys = np.full((size), BOARD_BORDER)
        start_points += list(zip(xs, ys))

        # horizontal start points (constant x)
        xs = np.full((size), BOARD_BORDER)
        ys = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
        start_points += list(zip(xs, ys))

        # vertical end points (constant y)
        xs = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
        ys = np.full((size), BOARD_WIDTH - BOARD_BORDER)
        end_points += list(zip(xs, ys))

        # horizontal end points (constant x)
        xs = np.full((size), BOARD_WIDTH - BOARD_BORDER)
        ys = np.linspace(BOARD_BORDER, BOARD_WIDTH - BOARD_BORDER, size)
        end_points += list(zip(xs, ys))

        return (start_points, end_points)

class Game(object):
    """game server"""


    def __init__(self, board, **kwargs):
        self.board = board
        pygame.init()
        screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_WIDTH))
        self.screen = screen
        self.font = pygame.font.SysFont("arial", 30)
        self.start_points, self.end_points = make_grid(board.width)

    def clear_screen(self):

        # fill board and add gridlines
        self.screen.fill(BOARD_BROWN)
        for start_point, end_point in zip(self.start_points, self.end_points):
            pygame.draw.line(self.screen, BLACK, start_point, end_point)

        # add guide dots
        guide_dots = [3, self.board.width // 2, self.board.height - 4]
        for col, row in itertools.product(guide_dots, guide_dots):
            x, y = self.board.colrow_to_xy(col, row)
            gfxdraw.aacircle(self.screen, x, y, DOT_RADIUS, BLACK)
            gfxdraw.filled_circle(self.screen, x, y, DOT_RADIUS, BLACK)

        pygame.display.flip()

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        self.clear_screen()
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        for m in moved:
            col = m // board.width
            row = m % board.width
            x, y = self.board.colrow_to_xy(col, row)
            if player1 == board.states.get(m, -1):
                gfxdraw.aacircle(self.screen, x, y, STONE_RADIUS, BLACK)
                gfxdraw.filled_circle(self.screen, x, y, STONE_RADIUS, BLACK)
        for m in moved:
            col = m // board.width
            row = m % board.width
            x, y = board.colrow_to_xy(col, row)
            if player2 == board.states.get(m, -1):
                gfxdraw.aacircle(self.screen, x, y, STONE_RADIUS, WHITE)
                gfxdraw.filled_circle(self.screen, x, y, STONE_RADIUS, WHITE)
        pygame.display.flip()

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

    def start_coach_play(self, player, coach , is_shown=1, temp=1e-3):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            if p1 == self.board.current_player:
                move, move_probs = player.get_action(self.board,
                                                     temp=temp,
                                                     return_prob=1)
            else:
                move, move_probs = coach.get_action(self.board, 1)
                # move_probs = np.zeros(self.board.width*self.board.height)
                # move_probs[move] = 1
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
