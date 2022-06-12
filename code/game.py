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
import torch_geometric as torch_g
from mcts_for_train import MCTS_Train
import torch


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
        self.width = int(kwargs.get('width', 9))
        self.height = int(kwargs.get('height', 9))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        self.vertex = []
        edge_u, edge_v = [], []
        for x in range(self.width):
            for y in range(self.height):
                u = self.location_to_move([x,y])
                self.vertex.append([0])
                edge_u.append(self.width*self.height)
                edge_v.append(u)
                edge_v.append(self.width*self.height)
                edge_u.append(u)
                v = 0
                if x > 0:
                    v = self.location_to_move([x-1,y])
                    edge_u.append(u)
                    edge_v.append(v)
                    edge_u.append(v)
                    edge_v.append(u)
                    if y > 0:
                        v = self.location_to_move([x-1,y-1])
                        edge_u.append(u)
                        edge_v.append(v)
                        edge_u.append(v)
                        edge_v.append(u)
                if y > 0:
                    v = self.location_to_move([x,y-1])
                    edge_u.append(u)
                    edge_v.append(v)
                    edge_u.append(v)
                    edge_v.append(u)
                    if x + 1 < self.width:
                        v = self.location_to_move([x+1,y-1])
                        edge_u.append(u)
                        edge_v.append(v)
                        edge_u.append(v)
                        edge_v.append(u)
        self.edge = torch.tensor([edge_u, edge_v], dtype = torch.long).cuda()
        self.states_graph = torch_g.data.Data(x = torch.tensor(self.vertex + [[0]], dtype = torch.float).cuda(), edge_index = self.edge)
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

    def judge(self, player, now, down, up, space = 1):
        """
        return value:
            0: live-3
            1: die-4
            2: live-4
            3: five-in-a-row
        """
        standar_score = [100, 1000, 20000] # 一步擋得掉的(活三和死四和可以創造活4的)，兩步(活4)，直接贏。
        s = ""
        for i in range(now - down*space, now + (up+1)*space, space):
            state = self.states.get(i, -1)
            if state == -1:
                s += "-"
            elif state == player:
                s += "o"
            else:
                s += "x"

        if ("ooooo" in s): return standar_score[2]
        if ("-oooo-" in s): return standar_score[1]
        if ("xoooox" in s): return 0
        if ("xoooo-" in s or "-oooox" in s): return standar_score[0]
        up = min(3, up) + down +1
        down = max(0, down-3)
        s = s[down:up]
        if ("o-oo" in s): return standar_score[0]
        if ("oo-o" in s): return standar_score[0]
        if ("-ooo-" in s): return standar_score[0]
        return 0

    def die_4_live_3_eval(self, player, enemy, new_action):
        """記錄下了這一個旗子之後的分數"""
        width = self.width
        height = self.height
        now_h = new_action // width
        now_w = new_action % width
        
        score = 0.0
        # row
        down = min(4, now_h)
        up = min(4, height-1-now_h)
        score += self.judge(player, new_action, down, up, width)

        # col
        down = min(4, now_w)
        up = min(4, width-1-now_w)
        score += self.judge(player, new_action, down, up)

        # left-up to right-down
        down = min(4, now_h, now_w)
        up = min(4, height-1-now_h, width-1-now_w)
        score += self.judge(player, new_action, down, up, width+1)

        # right-up to left-down 
        down = min(4, now_h, width-1-now_w)
        up = min(4, now_w, height-1-now_h)
        score += self.judge(player, new_action, down, up, width-1)

        return score
        
    def die_4_live_3(self, new_action):
        player = self.get_current_player()
        enemy = 1
        if player == 1: enemy = 2
        Attack_score = self.die_4_live_3_eval(player, enemy, new_action) # 我下這點的分數
        Defense_score = self.die_4_live_3_eval(enemy, player, new_action) # 對方下這點的分數
        if (Attack_score >= 20000.0):   # 會贏直接下
            return 150000.0 
        elif (Defense_score >= 20000.0 ):
            return 100000.0             # 不下這點會讓我直接輸，那我一定要先擋
        elif (Defense_score >= 1000):   # 對方下了會有活4
            return 50000
        else :
            return Attack_score + Defense_score
        

    def update_graph(self, move):
        if self.current_player == 1:
            self.vertex[move][0] = 1
        else:
            self.vertex[move][0] = -1

        self.states_graph = torch_g.data.Data(x = torch.tensor(self.vertex + [[0]], dtype = torch.float).cuda(), edge_index = self.edge)

    def do_move(self, move):
        self.states[move] = self.current_player
        self.update_graph(move)
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
        # pygame.init()
        # screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_WIDTH))
        # self.screen = screen
        # self.font = pygame.font.SysFont("arial", 30)
        self.start_points, self.end_points = make_grid(board.width)
    
    # def clear_screen(self):

    #     # fill board and add gridlines
    #     self.screen.fill(BOARD_BROWN)
    #     for start_point, end_point in zip(self.start_points, self.end_points):
    #         pygame.draw.line(self.screen, BLACK, start_point, end_point)

    #     # add guide dots
    #     guide_dots = [3, self.board.width // 2, self.board.height - 4]
    #     for col, row in itertools.product(guide_dots, guide_dots):
    #         x, y = self.board.colrow_to_xy(col, row)
    #         gfxdraw.aacircle(self.screen, x, y, DOT_RADIUS, BLACK)
    #         gfxdraw.filled_circle(self.screen, x, y, DOT_RADIUS, BLACK)

    #     pygame.display.flip()

    # def graphic(self, board, player1, player2):
    #     # Draw the board and show game info
    #     self.clear_screen()
    #     moved = list(set(range(board.width * board.height)) - set(board.availables))
    #     for m in moved:
    #         col = m // board.width
    #         row = m % board.width
    #         x, y = self.board.colrow_to_xy(col, row)
    #         if player1 == board.states[m]:
    #             gfxdraw.aacircle(self.screen, x, y, STONE_RADIUS, BLACK)
    #             gfxdraw.filled_circle(self.screen, x, y, STONE_RADIUS, BLACK)
    #     for m in moved:
    #         col = m // board.width
    #         row = m % board.width
    #         x, y = board.colrow_to_xy(col, row)
    #         if player2 == board.states[m]:
    #             gfxdraw.aacircle(self.screen, x, y, STONE_RADIUS, WHITE)
    #             gfxdraw.filled_circle(self.screen, x, y, STONE_RADIUS, WHITE)
    #     pygame.display.flip()
    
    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')
    
    def start_play(self, player1, player2, start_player=0, is_shown=0):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board()
        p1, p2 = self.board.players
        if start_player == 1:
            p1, p2 = p2, p1

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
            # self.board.die_4_live_3(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                if start_player == 1 and winner != -1:  
                    if p1 == winner:
                        return p2
                    else:
                        return p1
                else:
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

    def start_coach_play(self, player, coach , is_shown=0, temp=1e-3):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        if np.random.randint(2) == 0:
            p1, p2 = p2, p1
        
        player.set_player_ind(p1)
        coach.set_player_ind(p2)
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
            states.append(self.board.states_graph)
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
                        if winner == p1:
                            print("Game end. Winner is player: student")
                        else:
                            print("Game end. Winner is player: coach")
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
