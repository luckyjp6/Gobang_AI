# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import copy
import pickle
from game import Board, Game
from mcts_for_train import MCTS_Train
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
import pygame
import sys
from os.path import exists

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p
    
    def is_valid_click(self, col, row, move, board):
        """Check if placing a stone at (col, row) is valid on board
        Args:
            col (int): column number
            row (int): row number
            board (object): board grid (size * size matrix)
        Returns:
            boolean: True if move is valid, False otherewise
        """
        # TODO: check for ko situation (infinite back and forth)
        if col < 0 or col >= board.width:
            return False
        if row < 0 or row >= board.height:
            return False
        return move in board.availables

    def handle_click(self, board):
        x, y = pygame.mouse.get_pos()
        col, row = board.xy_to_colrow(x, y)
        move = board.location_to_move([col,row])
        if not self.is_valid_click(col, row, move, board):
            #pygame.mixer.Sound("wav/zoink.wav").play()
            return self.get_action(board)
        #pygame.mixer.Sound("wav/click.wav").play()
        return move

    def get_action(self, board):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.MOUSEBUTTONUP:
                    return self.handle_click(board)
                if event.type == pygame.QUIT:
                    sys.exit()

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 9, 9
    # model_file = 'best_policy_8_8_5.model'
    model_file = 'best_policy.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        # try:
        #     policy_param = pickle.load(open(model_file, 'rb'))
        # except:
        #     policy_param = pickle.load(open(model_file, 'rb'),
        #                                encoding='bytes')  # To support python3
        # best_policy = PolicyValueNet(width, height, policy_param)
        # best_policy = PolicyValueNetNumpy(width, height, policy_param)
        best_policy = PolicyValueNet(width, height, 'best_policy.model')
        mcts_player = MCTS_Train(
                                 c_puct=5,
                                 n_playout=500)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
