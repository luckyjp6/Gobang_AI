# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""
from sklearn.utils import shuffle
# import torch
# from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from mcts_for_train import MCTS_Train
from game import Board, Game
from mcts_pure import MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from scalable_policy_net_pytorch import PolicyValueNet
import torch_geometric as torch_g
import csv
save_timing = 100

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 9
        self.board_height = 9
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512 # mini-batch size for training  #512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 70 #打架次數100
        self.game_batch_num = 7000 #自我對亦次數
        self.best_win_ratio = 0.0
        self.least_lose = 10
        self.pure_mcts_playout_num = 1000 # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)


    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            self.data_buffer.extend(play_data)
    
    def collect_coachplay_data(self, n_games=1, coach = None):
        """collect coach-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_coach_play(self.mcts_player, coach, is_shown=0) # 這個 is shown 印出過程棋盤

            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = torch_g.data.Batch.from_data_list([data[0] for data in mini_batch])
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10, Enemy = None):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,Enemy,start_player=i % 2,is_shown=0) 
            win_cnt[winner] += 1
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_cnt

    def run(self):
        """run the training pipeline"""
        try:
            time = 0
            Coach = MCTS_Train( c_puct=5, n_playout= 50)
            self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=0)
            levels = [150, 350]
            level = 0
            # 二維 list，存要寫出的資料
            train_loss_result = []
            train_entropy_result = []
            train_win_times_result = []
            train_loss_time_result = []
            train_tie_time_result = []

            while True:
                self.collect_coachplay_data(self.play_batch_size, Coach)
                print("batch i:{}, episode_len:{}".format(
                        time+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                    # record system - train
                    tmp_list = []
                    tmp_list.extend((time,loss))
                    train_loss_result.append(tmp_list)

                    tmp_list = []
                    tmp_list.extend((time, entropy))
                    train_entropy_result.append(tmp_list)

                    if (time % save_timing ==0):
                        with open('train_result/loss/loss_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'loss'])
                            for row in train_loss_result:
                                writer.writerow(row)
                            train_loss_result.clear()

                        with open('train_result/entropy/entropy_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'entropy'])
                            for row in train_entropy_result:
                                writer.writerow(row)
                            train_entropy_result.clear()


                
                if (time+1) % self.check_freq == 0:
                    win_cnt = self.policy_evaluate(n_games = 10, Enemy = Coach)
                    self.policy_value_net.save_model('./current_policy.model')

                    # record system - train
                    tmp_list = []
                    tmp_list.extend((time, win_cnt[1])) # win
                    train_win_times_result.append(tmp_list)

                    tmp_list = []
                    tmp_list.extend((time, win_cnt[2])) # loss
                    train_loss_time_result.append(tmp_list)

                    tmp_list = []
                    tmp_list.extend((time, win_cnt[-1])) #tie
                    train_tie_time_result.append(tmp_list)
                    
                    if (time%save_timing == 0):
                        with open('train_result/win_times/win_times_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'win_times'])
                            for row in train_win_times_result:
                                writer.writerow(row)
                            train_win_times_result.clear()

                        with open('train_result/loss_times/loss_times_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'loss_times'])
                            for row in train_loss_times_result:
                                writer.writerow(row)
                            train_loss_times_result.clear()

                        with open('train_result/tie_times/tie_times_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'tie_times'])
                            for row in train_tie_times_result:
                                writer.writerow(row)
                            train_tie_times_result.clear()

                    if win_cnt[2] < self.least_lose:
                        print("New best policy!!!!!!!!")
                        self.least_lose = win_cnt[2] # -1是平手，1是贏，2是輸
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.least_lose == 0):
                            if level < len(levels):
                                print("NEXT LEVEL!!!")
                                print('level:', level)
                                self.least_lose = 10
                                level = level + 1
                                Coach = MCTS_Train( c_puct = 5, n_playout = levels[level])
                            else:
                                break
                time = time + 1 


            
            
            self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

            self.check_freq = 150

            # 二維 list，存要寫出的資料
            self_fight_loss_result = []
            self_fight_entropy_result = []
            self_fight_win_times_result = []
            self_fight_loss_time_result = []
            self_fight_tie_time_result = []

            for time in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        time+1, self.episode_len))
                if len(self.data_buffer) > 5:
                    loss, entropy = self.policy_update()
                    # record system - self fight
                    tmp_list = []
                    tmp_list.extend((time,loss))
                    self_fight_loss_result.append(tmp_list)
                    
                    tmp_list = []
                    tmp_list.extend((time, entropy))
                    self_fight_entropy_result.append(tmp_list)
                    if (time %save_timing ==0):
                        with open('self_fight_result/loss/loss_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'loss'])
                            for row in self_fight_loss_result:
                                writer.writerow(row)
                            self_fight_loss_result.clear()

                        with open('self_fight_result/entropy/entropy_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'entropy'])
                            for row in self_fight_entropy_result:
                                writer.writerow(row)
                            self_fight_entropy_result.clear()

                # check the performance of the current model,
                # and save the model params
                BEST_policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file='best_policy.model')
                BEST = MCTSPlayer(policy_value_function=BEST_policy_value_net, c_puct=5, n_playout=400, is_selfplay=1)
                if (time+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(time+1))
                    win_cnt = self.policy_evaluate(10, BEST)
                    
                    # record system - self fight   
                    tmp_list = []
                    tmp_list.extend((time, win_cnt[1])) # win
                    self_fight_win_time_result.append(tmp_list)

                    tmp_list = []
                    tmp_list.extend((time, win_cnt[2])) # loss
                    self_fight_loss_time_result.append(tmp_list)

                    tmp_list = []
                    tmp_list.extend((time, win_cnt[-1])) #tie
                    self_fight_tie_time_result.append(tmp_list)

                    if (time % save_timing == 0):
                        with open('self_fight_result/win_times/win_times_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'win_times'])
                            for row in self_fight_win_times_result:
                                writer.writerow(row)
                            self_fight_win_times_result.clear()

                        with open('self_fight_result/loss_times/loss_time_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'loss_times'])
                            for row in self_fight_loss_time_result:
                                writer.writerow(row)
                            self_fight_loss_time_result.clear()

                        with open('self_fight_result/tie_times/tie_time_result'+str(int(time/save_timing))+'.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile,delimiter=' ')
                            writer.writerow(['time', 'tie_times'])
                            for row in self_fight_tie_time_result:
                                writer.writerow(row)
                            self_fight_tie_time_result.clear()

                    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / 10
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio >= 0.5:
                        print("New best policy!!!!!!!!")
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        BEST_policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file='best_policy.model')
                        BEST = MCTSPlayer(policy_value_function=BEST_policy_value_net, c_puct=5, n_playout=400, is_selfplay=1)

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline() #"current_policy.model"
    training_pipeline.run()
