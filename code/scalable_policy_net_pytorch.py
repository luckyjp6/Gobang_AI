# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch_geometric as torch_g


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.hidden_dim = 256
        # action_size = board_width * board_height
        
        # GNN layers
        self.GIN1 = torch_g.nn.GINConv(nn.Linear(1, self.hidden_dim))
        self.GIN2 = torch_g.nn.GINConv(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.GIN3 = torch_g.nn.GINConv(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.Fc1 = nn.Linear(self.hidden_dim*3+1, self.hidden_dim*2)
        self.Fc2 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.fc_bn1 = nn.BatchNorm1d(self.hidden_dim*2)
        self.fc_bn2 = nn.BatchNorm1d(self.hidden_dim)

        self.act_Fc = nn.Linear(self.hidden_dim, 1)
        self.val_Fc = nn.Linear(self.hidden_dim, 1)
################################################################################################################
    def forward(self, vertex, edge_index, batch_size):
        # common layers

        x = F.relu(self.norm(self.GIN1(vertex, edge_index)))
        y = F.relu(self.norm(self.GIN2(x, edge_index)))
        z = F.relu(self.norm(self.GIN3(y, edge_index)))
        
        CONCAT = torch.cat([vertex, x, y, z], dim=-1)
        # action policy layers
        
        z = F.dropout(F.relu(self.fc_bn1(self.Fc1(CONCAT))))
        z = F.dropout(F.relu(self.fc_bn2(self.Fc2(z))))
        act = F.log_softmax(self.act_Fc(z), dim=0)
        val = torch.tanh(torch_g.nn.global_mean_pool(self.val_Fc(z), batch_size))
        return act, val
################################################################################################################

class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        # Adaptive Moment Estimation
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            if self.use_gpu:
                net_params = torch.load(model_file, map_location=torch.device('cuda'))
            else:
                net_params = torch.load(model_file, map_location=torch.device('cpu'))
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_graph_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_graph_batch = state_graph_batch
            log_act_probs, value = self.policy_value_net(state_graph_batch.x, state_graph_batch.edge_index, state_graph_batch.batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            raise Exception("NO GPU?!")

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        if self.use_gpu:
            state_graph_batch = board.states_graph
            log_act_probs, value = self.policy_value_net(state_graph_batch.x, state_graph_batch.edge_index, state_graph_batch.batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            raise Exception("NO GPU?!")
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = state_batch
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            raise Exception("NO GPU?!")

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch.x, state_batch.edge_index, state_batch.batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        log_act_probs = log_act_probs.view(16,82)
        log_act_probs = log_act_probs[:,:-1]
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calculate policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        # return loss.data[0], entropy.data[0]
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
