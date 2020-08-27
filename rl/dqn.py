#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/12/19 4:09 下午

@author: wanghengzhi
"""
import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn1 = nn.BatchNorm1d(128, affine=True)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32, affine=True)
        self.tanh2 = nn.Tanh()
        self.dropout = nn.Dropout()
        self.out = nn.Linear(32, action_dim)

    def forward(self, x):
        x = self.tanh1(self.bn1(self.fc1(x)))
        x = self.tanh2(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.out(x)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn1 = nn.BatchNorm1d(128, affine=True)

        self.adv_fc2 = nn.Linear(128, 32)
        self.val_fc2 = nn.Linear(128, 32)

        self.adv_out = nn.Linear(32, action_dim)
        self.val_out = nn.Linear(32, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.bn1(self.fc1(x)))
        adv = self.tanh(self.adv_fc2(x))
        adv = self.tanh(self.adv_out(adv))
        val = self.tanh(self.val_fc2(x))
        val = self.tanh(self.val_out(val)).expand(x.size(0), self.action_dim)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_dim)
        return x


class DQNAgent(object):
    def __init__(self, state_dim, action_dim, memory_capacity, lr, betas, gamma, target_iter,
                 is_double=False, is_dueling=False):
        self.learn_step_counter = 0  # for target updating
        self.state_dim = state_dim
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.target_iter = target_iter
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((memory_capacity, state_dim * 2 + 3))  # initialize memory
        self.loss_func = nn.MSELoss()
        self.is_double = is_double
        self.is_dueling = is_dueling
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.is_dueling:
            self.eval_net = DuelingDQN(state_dim, action_dim).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        else:
            self.eval_net = DQN(state_dim, action_dim).to(self.device)
            self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr, betas=betas)

    def store_transition(self, s, a, r, not_done, s_):
        transition = np.hstack((s, [a, r, not_done], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, batch_size=32):
        # target parameter update
        if self.learn_step_counter % self.target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(np.round(b_memory[:, :self.state_dim], 6)).to(self.device)
        # b_f = torch.FloatTensor(b_memory[:, N_STATES:N_STATES + 4]).to(self.device)
        b_a = torch.LongTensor(b_memory[:, self.state_dim:self.state_dim + 1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, self.state_dim + 1:self.state_dim + 2]).to(self.device)
        b_mask_not_done = torch.FloatTensor(b_memory[:, self.state_dim + 2:self.state_dim + 3]).to(self.device)
        b_s_ = torch.FloatTensor(np.round(b_memory[:, -self.state_dim:], 6)).to(self.device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        if self.is_double:
            # Double DQN
            q_next = self.eval_net(b_s_).detach()
            _, a_prime = q_next.max(1)
            q_target_next_values = self.target_net(b_s_).detach()
            q_target_s_a_prime = q_target_next_values.gather(1, a_prime.unsqueeze(1))
        else:
            # origin DQN
            q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
            q_target_s_a_prime = q_next.max(1)[0].view(batch_size, 1)
        q_target = b_r + b_mask_not_done * self.gamma * q_target_s_a_prime  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.item()

