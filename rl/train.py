#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/12/19 11:22 上午

@author: wanghengzhi
"""
import gc
import time
import sys
from collections import deque

import torch
import numpy as np

from dqn import DQNAgent
from ppo import PPO, Memory


def train_dqn(df, df_dense, df_wide, df_fail, state_dim, action_dim, memory_capacity, lr, betas, gamma, target_inter,
              epochs, model_path, is_double=False, is_dueling=False):
    dqn = DQNAgent(state_dim, action_dim, memory_capacity, lr, betas, gamma, target_inter, is_double, is_dueling)
    log_file = open(model_path + 'log_file', 'w+')

    print("backend:", dqn.device, file=log_file)
    print("is double DQN:", dqn.is_double, file=log_file)
    print("is dueling DQN:", dqn.is_dueling, file=log_file)

    cnt = 0
    loss_queue = deque()
    break_flag = False

    for epoch in range(epochs):
        gc.collect()
        print("epoch start:", epoch, file=log_file)
        now = time.time()
        for index in range(df.shape[0] - 1):
            row = df.iloc[index]
            state_dense = df_dense[index]
            state_wide = df_wide[index]
            fail_state = df_fail[index]
            state = np.concatenate((state_dense, state_wide, fail_state))
            action = row['action']
            reward = row['reward']
            next_funds = row['next_funds_channel_id']
            if next_funds == '-1':
                next_state = np.zeros(state_dim)
                not_done = 0
            else:
                next_row = df.iloc[index + 1]
                if next_row.uid == row.uid and next_row.funds_channel_id == next_funds:
                    next_state_dense = df_dense[index + 1]
                    next_state_wide = df_wide[index + 1]
                    next_fail_state = df_fail[index + 1]
                    next_state = np.concatenate((next_state_dense, next_state_wide, next_fail_state))
                    not_done = 1
                    cnt += 1
                else:
                    continue

            dqn.store_transition(state, action, reward, not_done, next_state)

            if dqn.memory_counter > memory_capacity:
                loss = dqn.learn()
                loss_queue.append(loss)
                while len(loss_queue) > 100000:
                    loss_queue.popleft()
                if dqn.learn_step_counter % 10 == 0:
                    if epoch > 1:
                        if len(loss_queue) >= 100000 and np.mean(loss_queue) < 0.5:
                            print('dqn has already convergence', np.mean(loss_queue), file=log_file)
                            break_flag = True
                            break
                if dqn.learn_step_counter % 1000 == 0:
                    sys.stdout.flush()
                    print('==============' + str(dqn.learn_step_counter) + '-th step loss:', np.mean(loss_queue), file=log_file)
        print("time cost:", time.time() - now, file=log_file)
        if break_flag:
            break
        else:
            torch.save(dqn.eval_net.state_dict(),
                       model_path + 'dqn_model_20190901_20191204_add_round_add_fail_' + str(epoch) + '-th_epoch.pkl')

    print(cnt, cnt / df.shape[0], file=log_file)
    sys.stdout.flush()
    torch.save(dqn.eval_net.state_dict(), model_path + 'dqn_model_20190901_20191204_add_round_add_fail.pkl')
    log_file.close()
    return dqn.eval_net.cpu().eval()


def train_ppo(df, df_dense, df_wide, df_fail, state_dim, action_dim, lr, betas, gamma, epochs, model_path):
    memory = Memory()
    n_latent_var = [128, 32]
    K_epochs = 4
    eps_clip = 0.2
    update_timestep = 2000
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    timestep = 0

    loss_file = open(model_path + 'loss.txt', 'a')

    for epoch in range(epochs):
        print("epoch start:" + str(epoch) + '\n')
        moving_loss = 0
        cnt = 0
        for index in range(df.shape[0]):
            timestep += 1
            row = df.iloc[index]
            state_dense = df_dense[index]
            state_wide = df_wide[index]
            fail_state = df_fail[index]
            state = np.concatenate((state_dense, state_wide, fail_state))
            action = row['action']
            reward = row['reward']
            done = row['done']
            ppo.policy_old.act(state, action, reward, done, memory)

            if timestep % update_timestep == 0:
                loss = ppo.update(memory)
                memory.clear_memory()
                timestep = 0
                moving_loss += np.mean(loss)
                cnt += 1

        loss_file.write(str(epoch) + '-th round loss: ' + str(round(moving_loss / cnt, 4)) + '\n')
        loss_file.flush()
        torch.save(ppo.policy.action_layer.state_dict(),
                   model_path + 'ppo_20191009_20191021_action_layer' + str(
                       epoch) + '-th_epoch.pkl')
        torch.save(ppo.policy.value_layer.state_dict(),
                   model_path + 'ppo_20191009_20191021_action_layer' + str(
                       epoch) + '-th_epoch.pkl')

    gc.collect()
    loss_file.close()

    return ppo.policy.action_layer.cpu().eval()

