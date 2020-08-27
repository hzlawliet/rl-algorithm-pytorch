#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/12/19 4:31 下午

@author: wanghengzhi
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder

from model_config import *
from train import train_ppo, train_dqn


def main(parameters):
    print(parameters)
    is_double = False
    is_dueling = False
    suffix = 'dqn'
    if parameters.model == 'dqn':
        if parameters.dueling == 'True':
            suffix = 'dueling_' + suffix
            is_dueling = True
        if parameters.double == 'True':
            suffix = 'double_' + suffix
            is_double = True
    else:
        suffix = 'ppo'

    # HyperParameters
    batch_size = 32
    lr = 0.002  # learning rate
    betas = (0.9, 0.999)
    gamma = 0.9  # reward discount
    target_iter = 1000  # target update frequency
    memory_capacity = 1000
    train_loss = 0
    epochs = 20
    # model_path = '/nfs/private/distribute-strategy/mdp/double_dueling/'
    model_path = 'mdp/' + suffix + '/'
    print(model_path)

    # load data
    raw_data = pd.read_csv('mdp/mdp_processed_data.csv')
    raw_data[cat_fea_name] = raw_data[cat_fea_name].fillna(raw_data[cat_fea_name].max() + 1)
    df = raw_data[raw_data.create_time < '2019-11-28:00:00:00']
    eval_data = raw_data[raw_data.create_time >= '2019-11-28:00:00:00']

    onehot = joblib.load('mdp/one_hot_online.model')
    onehot.handle_unknown = 'ignore'
    # onehot = OneHotEncoder().fit(df[cat_fea_name])
    # joblib.dump(onehot, '/nfs/private/distribute-strategy/mdp/one_hot_online.model')
    state_dim = len(high_fea_name) + len(continuous_fea_name) + len(cnt_fea_name) + len(binary_fea_name) + len(
        onehot.get_feature_names()) + 4
    print('state dim', state_dim)
    action_dim = 5

    df['action'] = df.funds_channel_id.apply(one_hot_action)

    df['reward'] = df.reward.apply(lambda x: -1 if x == -1 else x / 1000000)

    df['done'] = df.next_funds_channel_id.apply(lambda x: True if x != '-1' else False)

    df[high_fea_name + continuous_fea_name + cnt_fea_name + binary_fea_name] = df[
        high_fea_name + continuous_fea_name + cnt_fea_name + binary_fea_name].astype('float')

    # df[tongdun_fea_name] = 0
    df['mobile_city_id'] = 0

    df = df[df.duplicated(['uid', 'action'], keep='first') == False].sort_values(by=['uid', 'create_time'])

    df_dense = np.round(df[high_fea_name + continuous_fea_name + cnt_fea_name + binary_fea_name].values, 6)

    df_wide = onehot.transform(df[cat_fea_name]).A

    df_fail = df[['fail_a', 'fail_b', 'fail_c', 'fail_d']].values

    # train model
    if parameters.model == 'ppo':
        model = train_ppo(df, df_dense, df_wide, df_fail, state_dim, action_dim, lr, betas, gamma, epochs, model_path)
    else:
        model = train_dqn(df, df_dense, df_wide, df_fail, state_dim, action_dim, memory_capacity, lr, betas, gamma,
                          target_iter, epochs, model_path, is_double=is_double, is_dueling=is_dueling)


    # eval model
    continous = eval_data[high_fea_name + continuous_fea_name + cnt_fea_name + binary_fea_name].values
    catgory = onehot.transform(eval_data[cat_fea_name].values).A
    state_data = eval_data[['fail_a', 'fail_b', 'fail_c', 'fail_d']].values
    data = torch.FloatTensor(np.concatenate((continous, catgory, state_data), axis=1))

    model_result = model(data).data.numpy()

    print(np.round(model_result[:100], 2))

    eval_data['model_result'] = model_result.argmax(axis=1)
    eval_data['model_result'] = eval_data.model_result.apply(lambda x: inverse_action(x))

    print('========== total result ===========')
    print(eval_data.model_result.value_counts())

    c = eval_data.uid.value_counts()
    multi = c[c >= 3].index.values
    print('========== apply larger than 3times and not pass users ===========')
    print(eval_data[eval_data.uid.isin(multi) & (eval_data.funds_channel_id == 'e') & (
            eval_data.reward <= 0)].model_result.value_counts())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='use dqn or ppo', default='dqn')
    parser.add_argument('-d', '--double', help='use double or not', default='False')
    parser.add_argument('-dd', '--dueling', help='use dueling or not', default='False')
    args = parser.parse_args()
    main(args)

