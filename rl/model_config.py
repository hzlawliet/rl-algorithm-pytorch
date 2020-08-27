#!/usr/bin/env python
# coding=utf8


def one_hot_action(x):
    if x == 'a':
        return 0
    elif x == 'b':
        return 1
    elif x == 'c':
        return 2
    elif x == 'd':
        return 3
    else:
        return 4


def inverse_action(x):
    if x == 0:
        return 'a'
    elif x == 1:
        return 'b'
    elif x == 2:
        return 'c'
    elif x == 3:
        return 'd'
    else:
        return 'e'


continuous_fea_name = [
]

cnt_fea_name = [
]

high_fea_name = [
]

binary_fea_name = [
]

cat_fea_name = [
]


tongdun_fea_name = [
]

