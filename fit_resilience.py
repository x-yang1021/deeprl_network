#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:29:48 2022

@author: EVE
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline



def smooth(csv_path,weight=0.85):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    return save
    # save.to_csv('smooth_'+csv_path)


# csv_path = "/Users/EVE/Desktop/nut/Research_in_IC/figs/grid_training_reward.csv"
csv_path = "H:\GitHub\deeprl_network\\test.csv"
# csv_path = "/Users/EVE/Desktop/nut/Research_in_IC/figs/net_train_reward_1022.csv"
# csv_path = '/home/PJLAB/yuyi/Documents/nut/Research_in_IC/figs/log_1022/net_reward.csv'
# grid_reward_smooth = smooth(csv_path,weight=0.85)


# 原数据
grid_reward = pd.read_csv(csv_path)
# grid_reward = grid_reward.head(300)  # 取前200个episode
# test = grid_reward




# 移动平均
# test = grid_reward.rolling(5).mean()

# exp smooth
# test = grid_reward_smooth

# 线性插值
x = np.array(grid_reward['Step'].values)
y = np.array(grid_reward['Value'].values)
# x_smooth = np.linspace(x.min(), x.max(), 100)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
# y_smooth = make_interp_spline(x, y, k=3, bc_type="clamped")(x_smooth)
# 计算performance
test = pd.DataFrame.from_dict(
                                        {
                                              "Step": [_ for _ in x],
                                              "Value": [-1/_ for _ in y], # 计算performance
                                          } 
                                        )




eps = 50  # episode绝对值
eps_n = 1  # 实际episode是画图值的多少倍

# 画头尾图
plt.figure()
ax = plt.subplot(111)
plt.title('{}'.format(csv_path[-25:-10]))
plt.plot(np.arange(0,eps*eps_n,eps_n),test.head(eps)['Value'].values,'--.',alpha=0.4,label='first {} episodes'.format(eps*eps_n))
plt.plot(np.arange(0,eps*eps_n,eps_n),test.tail(eps)['Value'].values,'-',alpha=0.7,label='last {} episodes'.format(eps*eps_n))
ax.set_xlabel('episode')
ax.set_ylabel('performance')
# ax.set_ylim((0.00005,0.0005))

plt.legend()
plt.show()
# plt.savefig('/Users/EVE/Desktop/nut/Research_in_IC/figs/net_resilience.svg', dpi=200)
# plt.savefig('/Users/EVE/Desktop/nut/Research_in_IC/figs/net_resilience.png', dpi=200)

head = np.var(grid_reward.head(10)['Value'].values)
tail = np.var(grid_reward.tail(10)['Value'].values)
print('head{:.2e}, tail={:.2e}, tail-head{:.2e}'.format(head, tail, tail-head))








