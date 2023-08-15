"""
ATSC scenario: Hangzhou traffic network
@author: Tianshu Chu
"""

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from collections import deque
from envs.atsc_env import PhaseMap, PhaseSet, TrafficSimulator
from envs.real_net_data.build_file import gen_rou_file
import traci

sns.set_color_codes()

STATE_NAMES = ['wave','accident']
# node: (phase key, neighbor list)
NODES = {'cluster_5158769537_5158769538': ('2.0', ['cluster_5158769525_5158769526', '5064096736']),
         'cluster_5158769525_5158769526': ('2.1', ['cluster_5158769537_5158769538', 'cluster_2059544963_5167373335', 'cluster_9976813633_9976813660']),
         '5064096736': ('2.2', ['cluster_5158769537_5158769538', 'cluster_9976813632_9976813652']),
         'cluster_9976813632_9976813652': ('2.3', ['5064096736', 'cluster_9976813633_9976813660', '5158769572']),
         'cluster_9976813633_9976813660': ('2.4', ['cluster_9976813632_9976813652', 'cluster_5158769525_5158769526','cluster_9976813634_9976813659','5158769573']),
         '5158769572': ('2.5', ['cluster_9976813632_9976813652', '5158769573', 'cluster_5158769569_5158769570']),
         '5158769573': ('2.6', ['cluster_9976813633_9976813660','5158769572','cluster_5158769556_5158769557']),
         'cluster_5158769569_5158769570': ('2.4', ['5158769572', 'cluster_5158769556_5158769557']),
         'cluster_5158769556_5158769557': ('2.7', ['cluster_5158769569_5158769570', '5158769573', 'cluster_5064105232_5064105233']),
         'cluster_9976813634_9976813659': ('2.8', ['cluster_9976813633_9976813660', '5064105223','cluster_9976813636_9976813637_9976813661_9976813662','cluster_5064105232_5064105233']),
         'cluster_5064105232_5064105233': ('2.8', ['cluster_5158769556_5158769557', 'cluster_9976813634_9976813659', 'cluster_5064105239_5572714406']),
         'cluster_5064105239_5572714406': ('2.9', ['cluster_5064105232_5064105233','cluster_5064096677_5064096678_5064096689_5064096690']),
         '5064105223': ('2.9', ['cluster_9976813634_9976813659', 'cluster_9976813636_9976813637_9976813661_9976813662','5064105224']),
         'cluster_9976813636_9976813637_9976813661_9976813662': ('2.10', ['cluster_9976813634_9976813659','5064105223','cluster_9976813638_9976813656','cluster_5064096677_5064096678_5064096689_5064096690']),
         'cluster_5064096677_5064096678_5064096689_5064096690': ('4.0', ['cluster_5064105239_5572714406','cluster_9976813636_9976813637_9976813661_9976813662','cluster_5064096680_5064096687']),
         'cluster_2059544963_5167373335': ('2.11', ['cluster_5158769525_5158769526', 'cluster_5064105244_5167373383','5064105224']),
         '5064105224': ('2.12', ['cluster_2059544963_5167373335','5064105223','cluster_9976813638_9976813656']),
         'cluster_9976813638_9976813656': ('2.13', ['5064105224','cluster_9976813636_9976813637_9976813661_9976813662','cluster_5064096680_5064096687']),
         'cluster_5064096680_5064096687': ('2.8', ['cluster_5064096677_5064096678_5064096689_5064096690', 'cluster_9976813638_9976813656', '5064105247','cluster_2395874702_5064096683_5064096684_9363692528']),
         'cluster_5064105244_5167373383': ('2.14', ['cluster_2059544963_5167373335', 'cluster_2395875407_2890680726_5167373380_5167373381', '5064105248']),
         '5064105248': ('2.9', ['cluster_5064105244_5167373383', 'cluster_5064105249_5064105250','9318091992']),
         '9318091992': ('2.9', ['5064105248','cluster_9318091993_9318091994','5064105256']),
         '5064105256': ('2.9', ['9318091992', 'cluster_2569264934_2890680725', '5064105257']),
         '5064105257': ('2.9', ['5064105256', 'cluster_5064105258_5064105259', '5064105247']),
         '5064105247': ('2.15', ['cluster_5064096680_5064096687', '5064105257', 'cluster_2395874702_5064096683_5064096684_9363692528']),
         'cluster_2395875407_2890680726_5167373380_5167373381': ('2.16', ['cluster_5064105244_5167373383', 'cluster_5064105249_5064105250']),
         'cluster_5064105249_5064105250': ('2.11', ['cluster_2395875407_2890680726_5167373380_5167373381', '5064105248', 'cluster_9318091993_9318091994']),
         'cluster_9318091993_9318091994': ('2.11', ['cluster_5064105249_5064105250', '9318091992','cluster_2569264934_2890680725']),
         'cluster_2569264934_2890680725': ('2.11', ['cluster_9318091993_9318091994', '5064105256','cluster_5064105258_5064105259']),
         'cluster_5064105258_5064105259': ('2.11', ['cluster_2569264934_2890680725', '5064105257','cluster_2395874702_5064096683_5064096684_9363692528']),
         'cluster_2395874702_5064096683_5064096684_9363692528': ('2.11', ['cluster_5064105258_5064105259', '5064105247','cluster_5064096680_5064096687'])}

PHASES = {'2.0':['GGrrGGg', 'rrGGrrr'],
          '2.1':['GGggGGG','rrrrrrr'],
          '2.2':['GgGrr','rrGGG'],
          '2.3':['rrrrGggGG','GGGGrrrrr'],
          '2.4':['GGrrrrGgg','rrGGGGrrr'],
          '2.5':['rrrrGggGGg','GGGGrrrrrr'],
          '2.6':['GGGrr','rrrGG'],
          '2.7':['GggGGrrrr','rrrrrGGGG'],
          '2.8':['rrrrGGggrrrrGGgg','GGggrrrrGGggrrrr'],
          '2.9':['GggrrrGGg','rrrGGgGrr'],
          '2.10':['rrrrrGGggrrrrrGGgg','GGGggrrrrGGGggrrrr'],
          '4.0':['rrrrrrrGGgGrr','GGgGrrrrrrrrr','rrrGGGGrrrrrr','GrrrrrrrrrGGg'],
          '2.11':['rrrrGGGggrrrrGGGgg','GGggrrrrrGGggrrrrr'],
          '2.12':['rrrGGgGgg','GGgGrrrrr'],
          '2.13':['GGgGggrrr','GrrrrrGGg'],
          '2.14':['GGggrrrGGGg','rrrrGGgGrrr'],
          '2.15':['GgGr','rrGG'],
          '2.16':['rrrrrGGGggrrrrrGGGgg','GGGggrrrrrGGGggrrrrr']}



class RealNetPhase(PhaseMap):
    def __init__(self):
        self.phases = {}
        for key, val in PHASES.items():
            self.phases[key] = PhaseSet(val)


class RealNetController:
    def __init__(self, node_names, nodes):
        self.name = 'greedy'
        self.node_names = node_names
        self.nodes = nodes

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # get the action space
        phases = PHASES[NODES[node_name][0]]
        flows = []
        node = self.nodes[node_name]
        # get the green waves
        for phase in phases:
            wave = 0
            visited_ilds = set()
            for i, signal in enumerate(phase):
                if signal == 'G':
                    # find controlled lane
                    lane = node.lanes_in[i]
                    # ild = 'ild:' + lane
                    ild = lane
                    # if it has not been counted, add the wave
                    if ild not in visited_ilds:
                        j = node.ilds_in.index(ild)
                        wave += ob[j]
                        visited_ilds.add(ild)
            flows.append(wave)
        return np.argmax(np.array(flows))


class RealNetEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.flow_rate = config.getint('flow_rate')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _bfs(self, i):
        d = 0
        self.distance_mask[i, i] = d
        visited = [False]*self.n_node
        que = deque([i])
        visited[i] = True
        while que:
            d += 1
            for _ in range(len(que)):
                node_name = self.node_names[que.popleft()]
                for nnode in self.neighbor_map[node_name]:
                    ni = self.node_names.index(nnode)
                    if not visited[ni]:
                        self.distance_mask[i, ni] = d
                        visited[ni] = True
                        que.append(ni)
        return d

    def _get_node_phase_id(self, node_name):
        return self.phase_node_map[node_name]

    def _init_neighbor_map(self):
        self.neighbor_map = dict([(key, val[1]) for key, val in NODES.items()])
        self.neighbor_mask = np.zeros((self.n_node, self.n_node)).astype(int)
        for i, node_name in enumerate(self.node_names):
            for nnode in self.neighbor_map[node_name]:
                ni = self.node_names.index(nnode)
                self.neighbor_mask[i, ni] = 1
        logging.info('neighbor mask:\n %r' % self.neighbor_mask)

    def _init_distance_map(self):
        self.distance_mask = -np.ones((self.n_node, self.n_node)).astype(int)
        self.max_distance = 0
        for i in range(self.n_node):
            self.max_distance = max(self.max_distance, self._bfs(i))
        logging.info('distance mask:\n %r' % self.distance_mask)

    def _init_map(self):
        self.node_names = sorted(list(NODES.keys()))
        self.n_node = len(self.node_names)
        self._init_neighbor_map()
        self._init_distance_map()
        self.phase_map = RealNetPhase()
        self.phase_node_map = dict([(key, val[0]) for key, val in NODES.items()])
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        # comment out to call build_file.py
        return gen_rou_file(self.data_path,
                            thread=self.sim_thread)



    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_test_real.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = RealNetEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    env.train_mode = False
    time.sleep(1)
    # ob = env.reset(gui=True)
    controller = RealNetController(env.node_names, env.nodes)
    env.init_test_seeds(list(range(10000, 100001, 10000)))
    rewards = []
    for i in range(10):
        ob = env.reset(test_ind=i)
        global_rewards = []
        cur_step = 0
        while True:
            next_ob, reward, done, global_reward = env.step(controller.forward(ob))
            # for node_name, node_ob in zip(env.node_names, next_ob):
                # logging.info('%d, %s:%r\n' % (cur_step, node_name, node_ob))
            global_rewards.append(global_reward)
            rewards += list(reward)
            cur_step += 1
            if done:
                break
            ob = next_ob
        env.terminate()
        logging.info('step: %d, avg reward: %.2f' % (cur_step, np.mean(global_rewards)))
        time.sleep(1)
    env.plot_stat(np.array(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
