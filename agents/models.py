"""
IA2C and MA2C algorithms
@author: Tianshu Chu
"""

import os
from agents.utils import OnPolicyBuffer, MultiAgentOnPolicyBuffer, Scheduler
from agents.policies import (LstmPolicy, FPPolicy, ConsensusPolicy, NCMultiAgentPolicy,
                             IC3MultiAgentPolicy, DIALMultiAgentPolicy)
import logging
import numpy as np
import tensorflow as tf


class IA2C:
    """
    The basic IA2C implementation with decentralized actor and centralized critic,
    limited to neighborhood area only.
    """
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0):
        self.name = 'ia2c'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, model_config)

    def add_transition(self, ob, naction, action, reward, value, done):
        # if self.reward_norm > 0:
        #     reward = reward / self.reward_norm
        # if self.reward_clip > 0:
        #     reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer[i].add_transition(ob[i], naction[i], action[i], reward, value[i], done)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        for i in range(self.n_agent):
            obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(Rends[i], dt)
            if i == 0:
                self.policy[i].backward(self.sess, obs, nas, acts, dones, Rs, Advs, cur_lr,
                                        summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy[i].backward(self.sess, obs, nas, acts, dones, Rs, Advs, cur_lr)

    def forward(self, obs, done, nactions=None, out_type='p'):
        out = []
        if nactions is None:
            nactions = [None] * self.n_agent
        for i in range(self.n_agent): 
            cur_out = self.policy[i].forward(self.sess, obs[i], done, nactions[i], out_type)
            out.append(cur_out)
        return out

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False

    def reset(self):
        for i in range(self.n_agent):
            self.policy[i]._reset()

    def save(self, model_dir, global_step):
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def _init_algo(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                   total_step, seed, model_config):
        # init params
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.identical_agent = False
        if (max(self.n_a_ls) == min(self.n_a_ls)):
            # note for identical IA2C, n_s_ls may have varient dims
            self.identical_agent = True
            self.n_s = n_s_ls[0]
            self.n_a = n_a_ls[0]
        else:
            self.n_s = max(self.n_s_ls)
            self.n_a = max(self.n_a_ls)
        self.neighbor_mask = neighbor_mask
        self.n_agent = len(self.neighbor_mask)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_step = model_config.getint('batch_size')
        self.n_fc = model_config.getint('num_fc')
        self.n_lstm = model_config.getint('num_lstm')
        # init tf
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(seed)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self.sess = tf.compat.v1.Session(config=config)
        self.policy = self._init_policy()
        self.saver = tf.compat.v1.train.Saver(max_to_keep=5)
        # init exp buffer and lr scheduler for training
        if total_step:
            self.total_step = total_step
            self._init_train(model_config, distance_mask, coop_gamma)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = int(np.sum(self.neighbor_mask[i]))
            if self.identical_agent:
                policy.append(LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                         n_fc=self.n_fc, n_lstm=self.n_lstm, name='%d' % i))
            else:
                na_dim_ls = []
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j])
                policy.append(LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                         n_fc=self.n_fc, n_lstm=self.n_lstm, name='%d' % i,
                                         na_dim_ls=na_dim_ls, identical=False))
        return policy

    def _init_scheduler(self, model_config):
        # init lr scheduler
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)

    def _init_train(self, model_config, distance_mask, coop_gamma):
        # init lr scheduler
        self._init_scheduler(model_config)
        v_coef = model_config.getfloat('value_coef')
        e_coef = model_config.getfloat('entropy_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        self.trans_buffer = []
        for i in range(self.n_agent):
            # init loss
            self.policy[i].prepare_loss(v_coef, e_coef, max_grad_norm, alpha, epsilon)
            # init replay buffer
            self.trans_buffer.append(OnPolicyBuffer(gamma, coop_gamma, distance_mask[i]))


class IA2C_FP(IA2C):
    """
    In fingerprint IA2C, neighborhood policies (fingerprints) are also included.
    """
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0):
        self.name = 'ia2c_fp'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma, 
                        total_step, seed, model_config)

    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = np.sum(self.neighbor_mask[i])
            # neighborhood policies are included in local state
            if self.identical_agent:
                n_s1 = self.n_s_ls[i] + self.n_a*n_n
                policy.append(FPPolicy(n_s1, self.n_a, n_n, self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='%d' % i))
            else:
                na_dim_ls = []
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j])
                n_s1 = self.n_s_ls[i] + sum(na_dim_ls)
                policy.append(FPPolicy(n_s1, self.n_a_ls[i], n_n, self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='%d' % i,
                                       na_dim_ls=na_dim_ls, identical=False))
        return policy


class MA2C_NC(IA2C):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0):
        self.name = 'ma2c_nc'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, model_config)

    def add_transition(self, ob, p, action, reward, value, done):
        # if self.reward_norm > 0:
        #     reward = reward / self.reward_norm
        # if self.reward_clip > 0:
        #     reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        if self.identical_agent:
            self.trans_buffer.add_transition(np.array(ob), np.array(p), action,
                                             reward, value, done)
        else:
            pad_ob, pad_p = self._convert_hetero_states(ob, p)
            self.trans_buffer.add_transition(pad_ob, pad_p, action,
                                             reward, value, done)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        obs, ps, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(Rends, dt)
        self.policy.backward(self.sess, obs, ps, acts, dones, Rs, Advs, cur_lr,
                             summary_writer=summary_writer, global_step=global_step)

    def forward(self, obs, done, ps, actions=None, out_type='p'):
        if self.identical_agent:
            return self.policy.forward(self.sess, np.array(obs), done, np.array(ps),
                                       actions, out_type)
        else:
            pad_ob, pad_p = self._convert_hetero_states(obs, ps)
            return self.policy.forward(self.sess, pad_ob, done, pad_p,
                                       actions, out_type)

    def reset(self):
        self.policy._reset()

    def _convert_hetero_states(self, ob, p):
        pad_ob = np.zeros((self.n_agent, self.n_s))
        pad_p = np.zeros((self.n_agent, self.n_a))
        for i in range(self.n_agent):
            pad_ob[i, :len(ob[i])] = ob[i]
            pad_p[i, :len(p[i])] = p[i]
        return pad_ob, pad_p

    def _init_policy(self):
        if self.identical_agent:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                      n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)

    def _init_train(self, model_config, distance_mask, coop_gamma):
        # init lr scheduler
        self._init_scheduler(model_config)
        v_coef = model_config.getfloat('value_coef')
        e_coef = model_config.getfloat('entropy_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        gamma = model_config.getfloat('gamma')
        # init loss
        self.policy.prepare_loss(v_coef, e_coef, max_grad_norm, alpha, epsilon)
        # init replay buffer
        self.trans_buffer = MultiAgentOnPolicyBuffer(gamma, coop_gamma, distance_mask)


class IA2C_CU(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0):
        self.name = 'ma2c_cu'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                   n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)


class MA2C_IC3(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0):
        self.name = 'ma2c_ic3'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return IC3MultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                       self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return IC3MultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                       self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                       n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)


class MA2C_DIAL(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0):
        self.name = 'ma2c_dial'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                        n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)
