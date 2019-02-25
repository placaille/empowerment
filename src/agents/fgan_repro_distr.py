import torch
import os

import numpy as np
import seaborn as sns

from torch import nn, optim
from itertools import product
from torch.distributions import Categorical

import torch.nn.functional as F

# custom code
import models
import utils

class fGANReproDiscreteStaticAgent(object):
    def __init__(self, actions, observation_size, hidden_size, emp_num_steps,
                 alpha, divergence_name, mem_size, mem_fields, max_batch_size,
                 path_source_distr, device='cpu'):
        assert type(actions) is dict
        self.device = device
        self.alpha = alpha
        self.divergence_name = divergence_name
        self.max_batch_size = max_batch_size

        self.actions = actions
        actions_id = [str(x) for x in self.actions.values()]
        self.actions_keys = [''.join(act_seq) for act_seq in product(actions_id, repeat=emp_num_steps)]

        self.actions_seqs = {}
        for actions_key in self.actions_keys:
            self.actions_seqs[actions_key] = self.actions_keys.index(actions_key)
        self.num_actions_seqs = len(self.actions_seqs)

        self.fgan = utils.fGAN(self.divergence_name)
        # model used to compute score (or marginals/joint) (s`+a, conditioned on s)
        self.model_score = models.MLP(2*observation_size+len(self.actions_seqs), hidden_size, 1)
        self.model_score.to(self.device)

        self.obj_score = self.fgan.discr_obj
        self.optim_score = optim.Adam(self.model_score.parameters())

        # model used to compute the source distribution of actions
        self.model_source_distr = models.MLP(observation_size, hidden_size, len(self.actions_seqs))
        self.model_source_distr.load(path_source_distr)
        self.model_source_distr.to(self.device)

        self.memory = utils.Memory(mem_size, *mem_fields)
        self.seq_onehot = None
        self.empowerment_states = np.zeros(observation_size) + 1e-5

    def _convert_act_list_to_str(self, actions):
        return ''.join([str(x) for x in actions])

    def _convert_seq_id_to_onehot(self, seq_id):
        self.seq_onehot = torch.zeros(seq_id.shape[0], self.num_actions_seqs)
        self.seq_onehot.scatter_(1, seq_id, 1.0)
        return self.seq_onehot

    def score_train_step(self):
        batch = self.memory.sample_data(self.max_batch_size)

        obs_start = torch.FloatTensor(batch.obs_start)
        obs_end = torch.FloatTensor(batch.obs_end)

        seq_id = [self.actions_seqs[self._convert_act_list_to_str(seq)] for seq in batch.act_seq]
        seq_id = torch.LongTensor(seq_id)
        act_seq = self._convert_seq_id_to_onehot(seq_id.unsqueeze(1))
        act_seq_shfld = act_seq[torch.randperm(act_seq.shape[0])]

        stack_joint = torch.cat([obs_start, obs_end, act_seq], dim=1)
        stack_marginal = torch.cat([obs_start, obs_end, act_seq_shfld], dim=1)

        logits_joint = self.model_score(stack_joint.to(self.device))
        logits_marginal = self.model_score(stack_marginal.to(self.device))

        net_scores = self.obj_score(logits_joint, logits_marginal)
        emp_value = net_scores.data.squeeze()
        loss_score = -net_scores.mean()

        self.optim_score.zero_grad()
        loss_score.backward()
        self.optim_score.step()

        self._update_emp_values(obs_start, emp_value)
        return loss_score.item()

    def _update_emp_values(self, obs_start, empowerment_value):
        states = obs_start.argmax(dim=1)
        for (state, emp) in zip(states, empowerment_value):
            self.empowerment_states[state] += self.alpha * (emp - self.empowerment_states[state])

    def compute_empowerment_map(self, env):
        states_i, states_j = zip(*env.free_pos)

        # init map value to avg empowerment to simplify color mapping later
        empowerment_map = np.full(env.grid.shape, self.empowerment_states.mean(), dtype=np.float32)
        empowerment_map[states_i, states_j] = self.empowerment_states
        return empowerment_map

    def sample_source_distr(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            seq_logits = self.model_source_distr(obs.to(self.device))
            seq_distr = Categorical(logits=seq_logits)
        return seq_distr.sample().item()

    def save_models(self, tag, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.model_score.save(os.path.join(out_dir, tag + 'score.pth'))
        self.model_source_distr.save(os.path.join(out_dir, tag + 'source_distr.pth'))
