import torch
import os

import numpy as np
import seaborn as sns

from torch import nn, optim
from itertools import product
from torch.distributions import Categorical, RelaxedOneHotCategorical

import torch.nn.functional as F

# custom code
import models
import utils

class fGANGumbelDiscreteStaticAgent(object):
    def __init__(self, **kwargs):
        assert type(kwargs.get('actions')) is dict
        self.device = kwargs.get('device')
        self.alpha = kwargs.get('alpha')
        self.temperature_start = kwargs.get('temperature_start')
        self.temperature = self.temperature_start
        self.divergence_name = kwargs.get('divergence_name')
        self.max_batch_size = kwargs.get('max_batch_size')
        self.actions = kwargs.get('actions')

        observation_size = kwargs.get('observation_size')
        hidden_size = kwargs.get('hidden_size')
        path_score = kwargs.get('path_score')
        path_source_distr = kwargs.get('path_source_distr')
        train_score = kwargs.get('train_score')
        train_source_distr = kwargs.get('train_source_distr')

        mem_size = kwargs.get('mem_size')
        mem_fields = kwargs.get('mem_fields')

        actions_id = [str(x) for x in self.actions.values()]
        self.actions_keys = [''.join(act_seq) for act_seq in
                             product(actions_id, repeat=kwargs.get('emp_num_steps'))]

        self.actions_seqs = {}
        self.actions_lists = {}
        for actions_key in self.actions_keys:
            self.actions_seqs[actions_key] = self.actions_keys.index(actions_key)
            self.actions_lists[self.actions_keys.index(actions_key)] = [int(act) for act in actions_key]
        self.num_actions_seqs = len(self.actions_seqs)

        self.fgan = utils.fGAN(self.divergence_name)
        # model used to compute score (or marginals/joint) (s`+a, conditioned on s)
        self.model_score = models.MLP(2*observation_size+len(self.actions_seqs), hidden_size, 1)
        if path_score:
            self.model_score.load(path_score)
        self.model_score.to(self.device)

        self.model_source_distr = models.MLP(observation_size, hidden_size, len(self.actions_seqs))
        if path_source_distr:
            self.model_source_distr.load(path_source_distr)
        self.model_source_distr.to(self.device)

        self.obj = self.fgan.discr_obj
        params = []
        if train_score:
            params += list(self.model_score.parameters())
        if train_source_distr:
            params += list(self.model_source_distr.parameters())

        self.optim = optim.Adam(params)

        self.memory = utils.Memory(mem_size, *mem_fields)
        self.seq_onehot = None
        self.empowerment_states = np.zeros(observation_size) + 1e-5

    def _convert_act_list_to_str(self, actions):
        return ''.join([str(x) for x in actions])

    def _convert_seq_id_to_onehot(self, seq_id):
        self.seq_onehot = torch.zeros(seq_id.shape[0], self.num_actions_seqs).to(self.device)
        self.seq_onehot.scatter_(1, seq_id, 1.0)
        return self.seq_onehot

    def train_step(self):
        batch = self.memory.sample_data(self.max_batch_size)

        obs_start = torch.FloatTensor(batch.obs_start).to(self.device)
        obs_end = torch.FloatTensor(batch.obs_end).to(self.device)
        seq_soft_onehot = torch.stack(batch.seq_soft_onehot).to(self.device)
        seq_soft_onehot_shfl = seq_soft_onehot[torch.randperm(seq_soft_onehot.shape[0])]

        stack_joint = torch.cat([obs_start, obs_end, seq_soft_onehot], dim=1)
        stack_marginal = torch.cat([obs_start, obs_end, seq_soft_onehot_shfl], dim=1)

        logits_joint = self.model_score(stack_joint.to(self.device))
        logits_marginal = self.model_score(stack_marginal.to(self.device))

        constant, scores_joint, scores_marginal = self.obj(logits_joint, logits_marginal)
        scores_net = constant + scores_joint - scores_marginal
        emp_value = scores_net.data
        loss = - (constant.mean() + scores_joint.mean() - scores_marginal.mean())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self._update_emp_values(obs_start, emp_value)
        return loss.item()

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

    def compute_entropy_map(self, env):
        obs = torch.eye(len(env.free_states))
        with torch.no_grad():
            log_probs = F.log_softmax(self.model_source_distr(obs.to(self.device)), dim=-1)
            distr = RelaxedOneHotCategorical(self.temperature, logits=log_probs)
            entropy = -(distr.logits * distr.probs).sum(-1).cpu().numpy()

        states_i, states_j = zip(*env.free_pos)

        # init map value to avg entropy to simplify color mapping later
        entropy_map = np.full(env.grid.shape, entropy.mean(), dtype=np.float32)
        entropy_map[states_i, states_j] = entropy
        return entropy_map

    def sample_source_distr(self, obs):
        obs = torch.FloatTensor(obs)
        seq_logits = F.log_softmax(self.model_source_distr(obs.to(self.device)), dim=-1)
        seq_distr = RelaxedOneHotCategorical(self.temperature, logits=seq_logits)
        sample = seq_distr.rsample().cpu()
        out = {
            'soft_onehot': sample,
            'actions': self.actions_lists[sample.argmax().item()],
            }
        return out

    def anneal_temperature(self, iter):
        if iter % 1000 == 0:
            self.temperature = np.max((0.5, np.exp(-3e-5 * iter)))

    def save_models(self, tag, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.model_score.save(os.path.join(out_dir, tag + 'score.pth'))
        self.model_source_distr.save(os.path.join(out_dir, tag + 'source_distr.pth'))
