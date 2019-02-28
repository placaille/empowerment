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
    def __init__(self, actions, observation_size, hidden_size, emp_num_steps,
                 alpha, divergence_name, mem_size, mem_fields, max_batch_size,
                 temperature, device='cpu'):
        assert type(actions) is dict
        self.device = device
        self.alpha = alpha
        self.temperature = temperature
        self.divergence_name = divergence_name
        self.max_batch_size = max_batch_size

        self.actions = actions
        actions_id = [str(x) for x in self.actions.values()]
        self.actions_keys = [''.join(act_seq) for act_seq in product(actions_id, repeat=emp_num_steps)]

        self.actions_seqs = {}
        self.actions_lists = {}
        for actions_key in self.actions_keys:
            self.actions_seqs[actions_key] = self.actions_keys.index(actions_key)
            self.actions_lists[self.actions_keys.index(actions_key)] = [int(act) for act in actions_key]
        self.num_actions_seqs = len(self.actions_seqs)

        self.fgan = utils.fGAN(self.divergence_name)
        # model used to compute score (or marginals/joint) (s`+a, conditioned on s)
        self.model_score = models.MLPBatchNorm(2*observation_size+len(self.actions_seqs), hidden_size, 1)
        self.model_score.to(self.device)

        self.model_generator = models.MLP(observation_size, hidden_size, len(self.actions_seqs))
        self.model_generator.to(self.device)

        self.obj = self.fgan.discr_obj

        self.optim = optim.SGD(list(self.model_score.parameters()) +
                               list(self.model_generator.parameters()),
                               lr=0.001,
                               momentum=0.9)

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
            log_probs = F.log_softmax(self.model_generator(obs.to(self.device)), dim=-1)
            distr = RelaxedOneHotCategorical(self.temperature, logits=log_probs)
            entropy = -(distr.logits * distr.probs).sum(-1).cpu().numpy()

        states_i, states_j = zip(*env.free_pos)

        # init map value to avg entropy to simplify color mapping later
        entropy_map = np.full(env.grid.shape, entropy.mean(), dtype=np.float32)
        entropy_map[states_i, states_j] = entropy
        return entropy_map

    def sample_action_sequence(self, obs):
        obs = torch.FloatTensor(obs)
        seq_logits = F.log_softmax(self.model_generator(obs.to(self.device)), dim=-1)
        seq_distr = RelaxedOneHotCategorical(self.temperature, logits=seq_logits)
        sample = seq_distr.rsample().cpu()
        out = {
            'soft_onehot': sample,
            'actions': self.actions_lists[sample.argmax().item()],
            }
        return out

    def save_models(self, tag, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.model_score.save(os.path.join(out_dir, tag + 'score.pth'))
        self.model_generator.save(os.path.join(out_dir, tag + 'generator.pth'))
