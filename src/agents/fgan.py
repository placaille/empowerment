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

class fGANDiscreteStaticAgent(object):
    def __init__(self, **kwargs):
        assert type(kwargs.get('actions')) is dict
        self.device = kwargs.get('device')
        self.divergence_name = kwargs.get('diverg_name')
        self.max_batch_size = kwargs.get('batch_size')
        self.num_samples_grad = kwargs.get('num_samples_grad')
        self.actions = kwargs.get('actions')

        observation_size = kwargs.get('observation_size')
        hidden_size = kwargs.get('hidden_size')

        # optim config
        score_optim_name = kwargs.get('score_optim_name')
        score_lr = kwargs.get('score_lr')
        score_momentum = kwargs.get('score_momentum')
        score_weight_decay = kwargs.get('score_weight_decay')
        source_distr_optim_name = kwargs.get('source_distr_optim_name')
        source_distr_lr = kwargs.get('source_distr_lr')
        source_distr_momentum = kwargs.get('source_distr_momentum')
        source_distr_weight_decay = kwargs.get('source_distr_weight_decay')

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
        self.model_score = models.MLPShallow(2*observation_size+len(self.actions_seqs), hidden_size, 1)
        self.model_score.to(self.device)

        # optim score
        if score_optim_name == 'adam':
            self.optim_score = optim.Adam(self.model_score.parameters(),
                lr=score_lr, weight_decay=score_weight_decay)
        elif score_optim_name == 'sgd':
            self.optim_score = optim.SGD(self.model_score.parameters(),
                lr=score_lr, momentum=score_momentum, weight_decay=score_weight_decay)
        elif score_optim_name == 'rmsprop':
            self.optim_score = optim.RMSprop(self.model_score.parameters(),
                lr=score_lr, momentum=score_momentum, weight_decay=score_weight_decay)

        self.obj = self.fgan.discr_obj  # returns averything to compute empowerment

        # source distribution/generator/policy
        self.model_source_distr = models.LinearModel(observation_size, hidden_size, len(self.actions_seqs))
        self.model_source_distr.to(self.device)

        # optim source distr
        if source_distr_optim_name == 'adam':
            self.optim_source_distr = optim.Adam(self.model_source_distr.parameters(),
            lr=source_distr_lr, weight_decay=source_distr_weight_decay)
        elif source_distr_optim_name == 'sgd':
            self.optim_source_distr = optim.SGD(self.model_source_distr.parameters(),
            lr=source_distr_lr, momentum=source_distr_momentum, weight_decay=source_distr_weight_decay)
        elif source_distr_optim_name == 'rmsprop':
            self.optim_source_distr = optim.RMSprop(self.model_source_distr.parameters(),
            lr=source_distr_lr, momentum=source_distr_momentum, weight_decay=source_distr_weight_decay)

        self.memory = utils.Memory(mem_size, *mem_fields)

    def generate_rollouts(self, env, init_obs_all, actions_seqs, seq_onehots, add_to_memory=True):

        end_obs_all = []
        for (init_obs, action_seq, onehot) in zip(init_obs_all, actions_seqs, seq_onehots):
            env.reset(state=init_obs.argmax(-1))
            for action in action_seq:
                obs = env.step(action)

            end_obs_all.append(obs)
            if add_to_memory:
                self.memory.add_data(
                    obs_start=init_obs,
                    obs_end=obs,
                    act_seq=action_seq,
                    seq_onehot=onehot,
                )
        return np.array(end_obs_all)

    def train_step(self, env, init_state):

        obs_start = torch.FloatTensor(init_state).to(self.device)  # (batch, dims)
        seq_log_probs = F.log_softmax(self.model_source_distr(obs_start), dim=-1)  # (batch, dims)
        seq_distr = Categorical(logits=seq_log_probs)

        seqs_sampled = torch.LongTensor([seq_distr.sample() for _ in range(2)])  # a, a'
        seq_id = seqs_sampled[0]  # a

        # get s'_a, s'_a'
        action_seqs = {
            'onehot': self._convert_seq_id_to_onehot(seqs_sampled),
            'actions': [self.actions_lists[seq_id.item()] for seq_id in seqs_sampled],
        }
        obs_end_all = self.generate_rollouts(
            env=env,
            init_obs_all=[init_state for _ in range(2)],  # both seq start from the same state
            actions_seqs=action_seqs['actions'],
            seq_onehots=action_seqs['onehot'],
            add_to_memory=False
        )

        seq_onehot_1, seq_onehot_2 = action_seqs['onehot'][0], action_seqs['onehot'][1]
        obs_end_1, obs_end_2 = obs_end_all[0], obs_end_all[1]

        seq_onehot_1 = torch.FloatTensor(seq_onehot_1).to(self.device)
        seq_onehot_2 = torch.FloatTensor(seq_onehot_2).to(self.device)
        obs_end_1 = torch.FloatTensor(obs_end_1).to(self.device)
        obs_end_2 = torch.FloatTensor(obs_end_2).to(self.device)

        # combine into respective stacks
        stack_joint = torch.cat([obs_start, obs_end_1, seq_onehot_1], dim=-1)
        stack_marg_1 = torch.cat([obs_start, obs_end_2, seq_onehot_1], dim=-1)
        stack_marg_2 = torch.cat([obs_start, obs_end_1, seq_onehot_2], dim=-1)

        score_joint = self.fgan.pos_score(self.model_score(stack_joint))
        score_marg_1 = self.fgan.neg_score(self.model_score(stack_marg_1))
        score_marg_2 = self.fgan.neg_score(self.model_score(stack_marg_2))

        # loss for score (inverse sign for gradient descent)
        loss_score_joint = - score_joint
        loss_score_marginal = score_marg_2

        loss_score_total = loss_score_joint + loss_score_marginal

        # compute gradient source (inverse sign for gradient descent)
        grad_seq_log_probs = score_marg_1.detach() + score_marg_2.detach() - score_joint.detach()

        self.optim_score.zero_grad()
        self.optim_source_distr.zero_grad()
        loss_score_total.backward()
        seq_log_probs.gather(0, seq_id).backward(gradient=grad_seq_log_probs)
        self.optim_score.step()
        self.optim_source_distr.step()

        # prep out
        out = {'score':{}}
        out['score']['total'] = loss_score_total.item()
        out['score']['joint'] = loss_score_joint.item()
        out['score']['marginal'] = loss_score_marginal.item()

        return out

    def compute_empowerment_map(self, env, num_sample=1000):
        self.eval_mode()
        empowerment_states = utils.get_empowerment_values(
            agent=self,
            env=env,
            num_sample=num_sample,
        )
        self.train_mode()
        states_i, states_j = zip(*env.free_pos)

        # init map value to avg empowerment to simplify color mapping later
        empowerment_map = np.full(env.grid.shape, empowerment_states.mean(), dtype=np.float32)
        empowerment_map[states_i, states_j] = empowerment_states
        return empowerment_map, empowerment_states.mean()

    def compute_entropy_map(self, env):
        self.eval_mode()
        obs = torch.eye(len(env.free_states))
        with torch.no_grad():
            log_probs = F.log_softmax(self.model_source_distr(obs.to(self.device)), dim=-1)
            distr = Categorical(logits=log_probs)
            entropy = distr.entropy().cpu().numpy()

        self.train_mode()
        states_i, states_j = zip(*env.free_pos)

        # init map value to avg entropy to simplify color mapping later
        entropy_map = np.full(env.grid.shape, entropy.mean(), dtype=np.float32)
        entropy_map[states_i, states_j] = entropy
        return entropy_map, entropy.mean()

    def sample_source_distr(self, obs):
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            obs = torch.FloatTensor(obs).to(self.device)
        seq_logits = F.log_softmax(self.model_source_distr(obs), dim=-1)
        distr = Categorical(logits=seq_logits)
        action_ids = distr.sample()
        out = {
            'onehot': self._convert_seq_id_to_onehot(action_ids),
            'actions': [self.actions_lists[act.item()] for act in action_ids],
            }
        return out

    def _convert_act_list_to_str(self, actions):
        return ''.join([str(x) for x in actions])

    def _convert_seq_id_to_onehot(self, seq_id):
        self.seq_onehot = torch.eye(self.num_actions_seqs).to(self.device)
        return self.seq_onehot[seq_id]

    def eval_mode(self):
        self.model_score.eval()
        self.model_source_distr.eval()

    def train_mode(self):
        self.model_score.train()
        self.model_source_distr.train()

    def save_models(self, tag, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.model_score.save(os.path.join(out_dir, tag + 'score.pth'))
        self.model_source_distr.save(os.path.join(out_dir, tag + 'source_distr.pth'))
