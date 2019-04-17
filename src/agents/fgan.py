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
        self.actions = kwargs.get('actions')

        observation_size = kwargs.get('observation_size')
        hidden_size = kwargs.get('hidden_size')

        # optim config
        optim_name = kwargs.get('optim_name')
        lr = kwargs.get('lr')
        momentum = kwargs.get('momentum')
        weight_decay = kwargs.get('weight_decay')

        mem_size = kwargs.get('memory_size')
        mem_fields = kwargs.get('mem_fields')
        self.memory = utils.Memory(mem_size, *mem_fields)

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

        self.seq_onehot = np.eye(self.num_actions_seqs)
        self.empowerment_values = torch.zeros(observation_size).to(self.device)

        # model used to compute score (or marginals/joint) (s`+a, conditioned on s)
        self.model_score = models.MLPShallow(2*observation_size + self.num_actions_seqs, hidden_size, 1)
        self.model_score.to(self.device)

        self.obj = self.fgan.discr_obj  # returns averything to compute empowerment

        # source distribution/generator/policy
        self.model_policy = models.LinearModel(observation_size, hidden_size, self.num_actions_seqs + 1)
        self.model_policy.to(self.device)

        # optim
        params = list(self.model_score.parameters()) + list(self.model_policy.parameters())
        if optim_name == 'adam':
            self.optim = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optim_name == 'sgd':
            self.optim = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optim_name == 'rmsprop':
            self.optim = optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def generate_on_policy_rollouts(self, env, init_states, num_rollouts=1, add_to_memory=True):
        """
        returns (log_probs, samples, onehots, obs_end)
        """
        assert isinstance(init_states, list) or isinstance(init_states, tuple)
        obs_start = torch.FloatTensor(init_states).to(self.device)  # (batch, dims)
        logits, _ = self._model_policy(obs_start)
        seq_log_probs = F.log_softmax(logits, dim=-1)  # (batch, dims)
        seq_distrs = Categorical(logits=seq_log_probs)

        # prep for acting
        b, n, s, a = obs_start.shape[0], num_rollouts, obs_start.shape[-1], self.num_actions_seqs
        state_ids = obs_start.argmax(-1).cpu().numpy()  # (b)

        seqs_sampled = np.empty((n, b), dtype=np.long)
        for i in range(n):
            seqs_sampled[i] = seq_distrs.sample().cpu().numpy()

        seq_onehot_all = self._convert_seq_id_to_onehot(seqs_sampled)  # (n, b, a)

        obs_end_all = np.empty((n, b, s))
        for i in range(n):
            for j in range(b):
                action_seq = self.actions_lists[seqs_sampled[i, j]]

                env.reset(state=state_ids[j])
                for action in action_seq:
                    obs = env.step(action)

                obs_end_all[i, j] = obs
                if add_to_memory:
                    self.memory.add_data(
                        obs_start=init_states[j],
                        obs_end=obs.copy(),
                        act_seq=action_seq,
                        seq_onehot=seq_onehot_all[i, j],
                    )

        return seq_distrs, seqs_sampled, seq_onehot_all, obs_end_all

    def train_step(self, env):

        # import pdb;pdb.set_trace()
        batch = self.memory.sample_data(self.max_batch_size)
        obs_start_s = torch.FloatTensor(batch.obs_start).to(self.device)
        b_size = obs_start_s.shape[0]

        # score network
        obs_end_s = torch.FloatTensor(batch.obs_end).to(self.device)
        seq_onehot_s = torch.FloatTensor(batch.seq_onehot).to(self.device)
        seq_onehot_s_shfld = seq_onehot_s[torch.randperm(b_size)]

        stack_joint_s = torch.cat([obs_start_s, obs_end_s, seq_onehot_s], dim=-1)
        stack_marg_s = torch.cat([obs_start_s, obs_end_s, seq_onehot_s_shfld], dim=-1)

        score_joint_s = self.fgan.pos_score(self.model_score(stack_joint_s))
        score_marg_s = self.fgan.neg_score(self.model_score(stack_marg_s))

        # loss for score (inverse sign for gradient descent)
        loss_score_joint = - score_joint_s.mean()
        loss_score_marginal = score_marg_s.mean()
        loss_score_total = loss_score_joint + loss_score_marginal

        _, predict_emp = self._model_policy(obs_start_s)
        target_emp = self.fgan.constant + score_joint_s - score_marg_s

        # loss empowerment value
        loss_emp = F.mse_loss(predict_emp, target_emp.detach(), reduction='none').mean()

        # policy network
        init_states = [env.reset() for _ in range(b_size)]
        out = self.generate_on_policy_rollouts(env, init_states, num_rollouts=2, add_to_memory=False)
        seq_distrs, seq_sampled, seq_onehots_all, obs_end_all = out

        seq_id_1, seq_id_2 = seq_sampled
        seq_onehot_1, seq_onehot_2 = seq_onehots_all
        obs_end_1, obs_end_2 = obs_end_all

        obs_start_p = torch.FloatTensor(init_states).to(self.device)
        seq_id_1 = torch.LongTensor(seq_id_1).to(self.device)
        seq_onehot_1 = torch.FloatTensor(seq_onehot_1).to(self.device)
        seq_onehot_2 = torch.FloatTensor(seq_onehot_2).to(self.device)
        obs_end_1 = torch.FloatTensor(obs_end_1).to(self.device)
        obs_end_2 = torch.FloatTensor(obs_end_2).to(self.device)

        # combine into respective stacks
        stack_joint_p = torch.cat([obs_start_p, obs_end_1, seq_onehot_1], dim=-1)
        stack_marg_1 = torch.cat([obs_start_p, obs_end_2, seq_onehot_1], dim=-1)
        stack_marg_2 = torch.cat([obs_start_p, obs_end_1, seq_onehot_2], dim=-1)

        score_joint_p = self.fgan.pos_score(self.model_score(stack_joint_p))
        score_marg_1 = self.fgan.neg_score(self.model_score(stack_marg_1))
        score_marg_2 = self.fgan.neg_score(self.model_score(stack_marg_2))

        # compute gradient source (inverse sign for gradient descent)
        pol_grad_signal = (score_marg_1 + score_marg_2 - score_joint_p).squeeze(1) / b_size
        log_probs = seq_distrs.log_prob(seq_id_1)

        # setup for single backward pass
        tensors = [log_probs] + [loss_score_total + loss_emp]
        grad_tensors = [pol_grad_signal] + [torch.ones(1).to(self.device)]

        self.optim.zero_grad()
        torch.autograd.backward(tensors, grad_tensors)
        self.optim.step()

        out = {
            'score':{
                'loss_total': loss_score_total.item(),
                'loss_joint': loss_score_joint.item(),
                'loss_marginal': loss_score_marginal.item(),
            },
            'policy': {
                'grad_signal_mean': pol_grad_signal.mean(),
                'loss_emp': loss_emp.item(),
            }
        }
        return out

    def _model_policy(self, obs):
        return torch.split(self.model_policy(obs), [self.num_actions_seqs, 1], dim=-1)

    @torch.no_grad()
    def compute_empowerment_map(self, env, num_sample=1000):
        self.eval_mode()
        obs_start = torch.eye(len(env.free_pos)).to(self.device)
        _, pred_emp = self._model_policy(obs_start)
        empowerment_states = pred_emp.squeeze(1)
        self.train_mode()
        states_i, states_j = zip(*env.free_pos)
        self.empowerment_values = empowerment_states.data

        # init map value to avg empowerment to simplify color mapping later
        empowerment_map = np.full(env.grid.shape, empowerment_states.mean().item(), dtype=np.float32)
        empowerment_map[states_i, states_j] = empowerment_states.cpu().numpy()
        return empowerment_map, empowerment_states.mean().item()

    @torch.no_grad()
    def compute_entropy_map(self, env):
        self.eval_mode()
        obs = torch.eye(len(env.free_states)).to(self.device)
        logits, _ = self._model_policy(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        distr = Categorical(logits=log_probs)
        entropy = distr.entropy().cpu().numpy()

        self.train_mode()
        states_i, states_j = zip(*env.free_pos)

        # init map value to avg entropy to simplify color mapping later
        entropy_map = np.full(env.grid.shape, entropy.mean(), dtype=np.float32)
        entropy_map[states_i, states_j] = entropy
        return entropy_map, entropy.mean()

    def sample_policy(self, obs):
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            obs = torch.FloatTensor(obs).to(self.device)
        logits, _ = self._model_policy(obs)
        seq_logits = F.log_softmax(logits, dim=-1)
        distr = Categorical(logits=seq_logits)
        action_ids = distr.sample()
        out = {
            'onehot': self._convert_seq_id_to_onehot(action_ids.cpu().numpy()),
            'actions': [self.actions_lists[act.item()] for act in action_ids],
            }
        return out

    def _convert_act_list_to_str(self, actions):
        return ''.join([str(x) for x in actions])

    def _convert_seq_id_to_onehot(self, seq_id):
        return self.seq_onehot[seq_id]

    def eval_mode(self):
        self.model_score.eval()
        self.model_policy.eval()

    def train_mode(self):
        self.model_score.train()
        self.model_policy.train()

    def save_models(self, tag, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.model_score.save(os.path.join(out_dir, tag + 'score.pth'))
        self.model_policy.save(os.path.join(out_dir, tag + 'policy.pth'))
