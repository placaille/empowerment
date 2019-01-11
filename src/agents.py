import torch

import numpy as np
import seaborn as sns

from torch import nn, optim
from itertools import product

import torch.nn.functional as F

# custom code
import models


class DiscreteStaticAgent(object):
    def __init__(self, actions, observation_size, hidden_size, emp_num_steps,
                 beta, max_batch_size=32, device='cpu'):
        assert type(actions) is dict
        self.device = device
        self.beta = beta
        self.max_batch_size = max_batch_size

        self.actions = actions
        actions_id = [str(x) for x in self.actions.values()]
        actions_keys = [''.join(act_seq) for act_seq in product(actions_id, repeat=emp_num_steps)]

        self.actions_seqs = {}
        for actions_key in actions_keys:
            self.actions_seqs[actions_key] = actions_keys.index(actions_key)

        # model used to compute likelihood of action sequences
        self.decoder = models.MLP(2*observation_size, hidden_size, len(self.actions_seqs))
        self.decoder.to(self.device)

        self.obj_decoder = nn.CrossEntropyLoss()
        self.optim_decoder = optim.Adam(self.decoder.parameters())

        # model used to compute the source distribution of actions
        self.model_source_distr = models.MLP(observation_size, hidden_size, len(self.actions_seqs))
        self.model_source_distr.to(self.device)

        self.model_phi = models.MLP(observation_size, hidden_size, 1)
        self.model_phi.to(self.device)

        self.obj_source = nn.MSELoss(reduction='mean')
        self.optim_source = optim.Adam(
            list(self.model_source_distr.parameters()) + \
            list(self.model_phi.parameters())
        )

    def _convert_act_list_to_str(self, actions):
        return ''.join([str(x) for x in actions])

    def source_train_step(self, obs_start, obs_end, actions):
        obs_start = torch.FloatTensor(obs_start)
        obs_stack = torch.cat([obs_start, torch.FloatTensor(obs_end)])
        seq = self._convert_act_list_to_str(actions)

        logits_seq = self.decoder(obs_stack.unsqueeze(0).to(self.device))
        log_softmax_seq = F.log_softmax(logits_seq, dim=1).detach()
        target_energy = self.beta * log_softmax_seq[:, self.actions_seqs[seq]]

        logits_source_distr = self.model_source_distr(obs_start.unsqueeze(0).to(self.device))
        log_softmax_source_distr = F.log_softmax(logits_source_distr, dim=1)
        log_likelihood_source = log_softmax_source_distr[:, self.actions_seqs[seq]]

        phi = self.model_phi(obs_start.to(self.device))
        preds_energy = log_likelihood_source + phi

        loss_source = self.obj_source(preds_energy, target_energy)

        self.optim_source.zero_grad()
        loss_source.backward()
        self.optim_source.step()
        return loss_source.item()

    def decoder_train_step(self, obs_start, obs_end, actions):
        obs_stack = torch.cat([torch.FloatTensor(obs_start), torch.FloatTensor(obs_end)])
        seq = self._convert_act_list_to_str(actions)

        labels_seq = torch.LongTensor([self.actions_seqs[seq]]).to(self.device)
        logits_seq = self.decoder(obs_stack.unsqueeze(0).to(self.device))

        loss_decoder = self.obj_decoder(logits_seq, labels_seq)

        self.optim_decoder.zero_grad()
        loss_decoder.backward()
        self.optim_decoder.step()
        return loss_decoder.item()

    def compute_empowerment(self, state):
        with torch.no_grad():
            phi = self.model_phi(torch.FloatTensor(state).to(self.device))
        return 1 / self.beta * phi

    def compute_empowerment_map(self, env):
        all_states = np.eye(env.observation_space.n)

        empowerment = []
        empowerment_map = np.zeros(env.grid.shape).astype(np.float32)
        for start_id in range(0, env.observation_space.n, self.max_batch_size):
            states = torch.FloatTensor(all_states[start_id:start_id+self.max_batch_size])
            empowerment.append(self.compute_empowerment(states.to(self.device)))
        empowerment = np.concatenate(empowerment).squeeze()

        states_i, states_j = zip(*env.free_pos)
        empowerment_map[states_i, states_j] = empowerment
        return empowerment_map
