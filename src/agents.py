import torch

from torch import nn, optim
from itertools import product

import torch.nn.functional as F

# custom code
import models


class DiscreteStaticAgent(object):
    def __init__(self, actions, observation_size, hidden_size, emp_num_steps,
                 beta, device='cpu'):
        assert type(actions) is dict
        self.device = device
        self.beta = beta

        self.actions = actions
        actions_id = [str(x) for x in self.actions.values()]
        actions_keys = [''.join(act_seq) for act_seq in product(actions_id, repeat=emp_num_steps)]

        self.actions_seqs = {}
        for actions_key in actions_keys:
            self.actions_seqs[actions_key] = actions_keys.index(actions_key)

        # model used to compute likelihood of action sequences
        self.decoder = models.MLP(2*observation_size, hidden_size, len(self.actions_seqs))
        self.decoder.to(device)

        self.obj_decoder = nn.CrossEntropyLoss()
        self.optim_decoder = optim.Adam(self.decoder.parameters())

        # model used to compute the source distribution of actions
        self.energy_model = models.MLP(observation_size + len(self.actions_seqs), hidden_size, 1)
        self.energy_model.to(device)

        self.obj_energy = nn.MSELoss(reduction='mean')
        self.optim_energy = optim.Adam(self.energy_model.parameters())

    def _convert_act_list_to_str(self, actions):
        return ''.join([str(x) for x in actions])

    def _convert_seq_to_onehot(self, seq_idx):
        onehot = torch.zeros(len(self.actions_seqs))
        onehot[seq_idx] = 1.0
        return onehot

    def energy_train_step(self, obs_start, obs_end, actions):
        obs_stack = torch.cat([torch.FloatTensor(obs_start), torch.FloatTensor(obs_end)])
        seq = self._convert_act_list_to_str(actions)

        logits_seq = self.decoder(obs_stack.unsqueeze(0).to(self.device))
        log_softmax_seq = F.log_softmax(logits_seq, dim=1).detach()
        target_energy = self.beta * log_softmax_seq[:, self.actions_seqs[seq]]

        onehot_seq = self._convert_seq_to_onehot(self.actions_seqs[seq])
        seq_obs_stack = torch.cat([torch.FloatTensor(obs_start), torch.FloatTensor(onehot_seq)])
        logits_energy = self.energy_model(seq_obs_stack.to(self.device))

        loss_energy = self.obj_energy(logits_energy, target_energy)

        self.optim_energy.zero_grad()
        loss_energy.backward()
        self.optim_energy.step()
        return loss_energy.item()

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
