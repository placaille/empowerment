import torch

from torch import nn, optim
from itertools import product

# custom code
import models


class DiscreteStaticAgent(object):
    def __init__(self, actions, observation_size, hidden_size, emp_num_steps,
                 device='cpu'):
        assert type(actions) is dict

        actions_id = [str(x) for x in actions.values()]
        actions_keys = [''.join(act_seq) for act_seq in product(actions_id, repeat=emp_num_steps)]

        self.actions_seqs = {}
        for actions_key in actions_keys:
            self.actions_seqs[actions_key] = actions_keys.index(actions_key)

        # model used to compute likelihood of action sequences
        self.decoder = models.MLP(2*observation_size, hidden_size, len(self.actions_seqs))
        self.obj_decoder = nn.CrossEntropyLoss()
        self.optim_decoder = optim.Adam(self.decoder.parameters())

        self.device = device
        self.decoder.to(device)

    def _convert_act_list_to_str(self, actions):
        return ''.join([str(x) for x in actions])

    def decoder_train_step(self, obs_start, obs_end, actions):
        obs_stack = torch.cat([torch.FloatTensor(obs_start), torch.FloatTensor(obs_end)])
        seq = self._convert_act_list_to_str(actions)

        labels_seq = torch.LongTensor([self.actions_seqs[seq]]).to(self.device)
        logits_seq = self.decoder(obs_stack.to(self.device))

        loss_decoder = self.obj_decoder(logits_seq.unsqueeze(0), labels_seq)

        self.optim_decoder.zero_grad()
        loss_decoder.backward()
        self.optim_decoder.step()
        return loss_decoder.item()
