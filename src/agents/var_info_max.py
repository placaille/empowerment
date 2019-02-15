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


class VarInfoDiscreteStaticAgent(object):
    def __init__(self, actions, observation_size, hidden_size, emp_num_steps,
                 beta, mem_size, mem_fields, max_batch_size, device='cpu'):
        assert type(actions) is dict
        self.device = device
        self.beta = beta
        self.max_batch_size = max_batch_size

        self.actions = actions
        actions_id = [str(x) for x in self.actions.values()]
        self.actions_keys = [''.join(act_seq) for act_seq in product(actions_id, repeat=emp_num_steps)]

        self.actions_seqs = {}
        for actions_key in self.actions_keys:
            self.actions_seqs[actions_key] = self.actions_keys.index(actions_key)

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

        self.memory = utils.Memory(mem_size, *mem_fields)

    def _convert_act_list_to_str(self, actions):
        return ''.join([str(x) for x in actions])

    def source_train_step(self):
        batch = self.memory.sample_data(self.max_batch_size)

        obs_start = torch.FloatTensor(batch.obs_start)
        obs_stack = torch.cat([obs_start, torch.FloatTensor(batch.obs_end)], dim=1)

        seq_id = [self.actions_seqs[self._convert_act_list_to_str(seq)] for seq in batch.act_seq]
        seq_id = torch.LongTensor(seq_id).to(self.device)

        logits_seq = self.decoder(obs_stack.to(self.device))
        log_softmax_seq = F.log_softmax(logits_seq, dim=1).detach()
        target_energy = self.beta * log_softmax_seq.gather(1, seq_id.unsqueeze(1))

        logits_src_distr = self.model_source_distr(obs_start.to(self.device))
        log_softmax_src_distr = F.log_softmax(logits_src_distr, dim=1)
        log_likelihood_src = log_softmax_src_distr.gather(1, seq_id.unsqueeze(1))

        phi = self.model_phi(obs_start.to(self.device))
        preds_energy = log_likelihood_src + phi

        loss_source = self.obj_source(preds_energy, target_energy)

        self.optim_source.zero_grad()
        loss_source.backward()
        self.optim_source.step()
        return loss_source.item()

    def decoder_train_step(self):
        batch = self.memory.sample_data(self.max_batch_size)

        obs_stack = torch.cat([torch.FloatTensor(batch.obs_start),
                               torch.FloatTensor(batch.obs_end)], dim=1)
        seq_id = [self.actions_seqs[self._convert_act_list_to_str(seq)] for seq in batch.act_seq]

        labels_seq = torch.LongTensor(seq_id).to(self.device)
        logits_seq = self.decoder(obs_stack.to(self.device))

        loss_decoder = self.obj_decoder(logits_seq, labels_seq)

        self.optim_decoder.zero_grad()
        loss_decoder.backward()
        self.optim_decoder.step()
        return loss_decoder.item()

    def compute_empowerment(self, state):
        with torch.no_grad():
            phi = self.model_phi(state)
        return 1 / self.beta * phi.cpu().numpy()

    def compute_empowerment_map(self, env):
        all_states = np.eye(env.observation_space.n)
        states_i, states_j = zip(*env.free_pos)

        empowerment = []
        for start_id in range(0, env.observation_space.n, self.max_batch_size):
            states = torch.FloatTensor(all_states[start_id:start_id+self.max_batch_size])
            empowerment.append(self.compute_empowerment(states.to(self.device)))
        empowerment = np.concatenate(empowerment).squeeze()

        # init map value to avg empowerment to simplify color mapping later
        empowerment_map = np.full(env.grid.shape, empowerment.mean(), dtype=np.float32)
        empowerment_map[states_i, states_j] = empowerment
        return empowerment_map

    def sample_source_sequence(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            seq_logits = self.model_source_distr(obs.to(self.device))
            seq_distr = Categorical(logits=seq_logits)
        return seq_distr.sample().item()

    def save_models(self, tag, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.decoder.save(os.path.join(out_dir, tag + 'decoder.pth'))
        self.model_phi.save(os.path.join(out_dir, tag + 'phi.pth'))
        self.model_source_distr.save(os.path.join(out_dir, tag + 'source_distr.pth'))
