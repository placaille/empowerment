import torch

from itertools import product

# custom code
import models


class DiscreteStaticAgent(object):
    def __init__(self, actions, observation_size, hidden_size, emp_num_steps):
        assert type(actions) is dict

        actions_id = [str(x) for x in actions.values()]
        self.actions_keys = [''.join(act_seq) for act_seq in product(actions_id, repeat=emp_num_steps)]

        # model used to compute likelihood of action sequences
        self.act_seq_decoder = models.MLP(2*observation_size, hidden_size, len(self.actions_keys))

    def get_logits(self, obs_start, obs_end):
        return self.act_seq_decoder(torch.cat([obs_start, obs_end]).unsqueeze(0))
