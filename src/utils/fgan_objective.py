import torch
import math
from torch.nn import functional as F

class fGAN:
    def __init__(self, divergence_name):
        """
        divergence_name (str): ['kl', 'js']

        score/discriminator obj
        kl: E_pos[logits] - E_neg[exp(logits-1)]
        js: E_pos[log(sigmoid(logits))] - E_neg[-log(1-sigmoid(logits))]

        source distr obj

        """
        assert divergence_name in ['kl', 'js'], 'invalid name'
        self.divergence_name = divergence_name

        if self.divergence_name == 'kl':
            self.constant = 0
            self.pos_score = lambda logits: logits
            self.neg_score = lambda logits: (logits-1).exp()
        elif self.divergence_name == 'js':
            self.constant = math.log(4)
            self.pos_score = lambda logits: F.logsigmoid(logits)
            self.neg_score = lambda logits: -(F.logsigmoid(logits) - logits)

    def discr_obj(self, pos_logits, neg_logits):
        """
        objective (to maximize) for the discriminator
        *** RETURNS ALL ELEMENTS IN TENSORS (NOT THE MEAN)
        """
        constant = torch.tensor(self.constant).float().expand(pos_logits.shape[0], 1)
        return constant.to(pos_logits.device), self.pos_score(pos_logits), self.neg_score(neg_logits)
