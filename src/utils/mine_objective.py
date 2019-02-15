import torch


class UnbiasedLogMeanExp(torch.autograd.Function):
    """
    Class used to customize the gradients to account for the biased estimator
    >> This bias is reduced as per the MINE recommendations. The full paper
       can be found at https://arxiv.org/abs/1801.04062
    >> Execute the forward pass, but the backward pass result's is adjusting
       with the running average which is passed as an argument (in forward)
    """
    @staticmethod
    def forward(ctx, input, exp_moving_avg):
        exp = input.exp()
        ctx.save_for_backward(exp_moving_avg, exp)
        return exp.mean().log()

    @staticmethod
    def backward(ctx, grad_output):
        exp_moving_avg, exp = ctx.saved_tensors
        return grad_output * exp / exp_moving_avg / exp.shape[0], None


class UnbiasedMine:
    def __init__(self, ema_weight=0.9):
        self.ema_weight = ema_weight
        self.ema = 0

    def __call__(self, score_joint, score_marginal):
        self.ema = self.ema *(1-self.ema_weight) + self.ema_weight * score_marginal.exp().mean().detach()
        unbiased_log_mean_exp = UnbiasedLogMeanExp.apply(score_marginal, self.ema)
        return score_joint.mean() - unbiased_log_mean_exp.mean()

class BiasedMine:
    def __init__(self):
        pass

    def __call__(self, score_joint, score_marginal):
        return score_joint.mean() - score_marginal.exp().mean().log()
