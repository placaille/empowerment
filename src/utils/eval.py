import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical, RelaxedOneHotCategorical

@torch.no_grad()
def get_empowerment_values(agent, env, num_sample=1000):
    empowerment_values = []
    for state in env.free_states:
        init_obs_all = torch.FloatTensor(num_sample, len(env.free_states)).zero_().to(agent.device)
        init_obs_all[:, state] = 1.0
        init_obs_iterator = DataLoader(init_obs_all, batch_size=agent.max_batch_size)

        end_obs_all = []
        onehot_seq_all = []
        for init_obs in init_obs_iterator:
            out = agent.sample_source_distr(init_obs)
            for (obs, action_seq) in zip(init_obs, out['actions']):
                env.reset(state=state)
                prev_obs = obs
                for action in action_seq:
                    obs = env.step(action)

                end_obs_all.append(obs)
            onehot_seq_all.append(out['onehot'])

        end_obs_all = torch.tensor(end_obs_all, dtype=torch.float32).to(agent.device)
        onehot_seq_all = torch.cat(onehot_seq_all).to(agent.device)
        shuffled_ids = torch.randperm(onehot_seq_all.shape[0])
        onehot_seq_all_shfld = onehot_seq_all[shuffled_ids]

        all_data = TensorDataset(init_obs_all, end_obs_all, onehot_seq_all, onehot_seq_all_shfld)
        all_data_iterator = DataLoader(all_data, batch_size=agent.max_batch_size)

        for i, batch in enumerate(all_data_iterator):
            obs_start, obs_end, seq_onehot, seq_onehot_shfl = batch

            stack_joint = torch.cat([obs_start, obs_end, seq_onehot], dim=1)
            stack_marginal = torch.cat([obs_start, obs_end, seq_onehot_shfl], dim=1)

            logits_joint = agent.model_score(stack_joint)
            logits_marginal = agent.model_score(stack_marginal)

            constant, scores_joint, scores_marginal = agent.obj(logits_joint, logits_marginal)
            if i == 0:
                constant_sum = constant.data.sum()
                scores_joint_sum = scores_joint.data.sum()
                scores_marginal_sum = scores_marginal.data.sum()
            else:
                constant_sum += constant.data.sum()
                scores_joint_sum += scores_joint.data.sum()
                scores_marginal_sum += scores_marginal.data.sum()

        empowerment = (constant_sum + scores_joint_sum - scores_marginal_sum) / num_sample
        empowerment_values.append(empowerment.item())

    return np.array(empowerment_values)
