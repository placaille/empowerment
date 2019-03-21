import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical, RelaxedOneHotCategorical


def get_empowerment_values(agent, env, num_sample=1000):
    with torch.no_grad():
        empowerment_values = []
        for state in env.free_states:
            init_obs_all = torch.FloatTensor(num_sample, len(env.free_states)).zero_()
            init_obs_all[:, state] = 1.0
            init_obs_iterator = DataLoader(init_obs_all.to(agent.device), batch_size=agent.max_batch_size)

            end_obs_all = []
            onehot_seq_all = []
            for init_obs in init_obs_iterator:
                seq_logits = F.log_softmax(agent.model_source_distr(init_obs), dim=-1)
                relaxed_distr = RelaxedOneHotCategorical(agent.temperature, logits=seq_logits)
                soft_onehot = relaxed_distr.rsample()
                action_seqs = [agent.actions_lists[seq_id.item()] for seq_id in soft_onehot.argmax(-1)]

                for (obs, action_seq) in zip(init_obs, action_seqs):
                    env.reset(state=state)
                    prev_obs = obs
                    for action in action_seq:
                        obs = env.step(action)

                    end_obs_all.append(obs)
                onehot_seq_all.append(soft_onehot)

            end_obs_all = torch.tensor(end_obs_all, dtype=torch.float32).to(agent.device)
            onehot_seq_all = torch.cat(onehot_seq_all).to(agent.device)
            shuffled_ids = torch.randperm(onehot_seq_all.shape[0])
            onehot_seq_all_shfld = onehot_seq_all[shuffled_ids]

            all_data = TensorDataset(init_obs_all, end_obs_all, onehot_seq_all, onehot_seq_all_shfld)
            all_data_iterator = DataLoader(all_data, batch_size=agent.max_batch_size)

            for i, batch in enumerate(all_data_iterator):
                obs_start, obs_end, seq_soft_onehot, seq_soft_onehot_shfl = batch

                stack_joint = torch.cat([obs_start, obs_end, seq_soft_onehot], dim=1)
                stack_marginal = torch.cat([obs_start, obs_end, seq_soft_onehot_shfl], dim=1)

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
