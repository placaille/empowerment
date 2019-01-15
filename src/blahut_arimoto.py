import gym
import click
import numpy as np

from tensorboardX import SummaryWriter
from itertools import product
from collections import OrderedDict

# custom code
import custom_envs
import utils

def compute_rollouts(env, state, actions_seqs):
    rollouts = {}
    for actions_seq in actions_seqs.keys():
        env.reset(state=state)
        for action in actions_seq:
            obs = env.step(action)
        rollouts[actions_seq] = obs.argmax()
    return rollouts


def em_loop(p, seq_dict, rollouts, epsilon):
    """
    Applies only to deterministic discrete env (e.g. grid-worl)
    """
    # make dict with action sequences leading to states (state: act_seq)
    seqs_leading_to_state = OrderedDict()
    for act_seq, state in rollouts.items():
        if state not in seqs_leading_to_state.keys():
            seqs_leading_to_state[state] = []
        seqs_leading_to_state[state].append(act_seq)

    # start EM
    d = get_d(p, seq_dict, rollouts, seqs_leading_to_state)
    old_emp = -float('inf')
    emp = np.sum(p * d)

    while emp - old_emp > epsilon:
        z = np.sum(p * np.exp(d))
        p *= 1 / z * np.exp(d)
        old_emp = emp
        emp = np.sum(p * d)
        d = get_d(p, seq_dict, rollouts, seqs_leading_to_state)
    return p, emp


def get_d(p, seq_dict, rollouts, seqs_leading_to_state):
    """
    custom for the state evaluated (and results tied to deterministic env)
    d_a = sum_s p(s'|s,a) log (p(s'|s,a) / sum_a p(s'|s,a)p(a|s))
    >>> since env is deterministic, p(s'|s,a) = 1, it can be simplified to
    >>> d_a = -log [sum_a p(a|s)], where a are ways to get to s' from s
    """
    d = np.empty_like(p)
    for act_seq in seq_dict.keys():
        seqs = seqs_leading_to_state[rollouts[act_seq]]
        d[seq_dict[act_seq]] = - np.log(np.sum([p[seq_dict[seq]] for seq in seqs]))
    return d


@click.command()
@click.argument('env-name', type=click.Choice([
    'TwoRoom-v0',
    'CrossRoom-v0',
    'RoomPlus2Corrid-v0',
]))
@click.option('--num-steps', type=int, default=2, help='Num steps for empowerment')
@click.option('--epsilon', type=float, default=1e-4, help='Margin to stop EM-algorithm')
@click.option('--out-file', type=click.Path(dir_okay=False, writable=True), default=None)
def main(env_name, num_steps, epsilon, out_file):
    """
    Used to compute empowerment for discrete, deterministic environment (grid-world)
    """
    env = gym.make(env_name)
    writer = SummaryWriter()

    # get sequences and creates dict for later use
    actions_keys = list(product(env.actions.values(), repeat=num_steps))
    actions_seqs = OrderedDict()
    for actions_key in actions_keys:
        actions_seqs[actions_key] = actions_keys.index(actions_key)

    # compute empowerment
    empowerment = []
    for state in env.free_states:
        rollouts = compute_rollouts(env, state, actions_seqs)
        p_a = np.ones(len(actions_seqs)) / len(actions_seqs)  # start with uniform probs
        p_a, emp = em_loop(p_a, actions_seqs, rollouts, epsilon)
        empowerment.append(emp)
    empowerment = np.array(empowerment)

    # convert to a map of empowerment
    all_states = np.eye(env.observation_space.n)
    states_i, states_j = zip(*env.free_pos)

    # init map value to avg empowerment to simplify color mapping later
    empowerment_map = np.full(env.grid.shape, empowerment.mean(), dtype=np.float32)
    empowerment_map[states_i, states_j] = empowerment

    utils.log_empowerment_map(writer, empowerment_map,
                              mask=env.grid != env.free,
                              tag='true_empowerment_{}_steps/{}'.format(num_steps, env_name),
                              file_name=out_file)

if __name__ == '__main__':
    main()
