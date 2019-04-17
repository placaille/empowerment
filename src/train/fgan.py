import gym
import click
import torch
import os
import sys
import shutil
import random
import numpy as np
import pickle as pkl

from itertools import count
from collections import deque, namedtuple
from tensorboardX import SummaryWriter
from timeit import default_timer as timer

sys.path.append('src')

# custom code
import custom_envs
import agents
import utils

@click.command(cls=utils.CommandWithConfigFile('config_file'))
@click.option('--config-file', '-c', type=click.Path(exists=True, dir_okay=False))
@click.option('--pre-trained-dir', type=click.Path(file_okay=False, exists=True, readable=True), default='./models')
@click.option('--log-dir', type=click.Path(file_okay=False, exists=True, writable=True), default='./out')
@click.option('--tensorboard-dir', default=None)
@click.option('--env-name', default='TwoRoom-v0', type=click.Choice([
    'TwoRoom-v0',
    'CrossRoom-v0',
    'RoomPlus2Corrid-v0',
]))
@click.option('--diverg-name', default='js', type=click.Choice(['js', 'kl']))
@click.option('--seed', default=None)
@click.option('--optim-name', default='adam', type=click.Choice(['adam', 'sgd', 'rmsprop']))
@click.option('--lr', default=0.0001, type=float)
@click.option('--momentum', default=0.0, type=float)
@click.option('--weight-decay', type=float, default=0)
@click.option('--comment', type=str, default=None, help='Comment stored in the args')
@click.option('--force-cpu', default=False, type=bool)
@click.option('--num-steps', type=int, default=2, help='Num steps for empowerment')
@click.option('--hidden-size', type=int, default=32)
@click.option('--iter-per-eval', type=int, default=1000, help='Number of training iterations between evaluations')
@click.option('--iter-per-train', type=int, default=100, help='Number of rollouts between training steps')
@click.option('--iter-before-train', type=int, default=20000)
@click.option('--max-iter', type=int, default=1000000)
@click.option('--memory-size', type=int, default=100000)
@click.option('--batch-size', type=int, default=128)
@click.option('--samples-for-grad', type=int, default=1, help='Num samples for inner expectations')
def main(**kwargs):
    pre_trained_dir = os.path.expanduser(kwargs.get('pre_trained_dir'))
    log_dir = os.path.expanduser(kwargs.get('log_dir'))
    tensorboard_dir = os.path.expanduser(kwargs.get('tensorboard_dir'))
    env_name = kwargs.get('env_name')
    seed = kwargs.get('seed')
    num_steps = kwargs.get('num_steps')
    iter_per_eval = kwargs.get('iter_per_eval')
    iter_per_train = kwargs.get('iter_per_train')
    iter_before_train = kwargs.get('iter_before_train')
    max_iter = kwargs.get('max_iter')

    if tensorboard_dir is None:
        tensorboard_dir = log_dir
    print('---')
    for (k, v) in kwargs.items():
        print("{}: {}".format(k, v))
    print('---')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.info'), 'w') as f:
        f.write(str(kwargs))

    if  kwargs.get('force_cpu'):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('Initializing env..')
    env = gym.make(env_name)
    if seed is not None:
        env.seed(int(seed))

    print('Initializing agent and models..')
    agent = agents.fGANDiscreteStaticAgent(
        actions=env.actions,
        observation_size=env.observation_space.n,
        emp_num_steps=num_steps,
        mem_fields=['obs_start', 'obs_end', 'act_seq', 'seq_onehot'],
        device=device,
        **kwargs,
    )

    print('Initializing misc..')
    writer = SummaryWriter(tensorboard_dir)
    writer.add_text('args', str(kwargs))
    for (k, v) in kwargs.items():
        writer.add_text('{}'.format(k), str(v))
    cumul_loss = {
        'score_total': 0,
        'score_joint': 0,
        'score_marginal': 0,
        'emp_total': 0,
        'policy_total': 0,
    }
    start = timer()

    print('Generating random data..')
    for iter in range(iter_before_train):
        state = env.reset()
        agent.generate_on_policy_rollouts(env, [state], num_rollouts=1, add_to_memory=True)

    print('Starting training..')
    for iter in count(start=1):

        train_out = agent.train_step(env)

        cumul_loss['score_total'] += train_out['score']['loss_total']
        cumul_loss['score_joint'] += train_out['score']['loss_joint']
        cumul_loss['score_marginal'] += train_out['score']['loss_marginal']
        cumul_loss['emp_total'] += train_out['emp']['loss_total']
        cumul_loss['policy_total'] += train_out['policy']['loss_total']

        # log stuff
        if iter % iter_per_eval == 0 or iter == max_iter:
            avg_loss_score_total = cumul_loss['score_total'] / iter_per_eval
            avg_loss_score_joint = cumul_loss['score_joint'] / iter_per_eval
            avg_loss_score_marginal = cumul_loss['score_marginal'] / iter_per_eval
            avg_loss_emp = cumul_loss['emp_total'] / iter_per_eval
            avg_loss_policy = cumul_loss['policy_total'] / iter_per_eval

            empowerment_map, emp_mean = agent.compute_empowerment_map(env, kwargs.get('samples_for_eval'))
            entropy_map, entr_mean = agent.compute_entropy_map(env)

            env_step_tag = '{}-{}-steps'.format(env_name, num_steps)
            tag_emp = 'emp_{}'.format(env_step_tag)
            tag_pred = 'emp_pred_{}'.format(env_step_tag)
            tag_ent = 'entropy_{}'.format(env_step_tag)
            agent.save_models(tag=tag_emp+'_', out_dir=os.path.join(log_dir, 'models'))
            utils.log_value_map(writer, empowerment_map,
                                      mask=env.grid != env.free,
                                      tag=tag_pred,
                                      global_step=iter,
                                      file_name=os.path.join(log_dir, 'maps', tag_pred))
            utils.log_value_map(writer, entropy_map,
                                      mask=env.grid != env.free,
                                      tag=tag_ent,
                                      global_step=iter,
                                      file_name=os.path.join(log_dir, 'maps', tag_ent))
            writer.add_scalar('loss-{}/score-total'.format(env_step_tag), avg_loss_score_total, iter)
            writer.add_scalar('loss-{}/score-joint'.format(env_step_tag), avg_loss_score_joint, iter)
            writer.add_scalar('loss-{}/score-marginal'.format(env_step_tag), avg_loss_score_marginal, iter)
            writer.add_scalar('loss-{}/emp-total'.format(env_step_tag), avg_loss_emp, iter)
            writer.add_scalar('loss-{}/policy-total'.format(env_step_tag), avg_loss_policy, iter)
            writer.add_scalar('empowerment-{}/min'.format(env_step_tag), empowerment_map.min(), iter)
            writer.add_scalar('empowerment-{}/max'.format(env_step_tag), empowerment_map.max(), iter)
            writer.add_scalar('empowerment-{}/mean'.format(env_step_tag), emp_mean, iter)
            writer.add_scalar('entropy-{}/min'.format(env_step_tag), entropy_map.min(), iter)
            writer.add_scalar('entropy-{}/max'.format(env_step_tag), entropy_map.max(), iter)
            writer.add_scalar('entropy-{}/mean'.format(env_step_tag), entr_mean, iter)

            print('iter {:8d} - loss s/e/p {:5.3f}/{:5.3f}/{:5.3f} - emp {:5.3f} - entr {:5.3f} - {:4.1f}s'.format(
                iter,
                avg_loss_score_total,
                avg_loss_emp,
                avg_loss_policy,
                emp_mean,
                entr_mean,
                timer() - start,
            ))
            cumul_loss['score_total'] = 0
            cumul_loss['score_joint'] = 0
            cumul_loss['score_marginal'] = 0
            cumul_loss['emp_total'] = 0
            cumul_loss['policy_total'] = 0
            start = timer()

        if iter >= max_iter:
            break


if __name__ == '__main__':
    main()
