import gym
import click
import torch
import os
import sys
import shutil
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
@click.option('--diverg-name', default='js', type=click.Choice([
    'js',
    'kl'
]))
@click.option('--optim-name', default='adam', type=click.Choice([
    'adam',
    'sgd',
    'rmsprop',
]))
@click.option('--learning-rate', default=0.0001, type=float)
@click.option('--momentum', default=0.0, type=float)
@click.option('--comment', type=str, default=None, help='Comment stored in the args')
@click.option('--force-cpu', default=False, type=bool)
@click.option('--train-score', default=True, type=bool)
@click.option('--train-source-distr', default=True, type=bool)
@click.option('--num-steps', type=int, default=2, help='Num steps for empowerment')
@click.option('--hidden-size', type=int, default=32)
@click.option('--iter-per-eval', type=int, default=1000, help='Number of training iterations between evaluations')
@click.option('--max-iter', type=int, default=1000000)
@click.option('--emp-alpha', type=float, default=0.001, help='Emp table moving average weight (def. 0.001)')
@click.option('--entropy-weight', type=float, default=0.1, help='Entropy weight regularization (def. 0.1)')
@click.option('--gumbel-temp-start', type=float, default=0.5, help='Starting temp for Gumbel-Softax (def. 0.5)')
@click.option('--memory-size', type=int, default=100000)
@click.option('--samples-per-train', type=int, default=100)
@click.option('--batch_size', type=int, default=128)
def main(**kwargs):
    pre_trained_dir = os.path.expanduser(kwargs.get('pre_trained_dir'))
    log_dir = os.path.expanduser(kwargs.get('log_dir'))
    tensorboard_dir = os.path.expanduser(kwargs.get('tensorboard_dir'))
    env_name = kwargs.get('env_name')
    diverg_name = kwargs.get('diverg_name')
    optim_name = kwargs.get('optim_name')
    learning_rate = kwargs.get('learning_rate')
    momentum = kwargs.get('momentum')
    force_cpu = kwargs.get('force_cpu')
    train_score = kwargs.get('train_score')
    train_source_distr = kwargs.get('train_source_distr')
    num_steps = kwargs.get('num_steps')
    hidden_size = kwargs.get('hidden_size')
    iter_per_eval = kwargs.get('iter_per_eval')
    max_iter = kwargs.get('max_iter')
    emp_alpha = kwargs.get('emp_alpha')
    gumbel_temp_start = kwargs.get('gumbel_temp_start')
    entropy_weight = kwargs.get('entropy_weight')
    memory_size = kwargs.get('memory_size')
    samples_per_train = kwargs.get('samples_per_train')
    batch_size = kwargs.get('batch_size')

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

    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Initializing env..')
    env = gym.make(env_name)

    print('Initializing agent and models..')
    if pre_trained_dir is not None:
        path_score = None
        path_source_distr = None
        if not train_score:
            path_score = os.path.join(pre_trained_dir, '{}_score.pth'.format(env_name))
        if not train_source_distr:
            path_source_distr = os.path.join(pre_trained_dir, '{}_source_distr.pth'.format(env_name))
    agent = agents.fGANGumbelDiscreteStaticAgent(
        actions=env.actions,
        observation_size=env.observation_space.n,
        hidden_size=hidden_size,
        emp_num_steps=num_steps,
        alpha=emp_alpha,
        divergence_name=diverg_name,
        mem_size=2*batch_size,
        mem_fields=['obs_start', 'obs_end', 'act_seq', 'seq_soft_onehot'],
        max_batch_size=batch_size,
        temperature_start=gumbel_temp_start,
        train_score=train_score,
        train_source_distr=train_source_distr,
        path_score=path_score,
        path_source_distr=path_source_distr,
        optim_name=optim_name,
        learning_rate=learning_rate,
        momentum=momentum,
        device=device,
    )

    print('Initializing misc..')
    writer = SummaryWriter(tensorboard_dir)
    writer.add_text('args', str(kwargs))
    for (k, v) in kwargs.items():
        writer.add_text('{}'.format(k), str(v))
    action_seq = deque(maxlen=num_steps)
    cumul_loss_total = 0
    cumul_loss_joint = 0
    cumul_loss_marginal = 0
    start = timer()

    print('Starting training..')
    for iter in count(start=1):
        init_obs = [env.reset() for _ in range(2*batch_size)]
        action_seqs = agent.sample_source_distr(init_obs)
        for (obs, action_seq, soft_onehot) in zip(init_obs, action_seqs['actions'], action_seqs['soft_onehot']):
            env.reset(state=obs.argmax())
            prev_obs = obs
            for action in action_seq:
                obs = env.step(action)

            agent.memory.add_data(
                obs_start=prev_obs,
                obs_end=obs,
                act_seq=action_seq,
                seq_soft_onehot=soft_onehot,
            )
        losses = agent.train_step()
        cumul_loss_total += losses['loss_total']
        cumul_loss_joint += losses['loss_joint']
        cumul_loss_marginal += losses['loss_marginal']

        if iter % iter_per_eval == 0 or iter == max_iter:
            avg_loss_total = cumul_loss_total / iter_per_eval
            avg_loss_joint = cumul_loss_joint / iter_per_eval
            avg_loss_marginal = cumul_loss_marginal / iter_per_eval

            empowerment_map, emp_mean = agent.compute_empowerment_map(env)
            entropy_map, entr_mean = agent.compute_entropy_map(env)

            env_step_tag = '{}-{}-steps'.format(env_name, num_steps)
            tag_emp = 'emp_{}'.format(env_step_tag)
            tag_ent = 'entropy_{}'.format(env_step_tag)
            agent.save_models(tag=tag_emp+'_', out_dir=os.path.join(log_dir, 'models'))
            utils.log_value_map(writer, empowerment_map,
                                      mask=env.grid != env.free,
                                      tag=tag_emp,
                                      global_step=iter,
                                      file_name=os.path.join(log_dir, 'maps', tag_emp))
            utils.log_value_map(writer, entropy_map,
                                      mask=env.grid != env.free,
                                      tag=tag_ent,
                                      global_step=iter,
                                      file_name=os.path.join(log_dir, 'maps', tag_ent))
            writer.add_scalar('loss-{}/total'.format(env_step_tag), avg_loss_total, iter)
            writer.add_scalar('loss-{}/joint'.format(env_step_tag), avg_loss_joint, iter)
            writer.add_scalar('loss-{}/marginal'.format(env_step_tag), avg_loss_marginal, iter)
            writer.add_scalar('empowerment-{}/min'.format(env_step_tag), empowerment_map.min(), iter)
            writer.add_scalar('empowerment-{}/max'.format(env_step_tag), empowerment_map.max(), iter)
            writer.add_scalar('empowerment-{}/mean'.format(env_step_tag), emp_mean, iter)
            writer.add_scalar('entropy-{}/min'.format(env_step_tag), entropy_map.min(), iter)
            writer.add_scalar('entropy-{}/max'.format(env_step_tag), entropy_map.max(), iter)
            writer.add_scalar('entropy-{}/mean'.format(env_step_tag), entr_mean, iter)

            print('iter {:8d} - loss {:6.4f} - empowerment {:6.4f} - entropy {:6.4f} - {:4.1f}s'.format(
                iter,
                avg_loss_total,
                emp_mean,
                entr_mean,
                timer() - start,
            ))
            cumul_loss_total = 0
            cumul_loss_joint = 0
            cumul_loss_marginal = 0
            agent.anneal_temperature(iter)
            start = timer()

        if iter >= max_iter:
            break


if __name__ == '__main__':
    main()