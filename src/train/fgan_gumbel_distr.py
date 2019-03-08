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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@click.command(cls=utils.CommandWithConfigFile('config_file'))
@click.option('--config-file', '-c', type=click.Path(exists=True, dir_okay=False))
@click.option('--env-name', default='TwoRoom-v0', type=click.Choice([
    'TwoRoom-v0',
    'CrossRoom-v0',
    'RoomPlus2Corrid-v0',
]))
@click.option('--diverg-name', default='js', type=click.Choice([
    'js',
    'kl'
]))
@click.option('--comment', type=str, default=None, help='Comment stored in the args')
@click.option('--pre-trained-dir', type=click.Path(file_okay=False, exists=True, readable=True))
@click.option('--train-score', default=True, type=bool)
@click.option('--train-source-distr', default=True, type=bool)
@click.option('--log-dir', type=click.Path(file_okay=False, exists=True, writable=True), default='./out')
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
    env_name = kwargs.get('env_name')
    diverg_name = kwargs.get('diverg_name')
    pre_trained_dir = kwargs.get('pre_trained_dir')
    train_score = kwargs.get('train_score')
    train_source_distr = kwargs.get('train_source_distr')
    log_dir = kwargs.get('log_dir')
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
    print('---')
    for (k, v) in kwargs.items():
        print("{}: {}".format(k, v))
    print('---')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.info'), 'w') as f:
        f.write(str(kwargs))

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
        mem_size=batch_size,
        mem_fields=['obs_start', 'obs_end', 'act_seq', 'seq_soft_onehot'],
        max_batch_size=batch_size,
        temperature_start=gumbel_temp_start,
        train_score=train_score,
        train_source_distr=train_source_distr,
        path_score=path_score,
        path_source_distr=path_source_distr,
        device=device,
    )

    print('Initializing misc..')
    writer = SummaryWriter(log_dir)
    writer.add_text('args', str(kwargs))
    for (k, v) in kwargs.items():
        writer.add_text('{}'.format(k), str(v))
    action_seq = deque(maxlen=num_steps)
    cumul_loss = 0
    start = timer()

    print('Starting training..')
    for iter in count(start=1):
        init_obs = [env.reset() for _ in range(batch_size)]
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
        loss = agent.train_step()
        cumul_loss += loss

        if iter % iter_per_eval == 0 or iter == max_iter:
            avg_loss = cumul_loss / iter_per_eval

            empowerment_map = agent.compute_empowerment_map(env)
            entropy_map = agent.compute_entropy_map(env)

            tag_emp='emp_{}_steps_{}'.format(num_steps, env_name)
            tag_ent='entropy_{}_steps_{}'.format(num_steps, env_name)
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
            writer.add_scalar('loss-{}-{}step/fgan'.format(env_name, num_steps), avg_loss, iter)
            writer.add_scalar('empowerment-{}-{}step/min'.format(env_name, num_steps), empowerment_map.min(), iter)
            writer.add_scalar('empowerment-{}-{}step/max'.format(env_name, num_steps), empowerment_map.max(), iter)
            writer.add_scalar('entropy-{}-{}step/min'.format(env_name, num_steps), entropy_map.min(), iter)
            writer.add_scalar('entropy-{}-{}step/max'.format(env_name, num_steps), entropy_map.max(), iter)

            print('iter {:8d} - loss fgan {:6.4f} - entropy {:6.4f} - {:4.1f}s'.format(
                iter,
                avg_loss,
                entropy_map.mean(),
                timer() - start,
            ))
            cumul_loss = 0
            agent.anneal_temperature(iter)
            start = timer()

        if iter >= max_iter:
            break


if __name__ == '__main__':
    main()
