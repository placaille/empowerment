import gym
import click
import torch
import os
import sys
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
@click.option('--pre-trained-dir', type=click.Path(file_okay=False, exists=True, readable=True))
@click.option('--log-dir', type=click.Path(file_okay=False, exists=True, writable=True), default='./out')
@click.option('--num-steps', type=int, default=2, help='Num steps for empowerment')
@click.option('--hidden-size', type=int, default=32)
@click.option('--iter-per-eval', type=int, default=1000)
@click.option('--max-iter', type=int, default=1000000)
@click.option('--emp-alpha', type=float, default=0.001, help='Emp table moving average weight (def. 0.001)')
@click.option('--memory-size', type=int, default=100000)
@click.option('--samples-per-train', type=int, default=100)
@click.option('--batch_size', type=int, default=128)
def main(**kwargs):
    env_name = kwargs.get('env_name')
    diverg_name = kwargs.get('diverg_name')
    pre_trained_dir = kwargs.get('pre_trained_dir')
    log_dir = kwargs.get('log_dir')
    num_steps = kwargs.get('num_steps')
    hidden_size = kwargs.get('hidden_size')
    iter_per_eval = kwargs.get('iter_per_eval')
    max_iter = kwargs.get('max_iter')
    emp_alpha = kwargs.get('emp_alpha')
    memory_size = kwargs.get('memory_size')
    samples_per_train = kwargs.get('samples_per_train')
    batch_size = kwargs.get('batch_size')
    print(kwargs)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.info'), 'w') as f:
        f.write(str(kwargs))

    print('Initializing env..')
    env = gym.make(env_name)

    print('Initializing agent and models..')
    source_distr_path = os.path.join(pre_trained_dir, '{}_source_distr.pth'.format(env_name))
    agent = agents.fGANReproDiscreteStaticAgent(
        actions=env.actions,
        observation_size=env.observation_space.n,
        hidden_size=hidden_size,
        emp_num_steps=num_steps,
        alpha=emp_alpha,
        divergence_name=diverg_name,
        mem_size=memory_size,
        mem_fields=['obs_start', 'obs_end', 'act_seq'],
        max_batch_size=batch_size,
        path_source_distr=source_distr_path,
        device=device,
    )

    print('Initializing misc..')
    writer = SummaryWriter(log_dir)
    writer.add_text('args', str(kwargs))
    action_seq = deque(maxlen=num_steps)
    cumul_loss_score = 0
    start = timer()

    print('Starting training..')
    for iter in count(start=1):
        for _ in range(samples_per_train):
            obs = env.reset()
            prev_obs = obs
            for _ in range(num_steps):
                action = env.action_space.sample()
                action_seq.append(action)
                obs = env.step(action)

            agent.memory.add_data(obs_start=prev_obs, obs_end=obs, act_seq=list(action_seq))
        loss_score = agent.score_train_step()
        cumul_loss_score += loss_score

        if iter % iter_per_eval == 0 or iter == max_iter:
            avg_loss_score = cumul_loss_score / iter_per_eval

            empowerment_map = agent.compute_empowerment_map(env)

            tag='emp_{}_steps_{}'.format(num_steps, env_name)
            agent.save_models(tag=tag+'_', out_dir=os.path.join(log_dir, 'models'))
            utils.log_empowerment_map(writer, empowerment_map,
                                      mask=env.grid != env.free,
                                      tag=tag,
                                      global_step=iter,
                                      file_name=os.path.join(log_dir, 'maps', tag))
            writer.add_scalar('loss/fgan_score', avg_loss_score, iter)
            writer.add_scalar('empowerment/min', empowerment_map.min(), iter)
            writer.add_scalar('empowerment/max', empowerment_map.max(), iter)

            print('iter {:8d} - loss score {:6.4f} - {:4.1f}s'.format(
                iter,
                avg_loss_score,
                timer() - start,
            ))
            cumul_loss_score = 0
            start = timer()

        if iter >= max_iter:
            break


if __name__ == '__main__':
    main()
