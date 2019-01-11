import gym
import click
import torch

from itertools import count
from collections import deque
from tensorboardX import SummaryWriter
from timeit import default_timer as timer

# custom code
import custom_envs
import agents
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@click.command()
@click.argument('env-name', type=click.Choice([
    'TwoRoom-v0',
    'CrossRoom-v0',
    'RoomPlus2Corrid-v0',
]))
@click.option('--num-steps', type=int, default=2, help='Num steps for empowerment')
@click.option('--hidden-size', type=int, default=32)
@click.option('--batch-per-eval', type=int, default=10000)
@click.option('--source-beta', type=float, default=1.0)
def main(**kwargs):
    print(kwargs)
    env_name = kwargs.get('env_name')
    num_steps = kwargs.get('num_steps')
    hidden_size = kwargs.get('hidden_size')
    batch_per_eval = kwargs.get('batch_per_eval')
    source_beta = kwargs.get('source_beta')

    print('Initializing env..')
    env = gym.make(env_name)

    print('Initializing agent and models..')
    agent = agents.DiscreteStaticAgent(
        actions=env.actions,
        observation_size=env.observation_space.n,
        hidden_size=hidden_size,
        emp_num_steps=num_steps,
        beta=1.0,
        device=device,
    )

    print('Initializing misc..')
    writer = SummaryWriter()
    action_seq = deque(maxlen=num_steps)
    cumul_loss_decoder = 0
    cumul_loss_source = 0
    start = timer()

    print('Starting training..')
    for iter in count(start=1):
        obs = env.reset()
        prev_obs = obs
        for _ in range(num_steps):
            action = env.action_space.sample()
            action_seq.append(action)
            obs = env.step(action)

        loss_decoder = agent.decoder_train_step(prev_obs, obs, action_seq)
        loss_source = agent.source_train_step(prev_obs, obs, action_seq)

        cumul_loss_decoder += loss_decoder
        cumul_loss_source += loss_source

        if iter % batch_per_eval == 0:
            avg_loss_decoder = cumul_loss_decoder / batch_per_eval
            avg_loss_source = cumul_loss_source / batch_per_eval
            empowerment_map = agent.compute_empowerment_map(env)

            utils.log_empowerment_map(empowerment_map, env, writer, 'empowerment', iter)
            utils.log_loss(avg_loss_decoder, writer, 'loss/decoder', iter)
            utils.log_loss(avg_loss_source, writer, 'loss/source', iter)

            print('iter {:8d} - loss tot {:6.4f} - {:4.1f}s'.format(
                iter,
                avg_loss_decoder + avg_loss_source,
                timer() - start,
            ))
            cumul_loss_decoder = 0
            cumul_loss_source = 0
            start = timer()

        if iter == 100000:
            break


if __name__ == '__main__':
    main()
