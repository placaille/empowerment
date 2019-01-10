import gym
import click
import torch

from itertools import count
from collections import deque

# custom code
import custom_envs
import agents


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.argument('env-name', type=click.Choice([
    'TwoRoom-v0',
    'CrossRoom-v0',
    'RoomPlus2Corrid-v0',
]))
@click.option('--num-steps', type=int, default=2, help='Num steps for empowerment')
@click.option('--hidden-size', type=int, default=32, help='Num steps for empowerment')
def main(**kwargs):
    print(kwargs)
    env_name = kwargs.get('env_name')
    num_steps = kwargs.get('num_steps')
    hidden_size = kwargs.get('hidden_size')

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
    obs = env.reset()
    action_seq = deque(maxlen=num_steps)
    cumul_decoder_loss = 0
    cumul_energy_loss = 0

    print('Starting training..')
    for iter in count(start=1):
        prev_obs = obs
        for _ in range(num_steps):
            action = env.action_space.sample()
            action_seq.append(action)
            obs = env.step(action)

        loss_decoder = agent.decoder_train_step(prev_obs, obs, action_seq)
        loss_energy = agent.energy_train_step(prev_obs, obs, action_seq)

        cumul_decoder_loss += loss_decoder
        cumul_energy_loss += loss_energy

        if iter % 1000 == 0:
            print('loss decoder/energy: {:6.4f}/{:6.4f}'.format(
                cumul_decoder_loss / 1000,
                cumul_energy_loss / 1000
            ))
            cumul_decoder_loss = 0
            cumul_energy_loss = 0

        if iter == 100000:
            break


if __name__ == '__main__':
    main()
