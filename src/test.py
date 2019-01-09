import gym
import click
import torch
from itertools import count

# custom code
import custom_envs
import agents

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

    env = gym.make(env_name)

    print('Initializing agent and models..')
    agent = agents.DiscreteStaticAgent(
        actions=env.actions,
        observation_size=env.observation_space.n,
        hidden_size=hidden_size,
        emp_num_steps=num_steps
    )

    prev_obs = env.reset()
    for iter in count(start=1):
        action = env.action_space.sample()
        obs = env.step(action)
        logits = agent.get_logits(torch.FloatTensor(prev_obs), torch.FloatTensor(obs))
        print(logits.shape)
        if iter == 10:
            break
        prev_obs = obs


if __name__ == '__main__':
    main()
