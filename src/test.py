import gym
import click
from itertools import count

# custom code
import custom_envs

@click.command()
@click.argument('env-name', type=click.Choice([
    'TwoRoom-v0',
    'CrossRoom-v0',
    'RoomPlus2Corrid-v0',
]))
def main(**kwargs):
    env_name = kwargs.get('env')
    env = gym.make(env_name)
    env.reset()

    for iter in count(start=1):
        action = env.action_space.sample()
        obs = env.step(action)
        print(obs)
        if iter == 25:
            break


if __name__ == '__main__':
    main()
