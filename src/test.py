import gym
import click

# custom code
import custom_envs

@click.command()
@click.argument('env', type=click.Choice([
    'TwoRoom-v0',
    'CrossRoom-v0',
    'RoomPlus2Corrid-v0',
]))
def main(**kwargs):
    env_name = kwargs.get('env')
    env = gym.make(env_name)
    print(env)
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
