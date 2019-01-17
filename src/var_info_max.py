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
@click.option('--iter-per-eval', type=int, default=10000)
@click.option('--max-iter', type=int, default=1000000)
@click.option('--source-beta', type=float, default=1.0)
@click.option('--out-file', '-o', type=click.Path(dir_okay=False, writable=True), default=None)
@click.option('--memory-size', type=int, default=10000)
def main(**kwargs):
    print(kwargs)
    env_name = kwargs.get('env_name')
    num_steps = kwargs.get('num_steps')
    hidden_size = kwargs.get('hidden_size')
    iter_per_eval = kwargs.get('iter_per_eval')
    max_iter = kwargs.get('max_iter')
    source_beta = kwargs.get('source_beta')
    out_file = kwargs.get('out_file')
    memory_size = kwargs.get('memory_size')

    print('Initializing env..')
    env = gym.make(env_name)

    print('Initializing agent and models..')
    agent = agents.DiscreteStaticAgent(
        actions=env.actions,
        observation_size=env.observation_space.n,
        hidden_size=hidden_size,
        emp_num_steps=num_steps,
        beta=1.0,
        mem_size=memory_size,
        mem_fields=['obs_start', 'obs_end', 'act_seq'],
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

        agent.memory.add_data(obs_start=prev_obs, obs_end=obs, act_seq=list(action_seq))
        loss_decoder = agent.decoder_train_step()
        loss_source = agent.source_train_step()

        cumul_loss_decoder += loss_decoder
        cumul_loss_source += loss_source

        if iter % iter_per_eval == 0:
            avg_loss_decoder = cumul_loss_decoder / iter_per_eval
            avg_loss_source = cumul_loss_source / iter_per_eval
            empowerment_map = agent.compute_empowerment_map(env)

            utils.log_empowerment_map(writer, empowerment_map,
                                      mask=env.grid != env.free,
                                      tag='empowerment_{}_steps/{}'.format(num_steps, env_name),
                                      global_step=iter,
                                      file_name=out_file)
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

        if iter >= max_iter:
            break


if __name__ == '__main__':
    main()
