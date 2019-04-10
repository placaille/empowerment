import os
import random
import math
import itertools
import argparse

def linear_scale_float_sampling(min, max):
    return random.uniform(min, max)

def linear_scale_int_sampling(min, max):
    return random.randint(min, max)

def exp_scale_int_sampling(min, max, base=10):
    min_log, max_log = math.log(min, base), math.log(max, base)
    linear_int_sample = linear_scale_int_sampling(min_log, max_log)
    return int(math.pow(base, linear_int_sample))

def exp_scale_float_sampling(min, max, base=10, rounding=4):
    min_log, max_log = math.log(min, base), math.log(max, base)
    linear_sample = linear_scale_float_sampling(min_log, max_log)
    return round(math.pow(base, linear_sample), rounding)

def categorical_sampling(iterable):
    return random.choice(iterable)

def hybrid_float_sampling(categorical_value, float_prob, min, max, base=10):
    """with prob p, sample the float, otherwise return categorical value"""
    if random.random() < float_prob:
        return exp_scale_float_sampling(min, max, base)
    else:
        return categorical_value

def main(args):
    env_name='CrossRoom-v0'
    num_steps=2
    max_iter=10000000
    iter_per_eval=100000
    force_cpu='true'
    train_score='true'
    train_source_distr='true'
    batch_size = 1
    pre_trained_dir='~/Scratch/projects/augusta/pre_trained_models/fgan/repro/2_steps/kl'
    diverg_name='kl'
    comment= 'testing new log prob form'

    learning_rate_fn = lambda: exp_scale_float_sampling(min=10**-5, max=10**-1, base=10, rounding=7)
    momentum_fn = lambda: hybrid_float_sampling(categorical_value=0.0, float_prob=0.5, min=0.0001, max=0.9, base=10)
    optim_name_fn = lambda: categorical_sampling(['sgd', 'adam', 'rmsprop'])

    num_configs = args.num_configs
    config_id_start = args.config_id_start

    config_id_max = config_id_start + num_configs - 1
    config_id = config_id_start
    while config_id <= config_id_max:

        file_name = os.path.join('./configs', '{0:03d}.conf'.format(config_id))
        with open(file_name, 'w') as f:
            f.write('env-name: {}\n'.format(env_name))
            f.write('num-steps: {}\n'.format(num_steps))
            f.write('max-iter: {}\n'.format(max_iter))
            f.write('iter-per-eval: {}\n'.format(iter_per_eval))
            f.write('force-cpu: {}\n'.format(force_cpu))
            f.write('train-score: {}\n'.format(train_score))
            f.write('train-source-distr: {}\n'.format(train_source_distr))
            f.write('pre-trained-dir: {}\n'.format(pre_trained_dir))
            f.write('diverg-name: {}\n'.format(diverg_name))
            f.write('batch-size: {}\n'.format(batch_size))
            f.write('comment: {}\n'.format(comment))

            # sampling hyper-params
            l = learning_rate_fn()
            m = momentum_fn()
            o = optim_name_fn()
            if o == 'adam' and m != 0.0:
                continue

            f.write('learning-rate: {}\n'.format(l))
            f.write('momentum: {}\n'.format(m))
            f.write('optim-name: {}\n'.format(o))

        print(file_name)
        config_id += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='num_configs', type=int, default=1)
    parser.add_argument('-s', dest='config_id_start', type=int, default=1)

    args = parser.parse_args()

    main(args)
