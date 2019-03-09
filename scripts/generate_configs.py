import os
import itertools

def main():
    env_name='CrossRoom-v0'
    num_steps=2
    max_iter=150000
    iter_per_eval=1000
    force_cpu='true'
    train_score='false'
    train_source_distr='true'
    pre_trained_dir='projects/augusta/pre_trained_models/models/fgan/repro/2_steps/js'
    diverg_name='js'
    comment='hp search for training generator on optim and gumbel temp'

    batch_size=[64, 128, 256]
    gumbel_temp_start=[0.1, 0.2, 0.5, 2, 5]
    learning_rate=[0.00001, 0.0001, 0.001, 0.01]
    momentum=[0.0, 0.9]
    optim_name=['sgd', 'adam', 'rmsprop']

    iters = [
        batch_size,
        gumbel_temp_start,
        learning_rate,
        momentum,
        optim_name,
    ]

    for (b, t, l, m, o) in itertools.product(*iters):
        if o == 'adam' and m != 0.0:
            continue
        string = '{}_b-{}_t-{}_l-{}_m-{}_o'.format(b, t, l, m, o)
        file_name = os.path.join('./.configs', '{}.conf'.format(string))
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
            f.write('comment: {}\n'.format(comment))

            f.write('batch-size: {}\n'.format(b))
            f.write('gumbel-temp-start: {}\n'.format(t))
            f.write('learning-rate: {}\n'.format(l))
            f.write('momentum: {}\n'.format(m))
            f.write('optim-name: {}\n'.format(o))

        print(string)

if __name__ == '__main__':
    main()
