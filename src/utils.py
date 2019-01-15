import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import seaborn as sns

def log_empowerment_map(map, env, writer, tag, global_step=0, save_dir=None):
    fig = plt.figure()
    file_name = '{}.jpg'.format(tag)

    if save_dir is not None:
        file_name = os.path.join(save_dir, file_name)

    mask = env.grid != env.free
    with sns.axes_style("white"):
        ax = sns.heatmap(map, mask=mask, cmap='viridis', xticklabels=False, yticklabels=False)

    plt.savefig(file_name)

    # log to tensorboardX
    writer.add_figure(tag, fig, global_step)

def log_loss(loss, writer, tag, global_step):

    # log to tensorboardX
    writer.add_scalar(tag, loss, global_step)
