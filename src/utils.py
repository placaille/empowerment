import matplotlib.pyplot as plt
plt.switch_backend('agg')

import seaborn as sns

def log_empowerment_map(map, env, writer, tag, global_step):
    fig = plt.figure()
    mask = env.grid != env.free
    # import pdb;pdb.set_trace()
    with sns.axes_style("white"):
        ax = sns.heatmap(map, mask=mask, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.savefig('{}.jpg'.format(tag))

    # log to tensorboardX
    writer.add_figure(tag, fig, global_step)

def log_loss(loss, writer, tag, global_step):

    # log to tensorboardX
    writer.add_scalar(tag, loss, global_step)
