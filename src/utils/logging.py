import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import seaborn as sns

def log_value_map(writer, map, mask, tag, global_step=0, file_name=None):
    fig = plt.figure()
    with sns.axes_style("white"):
        ax = sns.heatmap(map, mask=mask, cmap='viridis', xticklabels=False, yticklabels=False)

    if file_name is not None:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        plt.savefig(file_name)

    # log to tensorboardX
    writer.add_figure(tag, fig, global_step)
