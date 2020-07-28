
import logging
import numpy as np

from sysnet.sources import LinearRegression

__all__ = ['FeatureElimination']


class FeatureElimination:

    logger = logging.getLogger('FeatureElimination')

    def __init__(self, model, datasets):
        self.model = model
        self.datasets = datasets
        self.results = {
            'val_loss': [],
            'importance': [],
            'axes': []
        }

    def run(self, axes):

        # compute with all axes once
        val_loss_baseline = self.train_eval(axes)
        self.logger.info(f"all attributes with {val_loss_baseline:.6f}")
        self.results['val_loss_baseline'] = val_loss_baseline
        self.__run(axes.copy())
        self.__sort()

    def __run(self, axes):
        ''' Recursively run `train_eval`

        '''
        val_loss = []

        for index in axes:
            axes_wo_index = axes.copy()
            axes_wo_index.remove(index)
            val_loss_wo_index = self.train_eval(axes_wo_index)
            val_loss.append(val_loss_wo_index)

        arg = np.argmin(val_loss)

        self.logger.info(
            f'attribute index-{axes[arg]} with {val_loss[arg]:.6f}')
        self.results['val_loss'].append(val_loss)
        self.results['axes'].append(axes.copy())
        self.results['importance'].append(axes.pop(arg))

        if len(axes) == 1:
            # this means that we are left with one map
            # append this map to the importance, and return
            self.results['importance'].append(axes[0])
            self.results['val_loss'].append([max(val_loss)])
            self.results['axes'].append([axes[0]])
            return self.results

        self.__run(axes)

    def train_eval(self, axes):
        # print(axes)
        self.model.fit(self.datasets['train'].x[:, axes],
                       self.datasets['train'].y)
        return self.model.evaluate(self.datasets['valid'].x[:, axes],
                                   self.datasets['valid'].y)

    def __sort(self):
        num_iterations = len(self.results['axes'])
        val_loss_dict = {}
        for i in range(num_iterations):
            for j in range(len(self.results['val_loss'][i])):

                key = '%d-%d' % (i, self.results['axes'][i][j])
                val_loss_dict[key] = self.results['val_loss'][i][j]

        val_loss = np.ones((num_iterations, num_iterations))*np.nan
        for i in range(num_iterations):
            for j, index in enumerate(self.results['importance']):
                key = '%d-%d' % (i, index)
                if key in val_loss_dict:
                    val_loss[i, j] = val_loss_dict[key]
        self.results['val_loss_sorted'] = val_loss

        arg_val_loss_min = np.diagonal(val_loss).argmin()
        axes_to_keep = self.results['importance'][arg_val_loss_min+1:]
        # axes_to_keep.sort()
        self.results['axes_to_keep'] = axes_to_keep

    def plot(self, ax=None, labels=None, **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if ax is None:
            fig, ax = plt.subplots()

        if labels is None:
            labels = ['sys-%d' %
                      j for j in range(len(self.results['importance']))]

        xlabels = [labels[j] for j in self.results['importance']]

        y_ = self.results['val_loss_sorted']**0.5 / \
            self.results['val_loss_baseline']**0.5

        sns.heatmap(y_,
                    xticklabels=xlabels,
                    square=True,
                    # center=0,
                    ax=ax,
                    linewidths=0.5,
                    cbar_kws={
                        "shrink": .5,
                        "label": r'$r_{\rm RMSE}$',
                        "extend": "max"},
                    **kwargs
                    )
        ax.set_xticklabels(xlabels, rotation=90)
        ax.set_yticks([])
        ax.xaxis.tick_top()

        return ax
