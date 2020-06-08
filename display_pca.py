import matplotlib.pyplot as plt
import matplotlib.collections as LineCollection
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


def display_corr_matrix(corr):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # Generate a custom diverging colormap
    cmap = sns.color_palette("GnBu_d")
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, mask=mask,
                cmap=cmap, square=True, center=0, linewidths=.5, cbar_kws={"shrink": .7})
    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.title('Correlation Matrix', weight='bold')
    plt.show()


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            fig, ax = plt.subplots(figsize=(8, 8))
            plt.title("Correlation Circle (F{} et F{})".format(d1 + 1, d2 + 1))

            # limits
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1*1.05, 1*1.05, -1*1.05, 1*1.05
            else:
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])
                plt.title("- Zoom - Correlation Circle (F{} et F{})".format(d1 + 1, d2 + 1))

            # arrow display
            # if more than 30 arrows, do not display end of the arrow
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                           pcs[d1, :], pcs[d2, :],
                           angles='xy', width=0.003, scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            # variable display
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x, y, labels[i], fontsize='8', ha='center', va='center', rotation=label_rotation,
                                 color="black")

            # circle display
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='black')
            plt.gca().add_artist(circle)

            # define limits
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # display middle lines
            plt.plot([-1, 1], [0, 0], color='silver', ls='-', linewidth=1)
            plt.plot([0, 0], [-1, 1], color='silver', ls='-', linewidth=1)

            # axes names, with explained variance %
            plt.xlabel('F{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

            plt.show(block=False)


def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_ * 100
    n_comp = len(scree)

    bs = 1 / np.arange(n_comp, 0, -1)
    bs = np.cumsum(bs)
    bs = bs[::-1]

    plt.bar(np.arange(n_comp)+1, scree)
    plt.plot(np.arange(n_comp)+1, scree.cumsum(),c="black",marker='o')
    plt.xlabel("Factor Number")
    plt.ylabel("Eigenvalue")
    plt.title("Scree plot - Explained variance vs # of factors")
    plt.xticks(np.arange(n_comp)+1)
    plt.show()

def display_broken_sticks_plot(pca, n):
    n_comp = len(pca.explained_variance_)

    bs = 1 / np.arange(n_comp, 0, -1)
    bs = np.cumsum(bs)
    bs = bs[::-1]

    eigval = (n - 1) / n * pca.explained_variance_

    plt.bar(np.arange(n_comp) + 1, eigval)
    plt.plot(np.arange(n_comp) + 1, bs, c='red', marker='o', label='treshold')

    plt.xticks(np.arange(n_comp)+1)
    plt.xlabel("Factor Number")
    plt.ylabel("Eigenval")
    plt.title("Broken Stick Model")
    plt.legend()
    plt.show()

def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()


def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None,
                             lims=None):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            ax1.set_xlim([-boundary, boundary])
            ax1.set_ylim([-boundary, boundary])

            if lims is None:
                ax2.set_xlim([np.min(X_projected[:, [d1]]) * 1.1, np.max(X_projected[:, [d1]] * 1.2)])
                ax2.set_ylim([np.min(X_projected[:, [d2]]) * 1.1, np.max(X_projected[:, [d2]] * 1.2)])
            else:
                xmin, xmax, ymin, ymax = lims
                ax2.set_xlim(xmin, xmax)
                ax2.set_ylim(ymin, ymax)

            for axes in (ax1, ax2):
                for i in range(len(X_projected)):
                    axes.annotate(labels[i], (X_projected[i, d1], X_projected[i, d2]))

                axes.plot([-20, 20], [0, 0], color='silver', linestyle='-', linewidth=1)
                axes.plot([0, 0], [-20, 20], color='silver', linestyle='-', linewidth=1)

                axes.set_xlabel('F{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
                axes.set_ylabel('F{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

                ax1.set_title("Projection (on F{} and F{})".format(d1 + 1, d2 + 1))
                ax2.set_title("- Zoom - Projection (on F{} and F{})".format(d1 + 1, d2 + 1))


def display_factorial_planes \
                (X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None, lims=None):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            sns.set_style("darkgrid")
            plt.figure(figsize=(12, 12))
            x_axis = X_projected[:, d1]
            y_axis = X_projected[:, d2]

            # display points
            if illustrative_var is None:
                sns.scatterplot(x_axis, y_axis, alpha=alpha)
            else:
                sns.scatterplot(x_axis, y_axis, hue=illustrative_var, alpha=alpha, palette='Set1')

                # display centroid only if there is no labels to display
                if labels is None:
                    centroids_all = []
                    for cluster_name in sorted(illustrative_var.unique()):
                        index_proj = illustrative_var.loc[illustrative_var == cluster_name].index
                        x_centroid, y_centroid = (np.mean(x_axis[index_proj]), np.mean(y_axis[index_proj]))
                        centroids_all.append([x_centroid, y_centroid])

                    plt.scatter([row[0] for row in centroids_all], [row[1] for row in centroids_all], s=85, alpha=0.5, marker='o', c='black', label='Centroids')

                    for i, (x, y) in enumerate(centroids_all):
                        plt.text(x, y + 0.05, sorted(illustrative_var.unique())[i], fontsize='10', weight='bold',
                                 ha='center', va='bottom')

            # display labels of the points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y + 0.05, labels[i], fontsize='8', ha='center', va='bottom')

            # graph limits
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            if lims is None:
                boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
                plt.xlim([-boundary, boundary])
                plt.ylim([-boundary, boundary])
            else:
                xmin, xmax, ymin, ymax = lims
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)

            # display middle lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # axes names with value
            plt.xlabel('F{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))

            plt.legend()
            plt.title("Projection (on F{} and F{})".format(d1 + 1, d2 + 1))

            plt.show(block=False)
