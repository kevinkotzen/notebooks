from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


def make_confusion_matrix(cf=[], y_reference=[], y_predicted=[],
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=False,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          precision_recall=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    Modified from: https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    y_reference:   Ground truth Labels
    y_predicted:   Predicted Labels
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    def kappa():
        if len(y_predicted) > 0:
            return cohen_kappa_score(y_reference, y_predicted)
        else:
            return None

    def format_number(number):
        if number < 1000:
            return "{0:0.0f}".format(number)
        elif number < 1000000:
            return "{0:0.00f}K".format(number / 1000)
        else:
            return "{0:0.00f}M".format(number / 1000000)

    # Generate the confusion matrix if it does not already exist
    if len(cf) == 0:
        if len(y_reference) == 0 or len(y_predicted) == 0:
            raise ValueError(
                "You have to either specify a confusion matrix cf= or provide y_reference=y_reference and y_predicted=y_predicted.")
        try:
            cf = confusion_matrix(y_reference, y_predicted)
            cf_norm = confusion_matrix(y_reference, y_predicted, normalize='true')
        except:
            raise ValueError("Could not generate a confusion matrix from the y_reference and y_predicted provided.")
    else:
        cf_norm = cf/cf.sum(axis=1)

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = [f"{format_number(value)}\n" for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf_norm.flatten()]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    precision = np.diagonal(cf) / np.sum(cf, axis=0)
    recall = np.diagonal(cf) / np.sum(cf, axis=1)
    accuracy = np.trace(cf) / float(np.sum(cf))
    count = np.sum(cf)

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories,
                ax=ax, annot_kws={"fontsize":11})
    if xyplotlabels:
        ax.set_ylabel('Reference')
        ax.set_xlabel('Prediction')

    if title:
        fig.suptitle(title)

    if precision_recall:
        precision_ax = ax.secondary_xaxis('top')
        precision_ax.set_xticks([0.5, 1.5, 2.5, 3.5], minor=False)
        precision_labels = ["{0:.2%}".format(v) for v in np.around(precision, decimals=2)]
        precision_ax.set_xticklabels(precision_labels)
        precision_ax.set_xlabel('Precision')

        recall_ax = ax.secondary_yaxis('right')
        recall_ax.set_yticks([0.5, 1.5, 2.5, 3.5], minor=False)
        recall_labels = ["{0:.2}%".format(v) for v in np.around(recall, decimals=2)]
        recall_ax.set_yticklabels(recall_labels)
        recall_ax.set_ylabel('Recall')
        pass

    if sum_stats:
        stats_text = "Accuracy={:0.3f}, k={:0.3f}, Count={}".format(
            accuracy, kappa(), format_number(count))

        ax.set_title(stats_text)

    return  fig
if(__name__ == '__main__'):
    print('__main__')
    y_test = np.array([1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4])
    y_pred = np.array([1,2,4,3, 1,2,4,3, 1,2,3,4, 1,4,3,4, 1,2,4,4, 1,2,4,4, 1,2,4,4, 1,2,4,4, 1,2,3,3, 1,2,3,3, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,4,4, 1,2,4,4, 1,2,4,4, 1,2,4,4, 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4])
    cf_matrix_4x4 = np.array([ [23,  5, 12, 7],
                    [5,  3, 30, 7],
                    [6,6,40,5],
                    [1,0,34,77]])

    make_confusion_matrix(y_reference=y_test, y_predicted=y_pred, figsize=(8, 6), cbar=False, title=False)

