import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, plot_tree

import glob

# visualization
import graphviz
import matplotlib.pyplot as plt
from imageio import imread

plt.rcParams["font.size"] = 16

# Custom function to customize the tree plot and hide values and samples
def custom_plot_tree(
    tree_model,
    feature_names=None,
    class_names=None,
    **kwargs,
):
    """Plot a decision tree without displaying sample counts."""

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        **kwargs,
    )

    for text in plt.gca().texts:
        node_text = re.sub(r"samples = .*\n", "", text.get_text())
        text.set_text(node_text)

    plt.show()

def cross_validate_std(*args, **kwargs):
    """Like cross_validate, except also gives the standard deviation of the score"""
    res = pd.DataFrame(cross_validate(*args, **kwargs))
    res_mean = res.mean()

    res_mean["std_test_score"] = res["test_score"].std()
    if "train_score" in res:
        res_mean["std_train_score"] = res["train_score"].std()
    return res_mean

def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)