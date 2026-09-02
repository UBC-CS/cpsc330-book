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

def summarize_cross_validation(model, X, y, **kwargs):
    """Run cross-validation and summarize each returned quantity.

    Parameters
    ----------
    model : estimator
        A scikit-learn predictive estimator or pipeline.
    X : array-like
        Feature data.
    y : array-like
        Target data.
    **kwargs
        Additional arguments passed to `sklearn.model_selection.cross_validate`.

    Returns
    ----------
    pandas.Series
        A numeric Series with one entry for each (measure, statistic)
        pair. Measures are ordered with validation and training scores
        first, followed by fitting and scoring times.
    """

    results = pd.DataFrame(cross_validate(model, X, y, **kwargs))
    summary = (
        results.agg(["mean", "std"])
        .T.rename(index={"test_score": "validation_score"})
        .rename_axis("measure")
    )

    preferred_order = [
        "validation_score",
        "train_score",
        "fit_time",
        "score_time",
    ]
    ordered_measures = [name for name in preferred_order if name in summary.index]
    ordered_measures.extend(name for name in summary.index if name not in ordered_measures)

    return summary.loc[ordered_measures].stack().rename_axis(["measure", "statistic"])
