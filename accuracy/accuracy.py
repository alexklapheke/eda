"""
File: accuracy.py
Author: Alex Klapheke
Email: alexklapheke@gmail.com
Github: https://github.com/alexklapheke
Description: Tools for evaluating the results of a fitted model

Copyright © 2020 Alex Klapheke

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


def accuracy_metrics(y_true, y_pred, f_score=False):
    """
    Takes a list of true outputs and model-predicted outputs, and
    returns a confusion matrix with classification metrics which is
    structured as follows:

    True pos.   False pos. │ PPV
    False neg.  True neg.  │ NPV
    ───────────────────────┼─────
    Sensitiv.   Specific.  │ Acc.

    Intuitively, each metric is derived solely from the row or column to
    which it is adjacent, and accuracy is derived from the whole table.
    """
    assert len(y_true) == len(y_pred), "Arrays must be the same length."

    # Cast arrays to use efficiently with numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate confusion matrix
    tp = np.sum((y_true == y_pred) & y_true)
    tn = np.sum((y_true == y_pred) & ~y_true)
    fp = np.sum((y_true != y_pred) & ~y_true)
    fn = np.sum((y_true != y_pred) & y_true)

    # Calculate accuracy measures
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    acc = (tp + tn) / (tp + fp + tn + fn)
    # f = 2 * ppv * sens / (ppv + sens)

    # Compile into data frame
    return DataFrame([
            [tp, fn, sens],
            [fp, tn, spec],
            [ppv, npv, acc]],
            columns=["Cond P", "Cond N", "PPV/NPV"],
            index=["Test P", "Test N", "Sens/Spec"])


def multiaccuracy(y_true, y_pred, normalize=False, totals=True):
    """Returns a pivot table showing the rates of predicted vs. true values.
       Use `multiaccuracy_heatmap` to generate a heatmap plot."""
    results = DataFrame({
        "true": y_true,
        "pred": y_pred,
        "one": 1
    }).pivot_table(index="true",
                   columns="pred",
                   values="one",
                   aggfunc="count",
                   margins=totals)
    return results / len(y_true) if normalize else results


def multiaccuracy_heatmap(y_true, y_pred, *args, **kwargs):
    """Returns a heatmap plot of the table returned by `multiaccuracy`.
    Example usage:

    multiaccuracy_heatmap(y, model.predict(X), cmap="viridis");"""
    from seaborn import heatmap

    return heatmap(multiaccuracy(y_true, y_pred,
                                 normalize=True,
                                 totals=False),
                   vmin=0, vmax=1, *args, **kwargs)


def fuzzy_accuracy(y_true, y_pred, tolerance):
    """Returns accuracy of a model trained on numeric data with a tolerance.
       For example, with a tolerance of 1, a model prediction of 9 for a true
       value of 10 will be counted in the "fuzzy accuracy"."""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred) <= tolerance)


def cohens_kappa(y_pred1, y_pred2):
    """Calculates Cohen's kappa."""
    assert len(y_pred1) == len(y_pred2), "Arrays must be the same length."

    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)

    p_a = np.mean(y_pred1 == y_pred2)
    p_e = np.sum([np.mean(y_pred1 == category) *
                  np.mean(y_pred2 == category)
                  for category in set(y_pred1) | set(y_pred2)])

    return (p_a - p_e) / (1 - p_e)


def _norm(m, s, x):
    """Defines the normal distribution"""
    return np.exp(-(x - m)**2 / (2 * s**2)) / (s * (2 * np.pi)**(1/2))


def test_LINE(y_true, y_pred):
    """Tests LINE assumptions for linear regression. Example usage:

        import numpy as np
        from sklearn.linear_model import LinearRegression

        # Generate data
        x = np.arange(100)
        np.random.shuffle(x)
        x = x + np.random.normal(size=100)
        y = x + np.random.normal(size=100) * 20
        X = x.reshape(-1, 1)

        # Fit data
        lr = LinearRegression().fit(X, y)

        # Show assumptions
        test_LINE(y, lr.predict(X))
    """

    # Cast arrays to use efficiently with numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Set up graphs
    fig = plt.figure(figsize=(12, 8))

    # Define our residuals
    resids = y_true - y_pred
    student = resids / resids.std()

    # Define outliers as studentized residuals e_i s.t. |e_i| >= 3
    outlier = np.abs(student) >= 3
    outlier_cmap = list(map(lambda x: "C" + str(int(x)), outlier))

    # Set up bins for graphing normal distribution
    bins = np.arange(
        start=min(resids),
        stop=max(resids),
        step=(max(resids) - min(resids)) / 100
    )

    # First test plot
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("Is the data [L]inear?")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.scatter(y_true, y_pred, marker=".")
    ax.plot((min(y_true), max(y_true)),
            (min(y_pred), max(y_pred)), color="red")

    # Second test plot
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("Is the error [I]ndependent along the x-axis?")
    ax.set_xlabel("Series")
    ax.set_ylabel("Studentized residuals")
    ax.scatter(range(len(student)), student, marker=".", c=outlier_cmap)
    ax.axhline(0, color="gray")

    # Third test plot
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("Is the error [N]ormal?")
    ax.set_xlabel("Residuals")
    ax.hist(resids, density=True)
    ax.plot(bins, _norm(np.mean(resids), np.std(resids), bins), color="red")

    # Fourth test plot
    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("Is the variance of error [E]qual everywhere?")
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Studentized residuals")
    ax.scatter(y_pred, student, marker=".", c=outlier_cmap)
    ax.axhline(0, color="gray")

    plt.tight_layout()
