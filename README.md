EDA is a small miscellany of tools for use with Pandas to facilitate exploratory data analysis. It is released under the MIT license.

# Installation

To install, place in a directory such as `~/bin/python`, then add the following line to your `~/.bashrc`:

``` {.bash}
export PYTHONPATH="${PYTHONPATH}:$HOME/bin/python"
```

You should then be able to `import eda` in Python applications.

# Use

## `summary` module

This module adds methods to Pandas data frames for exploring datasets. It is therefore only usable if you `import pandas`. It includes the following methods:

* `df.summary()`: Print a summary of the data frame `df`, including the first few rows, missing data, and summary statistics.
* `df.summary_by(col)`: Like `summary`, but grouped by column `col`.
* `df.missing()`: Show a bar plot of missing data by column.
* `df.missing_by(col)`: Like `missing`, but grouped by column `col`.
* `df.ordered([col1, col2, ...])`: Show rows which are in the wrong order; for example, `df.ordered(["start_date", "end_date"])` will show rows in which the start date precedes the end date.

## `accuracy` module

This module contains standalone functions for evaluating models.

* `accuracy_metrics(y_true, y_pred)`: Returns a labeled confusion matrix, along with measures such as sensitivity and specificity.
* `multiaccuracy(y_true, y_pred)`: Given the results of a multiclass classification, show a pivot table of class predictions.
* `multiaccuracy_heatmap(y_true, y_pred)`: Like `multiaccuracy`, but show a heatmap.
* `fuzzy_accuracy(y_true, y_pred, tolerance)`: For a multiclass classification of ordinal data, show the percent of results that were within `tolerance` of the true class.
* `cohens_kappa(y_pred1, y_pred2)`: Given the results of two models, calculate the degree to which they agree using [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa), from 0 (no agreement) to 1 (perfect agreement).
* `test_LINE(y_true, y_pred)`: Show some plots to help test the ["LINE" assumptions](http://people.duke.edu/~rnau/testing.htm) of a linear regression.
