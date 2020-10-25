EDA is a small miscellany of tools to facilitate exploratory data analysis and other common data science tasks. It is released under the MIT license.

# Installation

To install, place in a directory such as `~/bin/python`, then add the following line to your `~/.bashrc`:

```bash
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
* `df.missing_map()`: Show a heatmap of missing data to uncover patterns.
* `df.misordered(col1, col2, ...)`: Show rows which are in the wrong order; e.g., `df.misordered("start", "end")` will show rows in which the end date precedes the start date.
* `benford(iterable)`: Given an iterable of numerics, give the proportion of first digits to check conformity to [Benford's Law](https://en.wikipedia.org/wiki/Benford%27s_law). You can feed the input into `sparkline` (be sure to set `width=9`). The results should look like `█▅▃▂▂▂▁▁▁`.
* `benford_plot(iterable)`: Show a bar plot comparing the proportion of first digits to those predicted by Benford's Law.

## `accuracy` module

This module contains standalone functions for evaluating models.

* `accuracy_metrics(y_true, y_pred)`: Returns a labeled confusion matrix, along with measures such as sensitivity and specificity.
* `multiaccuracy(y_true, y_pred)`: Given the results of a multiclass classification, show a pivot table of class predictions.
* `multiaccuracy_heatmap(y_true, y_pred)`: Like `multiaccuracy`, but show a heatmap.
* `fuzzy_accuracy(y_true, y_pred, tolerance)`: For a multiclass classification of ordinal data, show the percent of results that were within `tolerance` of the true class.
* `cohens_kappa(y_pred1, y_pred2)`: Given the results of two models, calculate the degree to which they agree using [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa), from 0 (no agreement) to 1 (perfect agreement).
* `test_LINE(y_true, y_pred)`: Show some plots to help test the ["LINE" assumptions](http://people.duke.edu/~rnau/testing.htm) of a linear regression.

## `model` module

This module is for machine learning.

* `DBSCAN()`: an implementation of the [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) clustering algorithm, that doesn't require the high [memory overhead](https://stackoverflow.com/questions/16381577/scikit-learn-dbscan-memory-usage) of scikit-learn's implementation (sklearn computes a distance matrix which is O(n²) in space in the number of data points and can easily use several GB of memory). Uses sklearn's `.fit()`/`.predict()` convention and cen be used in [pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline).

## `nlp` module

This module provides some convenience functions for dealing with natural language.

* `tf_idf`: Compute [tf-idf](https://en.wikipedia.org/wiki/Tf-idf) score.
* `logodds_dirichlet`: Compute [log-odds ratio, uninformative Dirichlet prior](https://languagelog.ldc.upenn.edu/nll/?p=10073).

## `report` module

Like `summary`, this module adds methods to Pandas data frames.

* `sparkline(iterable)`: Produce a [sparkline](https://www.edwardtufte.com/bboard/q-and-a-fetch-msg?msg_id=0001OR&topic_id=1) given an iterable of numerics or date/time objects. For example, `sparkline(range(8))` produces `▁▂▃▄▅▆▇█`.
* `series.sparkline()`, `df.sparkline(col)`: Produce a sparkline of the given series or column of the data frame.
* `df.data_dictionary()`: Return a data dictionary in GitHub-flavored markdown, suitable for inclusion in a GitHub README. In an iPython environment, such as a Jupyter notebook, you can pretty-print this:

	```python
	from IPython.display import Markdown
	display(Markdown(df.data_dictionary()))
	```

	The output will look like the following:

	| Column         | Type      | Missing values | Range     | Distribution |
	|----------------|-----------|----------------|-----------|--------------|
	| `Sepal length` | `float64` | 0 (0%)         | 4.3 – 7.9 | ▃▇▅█▅█▆▃▂▃   |
	| `Sepal width`  | `float64` | 0 (0%)         | 2 – 4.4   | ▂▂▅▆█▇▃▃▁▁   |
	| `Petal length` | `float64` | 0 (0%)         | 1 – 6.9   | █▃▁▂▃▆▆▄▃▂   |
	| `Petal width`  | `float64` | 0 (0%)         | 0.1 – 2.5 | █▂▁▂▂▇▂▅▃▃   |
	| `Species`      | `object`  | 0 (0%)         |           |              |

