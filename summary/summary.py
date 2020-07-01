"""
File: summary.py
Author: Alex Klapheke
Email: alexklapheke@gmail.com
Github: https://github.com/alexklapheke
Description: Tools for basic exploratory data analysis

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

from pandas import DataFrame, concat
from numpy import logical_or
import operator
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def summary(self):
    """Describe the columns of a data frame with
    excerpted values, types, and summary statistics."""
    return concat([
            self.head(5).T,
            DataFrame({"type": self.dtypes}),
            DataFrame({"pct_missing": self.isna().mean() * 100}),
            self.describe(percentiles=[0.5]).drop("count").T
        ], axis=1)


def summary_by(self, col):
    """Describe the columns of a data frame, grouped by column
    `col` with excerpted values, types, and summary statistics."""
    dfg = self.groupby(col)
    return concat([
            dfg.first().T.add_prefix("head_"),
            dfg.dtypes.T.add_prefix("type_"),
            dfg.apply(lambda x: x.isna().mean() * 100)
               .T.add_prefix("pct_missing_")
        ], axis=1)


def missing(self, *args, **kwargs):
    """Display bar graph of missing data by column."""
    return self.\
        isna().\
        mean().\
        apply(lambda x: x * 100).\
        iloc[::-1].\
        plot.\
        barh(
            title="Percent data missing by column",
            xlim=(0, 100),
            color="red",
            *args,
            **kwargs
            )


def missing_by(self, col, *args, **kwargs):
    """Display bar graph of missing data by column, grouped by column `col`."""
    return self.\
        drop(col, axis=1).\
        isna().\
        join(self[col]).\
        groupby(col).\
        mean().\
        apply(lambda x: x * 100).\
        iloc[:, ::-1].\
        transpose().\
        plot.\
        barh(
           title="Percent data missing by column",
           xlim=(0, 100),
           *args,
           **kwargs
           )


def missing_map(self, figsize=(8, 5), *args, **kwargs):
    """Display heatmap of missing data to uncover patterns. Note that the
    columns of the dataframe are shown on the y-axis."""
    cmap = LinearSegmentedColormap.from_list("cmap", ["#00000000", "red"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(self.T.isna(), aspect="auto", interpolation="none", cmap=cmap)
    ax.set_title("Missing data")
    ax.set_yticks(range(len(self.columns)))
    ax.set_yticklabels(self.columns)

    return ax


def misordered(self, cols, ascending=False, allow_equal=True):
    """Find rows of a dataframe in which the data are in the wrong order. For
       example, in a data frame `df` that looks like:

       start      │ middle     │ end
       ───────────┼────────────┼───────────
       2001-01-01 │ 2000-06-01 │ 2001-12-31
       2002-04-01 │ 2002-08-01 │ 2002-10-01
       2002-01-01 │ 2002-06-01 │ 2001-10-31

       Running `df.misordered(["start", "middle", "end"]) will print the first
       row, in which the middle date precedes the start date, and the third
       row, in which the end date precedes the start date. The following
       Boolean arguments are also accepted:

       `ascending`: Specifiy whether the *proper* order of columns is ascending
       `allow_equal`: Allow values in adjacent columns to be equal"""

    if ascending:
        op = operator.lt if allow_equal else operator.le
    else:
        op = operator.gt if allow_equal else operator.ge

    mask = [op(self[a], self[b]) for a, b in zip(cols, cols[1:])]
    indices = self[logical_or.reduce(mask)].index
    return self.loc[indices, cols]


DataFrame.summary = summary
DataFrame.summary_by = summary_by
DataFrame.missing = missing
DataFrame.missing_by = missing_by
DataFrame.missing_map = missing_map
DataFrame.misordered = misordered
