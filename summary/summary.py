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

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from eda.report import sparkline


def _data_range(col):
    if col.dtype == "O":
        return ""

    col_min = str(_safe_agg(col, "min"))
    col_max = str(_safe_agg(col, "max"))

    return col_min + " – " + col_max if col_min and col_max else ""


def _safe_agg(col, fun):
    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

    if fun == "benford":
        return sparkline(benford(col), width=9)

    try:
        out = col.agg(fun)
    except (TypeError, ValueError):
        out = ""

    # Some functions implicitly convert between datetime types ¯\_(ツ)_/¯
    if is_datetime64_any_dtype(col) or is_datetime64_any_dtype(out):
        try:
            return out.strftime("%b %_d, %Y")
        except AttributeError:
            return out
    elif is_numeric_dtype(out):
        return "{:n}".format(out)
    else:
        return out


def summary(self, **kwargs):
    """Describe the columns of a data frame with
    excerpted values, types, and summary statistics."""
    missing = self.isna().sum()

    missing_zip = zip(missing, missing / self.shape[0])

    df = DataFrame({
        "Type": self.dtypes,
        "Missing values": ["{:,} ({:.0%})".format(no, pct)
                           for no, pct in missing_zip],
        "Range": [_data_range(self[col]) for col in self],
        "Distribution": [self.sparkline(col, hist=True) for col in self],
        **{title: [_safe_agg(self[col], stat) for col in self]
           for title, stat in kwargs.items()}
    }, index=self.columns)

    df.index.name = "Column"
    return df


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
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cmap", ["#00000000", "red"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(self.T.isna(), cmap=cmap)
    ax.set_title("Missing data")
    ax.set_ylabel("Column")
    ax.set_xlabel("Row position")
    ax.set_xticks(range(self.shape[0]))
    ax.set_yticks(range(len(self.columns)))
    ax.set_yticklabels(self.columns)

    return ax


def misordered(self, *cols, ascending=True, allow_equal=True):
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
    import operator

    if ascending:
        op = operator.gt if allow_equal else operator.ge
    else:
        op = operator.lt if allow_equal else operator.le

    mask = [op(self[a], self[b]) for a, b in zip(cols, cols[1:])]
    indices = self[np.logical_or.reduce(mask)].index
    return self.loc[indices, cols]


def _get_first_digit(x):
    if x == 0:
        return 0
    else:
        return int(str(x).lstrip('0.-')[0])


def benford(iterable):
    try:
        iterable = list(map(int, iterable))
    except (TypeError, ValueError):
        return []

    digits = np.arange(1, 10)
    return [(np.array(list(map(_get_first_digit, iterable))) == n).mean()
            for n in digits]


def benford_plot(iterable, ax=None, *args, **kwargs):
    digits = np.arange(1, 10)

    predicted = [np.log10(1 + 1/n) for n in digits]
    actual = benford(iterable)

    if not ax:
        ax = plt.gca()

    ax.bar(digits - 1/6, predicted,
           width=1/3,
           align="center",
           color="C0",
           label="Predicted",
           *args, **kwargs
           )
    ax.bar(digits + 1/6, actual,
           width=1/3,
           align="center",
           color="C1",
           label="Actual",
           *args, **kwargs
           )
    ax.set_xticks(digits)
    ax.legend()

    return ax


DataFrame.summary = summary
DataFrame.missing = missing
DataFrame.missing_by = missing_by
DataFrame.missing_map = missing_map
DataFrame.misordered = misordered
