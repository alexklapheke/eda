"""
File: report.py
Author: Alex Klapheke
Email: alexklapheke@gmail.com
Github: https://github.com/alexklapheke
Description: Tools for creating data reports

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

from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from re import match
import numpy as np


def sparkline(series, width=10, plottype="bar", hist=False):
    """Generate a basic sparkline graph, consisting of `width` bars,
    of an iterable of numeric data. Each bar represents the mean of
    that fraction of the data. Allowed `plottype`s are "bar", "line",
    and "shade". If `hist` is true, plot a histogram of the data in
    `width` bins."""

    np.seterr('raise')

    plottypes = {
        "bar":   "▁▂▃▄▅▆▇█",
        "line":   "⎽⎼─⎻⎺",
        "shade": "░▒▓█"
    }

    try:
        chars = plottypes[plottype]
    except KeyError:
        print("Allowed plot types: " + ", ".join(plottypes.keys()))

    # Convert to proper type
    if is_datetime64_any_dtype(series):
        series = np.array([np.nan if np.isnat(x) else int(x)
                           for x in np.array(series, dtype=np.datetime64)])
    else:
        series = np.array(series)

    # If not numeric, there's nothing to plot
    if not is_numeric_dtype(series):
        return " " * width

    smin = np.nanmin(series)
    smax = np.nanmax(series)
    graph = ""

    if hist:

        # Drop NaNs, as they will not count anyway
        series = series[~np.isnan(series)]

        # If we have fewer levels than bins, just use a bin for each level
        width = min(width, len(np.unique(series)))

        # Since we do strict less-than below, if the highest bin edge
        # were the max of the series, the max of the seires would not
        # be counted. So we make the highest bin edge infinity.
        bins = np.linspace(smin, smax, width, endpoint=False)
        bins = np.append(bins, np.inf)

        # Divide into bins
        levels = [np.sum((series >= bmin) & (series < bmax))
                  for bmin, bmax in zip(bins, bins[1:])]

        for level in levels:
            level = level / max(levels) * (len(chars) - 1)
            graph += chars[int(round(level))]

    else:

        # If we have fewer rows than bars, just use a bar for each row
        width = min(width, series.shape[0])

        # chunk into {width} chunks
        chunk_indices = np.linspace(0, series.shape[0], width+1)
        chunks = map(lambda tup: np.nanmean(series[int(tup[0]):int(tup[1])]),
                     zip(chunk_indices, chunk_indices[1:]))

        for i in chunks:
            # Normalize to be between 0 and len(chars)
            try:
                level = (i - smin) / (smax - smin) * (len(chars) - 1)
            except FloatingPointError:
                level = 0

            # If the data is missing, replace with a *nonbreaking* space
            graph += "\u00A0" if np.isnan(level) else chars[int(round(level))]

    return graph


def _sparkline_series(self, *args, **kwargs):
    """Generate a basic sparkline graph, consisting of `width` bars, of an
    iterable of numeric data. Each bar represents the mean of that fraction of
    the data. Allowed `plottype`s are "bar" and "shade". If `hist` is true,
    plot a histogram of the data in `width` bins."""
    return sparkline(self, *args, **kwargs)


def _sparkline_dataframe(self, col, *args, **kwargs):
    """Generate a basic sparkline graph, consisting of `width` bars, of an
    iterable of numeric data. Each bar represents the mean of that fraction of
    the data. Allowed `plottype`s are "bar" and "shade". If `hist` is true,
    plot a histogram of the data in `width` bins."""
    return sparkline(self[col], *args, **kwargs)


def _markdowntable(*columns, caption=""):
    """Format list of objects as row in markdown table"""

    # List of widths of each column
    widths = [max(len(str(cell)) for cell in col) for col in columns]

    table = caption + "\n\n"

    for row in zip(*columns):
        # Format row elements, separated by pipes
        row_str = " ".join(["| {:{w}}".format(str(cell), w=w)
                            for cell, w in zip(row, widths)]) + " |\n"

        # Turn empty rows into separator rows. GitHub markdown calls
        # for separators that are only dashes and pipes, no plus
        # signs: <https://github.github.com/gfm/#tables-extension->
        if match("^[| ]+$", row_str):
            row_str = row_str.replace(" ", "-")

        table += row_str

    return table


def _data_range(df):
    _range = []

    for colname in df.columns:
        col = df[colname]

        if is_datetime64_any_dtype(col):
            _range.append(col.min().strftime("%b %_d, %Y") + " – " +
                          col.max().strftime("%b %_d, %Y"))
        elif is_numeric_dtype(col):
            _range.append("{:n}".format(col.min()) + " – " +
                          "{:n}".format(col.max()))
        else:
            _range.append("")

    return _range


def data_dictionary(self):
    """Generate a data dictionary for a given data frame in GitHub-flavored
    markdown, with a description column to be filled in by hand. To write out:

    with open("datadict.md", "w")  as outfile:
        print(df.data_dictionary(), file=outfile)"""

    missing = self.isna().sum()
    missing_total = missing.sum()
    s = "" if missing_total == 1 else "s"

    missing_zip = zip(missing, missing / self.shape[0])

    caption = f"Data frame, {self.shape[0]:,} rows with " +\
              f"{missing_total:,} missing value{s}:"

    # The row of empty strings will be turned into a separator line
    return _markdowntable(
            ["Column",         ""] + list(map("`{}`".format, self.columns)),
            ["Type",           ""] + list(map("`{}`".format, self.dtypes)),
            ["Missing values", ""] + ["{:,} ({:.0%})".format(no, pct)
                                      for no, pct in missing_zip],
            ["Range",          ""] + _data_range(self),
            ["Distribution",   ""] + [self.sparkline(col, hist=True)
                                      for col in self],
            ["Description",    ""] + [""] * self.shape[1],
            caption=caption
            )


DataFrame.data_dictionary = data_dictionary
Series.sparkline = _sparkline_series
DataFrame.sparkline = _sparkline_dataframe
