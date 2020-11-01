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
import numpy as np


def sparkline(series, width=10, plottype="bar", hist=False):
    """Generate a basic sparkline graph, consisting of `width` bars,
    of an iterable of numeric data. Each bar represents the mean of
    that fraction of the data. Allowed `plottype`s are "bar", "line",
    and "shade". If `hist` is true, plot a histogram of the data in
    `width` bins."""

    if len(series) == 0:
        return ""

    np.seterr('raise')

    plottypes = {
        "bar":   "▁▂▃▄▅▆▇█",
        "line":   "⎽⎼─⎻⎺",
        "shade": "░▒▓█"
    }
    missing = "\u00A0"  # Character to use when data is missing

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

            graph += missing if np.isnan(level) else chars[int(round(level))]

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


def _to_markdown(df):
    """Format list of objects as row in markdown table"""
    df = df.astype(str)

    # List of character widths of each column
    widths = [max(len(col), *(len(cell) for cell in df[col]))
              for col in df.columns]

    # Header row
    table = " ".join(["| {:{w}}".format(str(cell), w=w)
                      for cell, w in zip(df.columns, widths)]) + " |\n"

    # Separator row. GitHub markdown calls for separators
    # that are only dashes and pipes, no plus signs:
    # <https://github.github.com/gfm/#tables-extension->
    table += "|" + "|".join(["-"*(w+2) for w in widths]) + "|\n"

    # Body rows
    table += "\n".join([" ".join(["| {:{w}}".format(str(cell), w=w)
                        for cell, w in zip(row, widths)]) + " |"
                        for _, row in df.iterrows()])

    return table


def data_dictionary(self, **kwargs):
    """Generate a data dictionary for a given data frame in GitHub-flavored
    markdown, with a description column to be filled in by hand. To write out:

        with open("datadict.md", "w")  as outfile:
            print(df.data_dictionary(), file=outfile)

    You can add any aggregation columns with the syntax Title=Function. Any
    function that can be passed to pandas.DataFrame.agg can be passed here.
    Passing False will add a blank column. For example:

        df.data_dictionary(Mean=np.mean, Median="median", Description=False)

    will create mean and median columns, and a blank "Description" column to be
    filled in manually by the user."""
    summary = self.summary(**kwargs).reset_index()

    missing_total = self.isna().sum().sum()
    s = "" if missing_total == 1 else "s"
    caption = f"Data frame, {self.shape[0]:,} rows with " +\
              f"{missing_total:,} missing value{s}:"

    # The row of empty strings will be turned into a separator line
    return caption + "\n\n" + _to_markdown(summary)


DataFrame.data_dictionary = data_dictionary
Series.sparkline = _sparkline_series
DataFrame.sparkline = _sparkline_dataframe
