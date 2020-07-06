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

from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from pandas._libs.tslibs.timestamps import Timestamp
from re import match
import numpy as np


def sparkline(series, width=8, plottype="bar"):
    """Generate a basic sparkline graph, consisting of `width` bars, of an
    iterable of numeric data. Each bar represents the mean of that fraction of
    the data. Allowed `plottype`s are "bar" and "shade"."""

    plottypes = {
        "bar":   "▁▂▃▄▅▆▇█",
        "shade": "░▒▓█"
    }

    try:
        chars = plottypes[plottype]
    except KeyError:
        raise KeyError("Allowed plot types: " + ", ".join(plottypes.keys()))

    # Convert to proper type
    series = np.array(series)
    if not is_numeric_dtype(series):
        return " " * width

    # If we have fewer rows than bars, just use a bar for each row
    width = min(width, series.shape[0])

    # chunk into {width} chunks
    chunk_indices = np.linspace(0, series.shape[0], width+1)
    chunks = map(lambda tup: np.nanmean(series[int(tup[0]):int(tup[1])]),
                 zip(chunk_indices, chunk_indices[1:]))

    smin = np.nanmin(series)
    smax = np.nanmax(series)
    graph = ""

    for i in chunks:
        # Normalize to be between 0 and len(chars)
        level = (i - smin) / (smax - smin) * (len(chars) - 1)

        # If the data is missing, replace with a *nonbreaking* space character
        graph += " " if np.isnan(level) else chars[int(round(level))]

    return graph


def _markdowntable(*columns, caption=""):
    """Format list of objects as row in markdown table"""

    # List of widths of each column
    widths = [max(len(str(cell)) for cell in col) for col in columns]

    table = caption + "\n\n"

    for row in zip(*columns):
        row_str = " ".join(["| {:{w}}".format(str(cell), w=w)
                            for cell, w in zip(row, widths)]) + " |\n"

        # GitHub markdown calls for separators that are only dashes and pipes,
        # no plus signs: <https://github.github.com/gfm/#tables-extension->
        if match("^[| ]+$", row_str):
            row_str = row_str.replace(" ", "-")

        table += row_str

    return table


def _format(obj):
    """Format numbers and dates"""
    if type(obj) == str:
        return ""
    elif type(obj) == Timestamp:
        return obj.strftime("%b %_d, %Y")
    else:
        return "{:,}".format(obj)


def data_dictionary(self):
    """Generate a data dictionary for a given data frame in GitHub-flavored
    markdown, with a description column to be filled in by hand. To write out:

    with open("datadict.md", "w")  as outfile:
        print(df.data_dictionary(), file=outfile)"""

    missing_total = self.isna().sum().sum()
    s = "" if missing_total == 1 else "s"

    caption = f"Data frame, {self.shape[0]} rows with " +\
              f"{missing_total} missing value{s}:"

    mins = map(_format, self.min())
    maxs = map(_format, self.max())

    # The row of empty strings will be turned into a separator line
    return _markdowntable(
            ["Column",         ""] + list(map("`{}`".format, self.columns)),
            ["Type",           ""] + list(map("`{}`".format, self.dtypes)),
            ["Missing values", ""] + list(self.isna().sum()),
            ["Range",          ""] + list(map(" – ".join, zip(mins, maxs))),
            ["Distribution",   ""] + [sparkline(self[col]) for col in self],
            ["Description",    ""] + [""] * self.shape[1],
            caption=caption
            )


DataFrame.data_dictionary = data_dictionary
