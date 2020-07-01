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
from re import match


def _markdowntable(*columns, caption=""):
    """Format list of objects as row in markdown table"""
    cols = []
    for i, col in enumerate(columns):
        width = max(map(lambda x: len(str(x)), col))
        cols.append(["| " + "{:{w}}".format(str(c), w=width) for c in col])

    table = caption + "\n\n"

    for row in zip(*cols):
        row_str = " ".join(map(str, row)) + " |\n"

        # GitHub markdown calls for separators that are only dashes and pipes,
        # no plus signs: <https://github.github.com/gfm/#tables-extension->
        if match("^[| ]+$", row_str):
            row_str = row_str.replace(" ", "-")

        table += row_str

    return table


def data_dictionary(self):
    """Generate a data dictionary for a given data frame in GitHub-flavored
    markdown, with a description column to be filled in by hand. It is
    recommended you name your data frame: df.name = "..."."""

    try:
        name = self.name
    except AttributeError:
        name = "unnamed data frame"

    columns = ["Column", ""]
    columns.extend(map("`{}`".format, self.columns))

    dtypes = ["Type", ""]
    dtypes.extend(map("`{}`".format, self.dtypes))

    shape = ["Shape", ""]
    shape.extend(self.apply(lambda col: col.shape))

    missing = ["Missing values", ""]
    missing.extend(self.isna().sum())

    description = ["Description", ""]
    description.extend([""] * len(columns))

    missing_total = self.isna().sum().sum()
    s = "" if missing_total == 1 else "s"

    caption = f"Data dictionary for {name}, shape {self.shape} with " +\
              f"{missing_total} missing value{s}."

    return _markdowntable(
            columns,
            dtypes,
            shape,
            missing,
            description,
            caption=caption
            )


DataFrame.data_dictionary = data_dictionary
