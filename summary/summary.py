"""
File: eda.py
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

import pandas as pd


def summary(self):
    """Describe the columns of a data frame with
    excerpted values, types, and summary statistics."""
    return pd.concat([
            self.head(5).T,
            pd.DataFrame({"type": self.dtypes}),
            pd.DataFrame({"pct_missing": self.isna().mean() * 100}),
            self.describe(percentiles=[0.5]).drop("count").T
        ], axis=1)


def summary_by(self, col):
    """Describe the columns of a data frame, grouped by column
    `col` with excerpted values, types, and summary statistics."""
    dfg = self.groupby(col)
    return pd.concat([
            dfg.first().T.add_prefix("head_"),
            dfg.dtypes.T.add_prefix("type_"),
            dfg.apply(lambda x: x.isna().mean() * 100)
               .T.add_prefix("pct_missing_")
        ], axis=1)


def missing(self, *args, **kwargs):
    """Display bar graph of missing data by column."""
    self.\
        isna().\
        mean().\
        apply(lambda x: x * 100).\
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
    self.\
        drop(col, axis=1).\
        isna().\
        join(self[col]).\
        groupby(col).\
        mean().\
        apply(lambda x: x * 100).\
        transpose().\
        plot.\
        barh(
           title="Percent data missing by column",
           xlim=(0, 100),
           *args,
           **kwargs
           )


pd.DataFrame.summary = summary
pd.DataFrame.summary_by = summary_by
pd.DataFrame.missing = missing
pd.DataFrame.missing_by = missing_by
