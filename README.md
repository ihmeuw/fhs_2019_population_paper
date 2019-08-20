# Introduction

This module will be used to make most of the plots for the FHS populations paper
2019.

This module is probably not going to be importable.

# Making Plots

## Requirements

The `fbd_research` repository must be in your `PYTHONPATH` or installed into
a Python 3 environment.

The `fbd_core` repo _should_ be installed into the environments (it is
installed in `kpy3`, for instance). If you use a version of `fbd_core` on your
`PYTHONPATH`, make sure it is up to date.


## Plot

Make sure to label plots with figure numbers and what they are showing.


## Plug

Generally, the format for plugging involves putting a module-level docstring
that has the relevant paragraph(s) that are being plugged. They have XXs or
numbers or something similar where they need to be plugged. They should at least
print out something that is easy to read for plugging in the numbers to the
text. You might also output json or txt files (see previous paper dir). It can
also be good to add comments above the print statements that say which sentence
the numbers are for.
