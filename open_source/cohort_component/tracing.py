"""
It's like logging, but for plots. Touch a file called plot in the current
directory to turn this on. That way it doesn't accidentally go into git
turned on.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os


LOGGER = logging.getLogger("fbd_research.cohort_component.tracing")
PLOT_ON = os.path.exists("plot")
WROTE_ALREADY = set()


def plot(name, x, y=None):
    """
    This is the equivalent of logging, except it logs a PDF.
    """
    if not PLOT_ON:
        return

    if not __debug__:
        return

    if name in WROTE_ALREADY:
        return

    LOGGER.debug("traceplot {}".format(name))
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    sns.set(color_codes=True)

    filename = name if name.endswith(".pdf") else name + ".pdf"

    with PdfPages(filename) as pdf:
        plt.figure(figsize=(9,9))
        if y is not None:
            plt.plot(x, y)
        else:
            plt.plot(x)
        plt.title(name)
        pdf.savefig()
        plt.close()

    WROTE_ALREADY.add(name)
