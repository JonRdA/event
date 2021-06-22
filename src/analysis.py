
"""Main data analysis code for general functions using modules.

"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import dsp
import settings
from event import Event

logger = logging.getLogger(__name__)


# Summary functions

def loop_dir(directory, func, *args, **kwargs):
    """Loop over the directory of event files and apply function.

    Args:
        directory (path): directory path.
        func (function): method to be aplied to the events.
        *args: methods positional arguments.
        **kwargs: method's keyword arguments.

    Returns:
        pd.Series: func returned object per event file.
    """
    results = pd.Series(dtype=object)

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = Path(directory, filename)
            ev = Event(path)

            ev.load_data()
            results[filename] = func(ev, *args, **kwargs)
            logger.debug(f"Applied {func.__name__} to '{filename}'.")
        else:
            pass
            logger.warning(f"File '{filename}' skipped on '{directory}")

    return results


def main():
    dire = Path(r"C:\Users\Jon\Desktop\Bases de datos\Output")
    # ev_1 = Event(dire + "/2020.09.21_09.48.28.txt")
    # ev_1.load_data(True)

    # a = ev_1.main_freq()
    # a = pd.Series(a.iloc[0, :])
    # print(a)
    a = loop_dir(dire, Event.main_freq)
    print(a)


if __name__ == '__main__':

    # If module directly run, load log configuration for all modules.
    import logging.config
    logging.config.fileConfig('../log/logging.conf')
    logger = logging.getLogger('event')

    main()
