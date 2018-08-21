#!/usr/bin/env python
"""

$ leaf.py --help
usage: leaf.py [-h] [-s STEM_SIZE] filename

"""

import argparse
import collections
import logging
import math
import os
import sys

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

DXPLORER_CALL = os.getenv("CALLSIGN", "W6BSD")
DXPLORER_KEY = os.getenv("KEY")
DXPLORER_URL = "http://dxplorer.net/wspr/tx/spots.json"

FIG_SIZE = (16, 6)

class WsprData(object):
    __slot__ = [
        "distance", "tx_call", "timestamp", "drift", "tx_grid", "rx_call", "power_dbm",
        "rx_grid", "azimuth", "snr", "freq",
    ]

    def __init__(self, *args, **kwargs):
        for key, val, in kwargs.items():
            if key == 'timestamp':
                val = datetime.utcfromtimestamp(val)
            setattr(self, key, val)

def download():
    params = dict(
        callsign=DXPLORER_CALL,
        key=DXPLORER_URL,
        timelimit='1d',
    )
    logging.info('Downloaded %d records', len(data))
    resp = requests.get(url=DXPLORER_URL, params=params)
    data = resp.json()
    return [WsprData(**d) for d in data]

def reject_outliers(data, magnitude=1.2):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25

    min = q25 - (iqr * magnitude)
    max = q75 + (iqr * magnitude)

    return [x for x in data if min <= x <= max]

def azimuth(data):
    pass

def boxPlot(data):
    # basic plot
    print 'Drawing boxplot'
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.boxplot(data)
    ax.grid(True)

    plt.title('Distances')
    plt.grid(linestyle='dotted')
    plt.savefig('boxplot.png')
    plt.close()

def violinPlot(data):
    print 'Drawing violinplot'
    # get only the relevant data
    values = []
    for val in data:
        values.append(reject_outliers(val))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.violinplot(values, showmeans=False, showmedians=True)
    ax.grid(True)

    plt.title('Distances')
    plt.grid(linestyle='dotted')
    plt.savefig('violin.png')
    plt.close()

def main():
    collection = collections.defaultdict(list)

    wspr_data = download()
    for val in wspr_data:
        key = val.timestamp.day * 100 + val.timestamp.hour
        collection[key].append(val.distance)

    values = [np.array(v[1]) for v in sorted(collection.items())]

    boxPlot(values)
    violinPlot(values)

if __name__ == "__main__":
    main()

# Local Variables:
# python-indent-offset: 4
# tab-width: 4
# python-indent: 4
# End:
