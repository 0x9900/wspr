#!/usr/bin/env python
"""

To read leaf.py you need to install the following packages:
- matplotlib
- numpy
- requests

"""

import argparse
import collections
import logging
import math
import os
import sys

from datetime import datetime
from functools import partial

import json
import matplotlib.pyplot as plt
import numpy as np
import requests

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

DXPLORER_CALL = os.getenv("CALLSIGN", "W6BSD")
DXPLORER_KEY = os.getenv("KEY")
DXPLORER_URL = "http://dxplorer.net/wspr/tx/spots.json"

class Config(object):
  target = '/tmp'
  granularity = 5
  fig_size = (16, 6)
  timelimit = '1d'


class WsprData(object):
  __slot__ = [
    "distance", "tx_call", "timestamp", "drift", "tx_grid", "rx_call", "power_dbm",
    "rx_grid", "azimuth", "snr", "freq",
  ]

  def __init__(self, *args, **kwargs):
    for key, val, in kwargs.items():
      setattr(self, key, val)


def readfile(filename):
  try:
    with open(filename, 'rb') as fdi:
      data = json.load(fdi)
  except (ValueError, IOError) as err:
    logging.error(err)
    sys.exit(os.EX_OSFILE)

  return [WsprData(**d) for d in data]

def download():
  params = dict(
    callsign=DXPLORER_CALL,
    key=DXPLORER_URL,
    timelimit=Config.timelimit,
  )
  resp = requests.get(url=DXPLORER_URL, params=params)
  data = resp.json()
  logging.info('Downloaded %d records', len(data))
  return [WsprData(**d) for d in data]

def reject_outliers(data, magnitude=1.2):
  q25, q75 = np.percentile(data, [25, 75])
  iqr = q75 - q25

  min = q25 - (iqr * magnitude)
  max = q75 + (iqr * magnitude)

  return [x for x in data if min <= x <= max]

def azimuth(wspr_data):
  filename = os.path.join(Config.target, 'azimuth.png')
  logging.info('Drawing azimuth to %s', filename)

  data = collections.defaultdict(set)
  for node in wspr_data:
    key = Config.granularity * (node.azimuth / Config.granularity)
    data[key].add(node.distance)

  elements = []
  for azim, dists in data.iteritems():
    for dist in reject_outliers(list(dists)):
      elements.append((azim, dist))

  az, el = zip(*elements)

  fig = plt.figure()
  ax = fig.add_subplot(111, polar=True)
  ax.set_theta_zero_location("N")
  ax.set_theta_direction(-1)

  ax.scatter(az, el)
  ax.set_title('Azimuth\nDistance', loc='left')
  plt.savefig(filename)
  plt.close()

def boxPlot(wspr_data):
  # basic plot
  filename = os.path.join(Config.target, 'boxplot.png')
  logging.info('Drawing boxplot to %s', filename)

  collection = collections.defaultdict(list)
  for val in wspr_data:
    key = 3600 * (val.timestamp / 3600)
    collection[key].append(val.distance)

  data = []
  for key, values in sorted(collection.items()):
    data.append((datetime.utcfromtimestamp(key), values))

  hours, values = zip(*data)
  fig, ax = plt.subplots(figsize=Config.fig_size)
  bplot = ax.boxplot(values, sym="b.", patch_artist=True)
  for patch in bplot['boxes']:
    patch.set(color='b', linewidth=1)

    patch.set_color('lightblue')

  ax.grid(True)
  ax.set_xticklabels(['{}'.format(h.strftime('%d %R')) for h in hours])

  plt.title('Distances')
  plt.grid(linestyle='dotted')
  plt.savefig(filename)
  plt.close()

def violinPlot(wspr_data):
  filename = os.path.join(Config.target, 'violin.png')
  logging.info('Drawing violin to %s', filename)

  # get only the relevant data and reject the outliers
  collection = collections.defaultdict(list)
  for val in wspr_data:
    key = 3600 * (val.timestamp / 3600)
    collection[key].append(val.distance)

  data = []
  for key, values in sorted(collection.items()):
    data.append((datetime.utcfromtimestamp(key), reject_outliers(values)))

  hours, values = zip(*data)
  fig, ax = plt.subplots(figsize=Config.fig_size)
  ax.grid(True)
  ax.set_xticklabels(['{}'.format(h.strftime('%d %R')) for h in hours])

  ax.violinplot(values, showmeans=False, showmedians=True)

  plt.title('Distances')
  plt.grid(linestyle='dotted')
  plt.savefig(filename)
  plt.close()

def main():
  parser = argparse.ArgumentParser(description='WSPR Stats.')
  parser.add_argument('-d', '--debug', action='store_true', default=False,
                      help='Print information useful for debugging')
  parser.add_argument('-t', '--target-dir', default='/tmp',
                      help=('Target directory where the images will be '
                            'saved [default: %(default)s]'))
  parser.add_argument('-f', '--file', help='JSON file from DXPlorer.net')
  pargs = parser.parse_args()
  Config.target = pargs.target_dir
  if pargs.debug:
    _logger = logging.getLogger()
    _logger.setLevel('DEBUG')
    del _logger

  if pargs.file:
    wspr_data = readfile(pargs.file)
  else:
    wspr_data = download()

  boxPlot(wspr_data)
  violinPlot(wspr_data)
  azimuth(wspr_data)

if __name__ == "__main__":
  main()
