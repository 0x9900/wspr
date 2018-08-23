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

try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    Basemap = None

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

DXPLORER_URL = "http://dxplorer.net/wspr/tx/spots.json"

class Config(object):
  target = '/tmp'
  granularity = 5
  fig_size = (16, 6)
  timelimit = '28h'
  callsign = os.getenv("CALLSIGN", '').upper()
  key = os.getenv("KEY")

class WsprData(object):
  __slot__ = [
    "distance", "tx_call", "timestamp", "drift", "tx_grid", "rx_call", "power_dbm",
    "rx_grid", "azimuth", "snr", "freq",
  ]

  def __init__(self, *args, **kwargs):
    for key, val, in kwargs.items():
      setattr(self, key, val)

def grid2latlong(maiden):
    """
    Transform a maidenhead grid locator to latitude & longitude
    """
    maiden = maiden.strip().upper()
    assert len(maiden) in [2, 4, 6, 8], 'Locator length error: 2, 4, 6 or 8 characters accepted'
    charA = ord('A')

    multipliers = [20, 10, 2, 1, 5./60, 2.5/60, 5./600, 2.5/600]
    maiden = [ord(n) - charA if n >= 'A' else int(n) for n in maiden]
    lon = -180.
    lat = -90.
    for idx in range(0, len(maiden), 2):
        lon += maiden[idx] * multipliers[idx]
    for idx in range(1, len(maiden), 2):
        lat += maiden[idx] * multipliers[idx]

    return lon, lat

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
    callsign=Config.callsign,
    key=Config.key,
    timelimit=Config.timelimit,
  )
  resp = requests.get(url=DXPLORER_URL, params=params)
  data = resp.json()
  if 'Error' in data:
    logging.error(data['Error'])
    sys.exit(os.EX_OSFILE)
  logging.info('Downloaded %d records', len(data))
  return [WsprData(**d) for d in data]

def reject_outliers(data, magnitude=1.5):
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
  fig.text(.01, .02, 'http://github.com/0x9900/wspr')
  fig.suptitle('[{}] Azimuth x Distance'.format(Config.callsign),  fontsize=14, fontweight='bold')
  ax = fig.add_subplot(111, polar=True)
  ax.set_theta_zero_location("N")
  ax.set_theta_direction(-1)

  ax.scatter(az, el)
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
  fig.text(0.01, 0.02, 'http://github.com/0x9900/wspr')
  fig.suptitle('[{}] Distances'.format(Config.callsign),  fontsize=14, fontweight='bold')

  bplot = ax.boxplot(values, sym="b.", patch_artist=True)
  for patch in bplot['boxes']:
    patch.set(color='lightblue', linewidth=1)

  ax.grid(True)
  ax.set_xticklabels(['{}'.format(h.strftime('%R')) for h in hours])
  ax.set_ylabel('Km')

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
  fig.text(.01, .02, 'http://github.com/0x9900/wspr')

  fig.suptitle('[{}] Distances'.format(Config.callsign),  fontsize=14, fontweight='bold')
  ax.grid(True)
  ax.set_xticklabels(['{}'.format(h.strftime('%d %R')) for h in hours])
  ax.set_ylabel('Km')

  ax.violinplot(values, showmeans=False, showmedians=True)

  plt.grid(linestyle='dotted')
  plt.savefig(filename)
  plt.close()

def contactMap(wspr_data):
  filename = os.path.join(Config.target, 'contactmap.png')
  logging.info('Drawing connection map to %s', filename)

  fig, ax = plt.subplots(figsize=(14, 10))
  fig.text(0.01, 0.02, 'http://github/com/0x9900/wspr')
  fig.suptitle('[{}] Heatmap'.format(Config.callsign), fontsize=14, fontweight='bold')

  slon, slat = grid2latlong(wspr_data[0].tx_grid)

  logging.info("lat: %f / lon: %f", slat, slon)
  map = Basemap(projection='cyl', lon_0=slon, resolution='c')
  map.fillcontinents(color='linen', lake_color='aqua')
  map.drawcoastlines()
  map.drawmapboundary(fill_color='aqua')
  map.drawparallels(np.arange(-90.,90.,30.))
  map.drawmeridians(np.arange(-180.,180.,60.))

  # draw great circle route between NY and London
  for data in wspr_data:
    slon, slat = grid2latlong(data.tx_grid)
    dlon, dlat = grid2latlong(data.rx_grid)
    map.drawgreatcircle(slon, slat, dlon, dlat, linewidth=.5, color='g')
    x, y = map(dlon, dlat)
    map.plot(x, y, 'go', markersize=3, alpha=.5)
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

  if not pargs.file and not any([Config.callsign, Config.key]):
    logging.error('Call sign or key missing')
    sys.exit(os.EX_NOPERM)

  if pargs.file:
    wspr_data = readfile(pargs.file)
  else:
    wspr_data = download()

  boxPlot(wspr_data)
  violinPlot(wspr_data)
  azimuth(wspr_data)

  if Basemap:
    contactMap(wspr_data)
  else:
    logging.warning('Install maptplotlib.Basemap to generate the maps')

if __name__ == "__main__":
  main()
