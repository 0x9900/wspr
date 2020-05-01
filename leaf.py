#!/usr/bin/env python
"""To use wspr/leaf.py you need to set 2 environment variables one
with your call sign the second one with your wspr (dxplorer) key.

For example:
$ export CALLSIGN="W6BSD"
$ export KEY="aGAT9om5wASsmx8CIrH48MB8Dhh"

"""

import argparse
import collections
import logging
import os
import sys

from datetime import datetime

import json
import matplotlib.dates as mdates
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
  """Store Configuration and global variables"""
  # pylint: disable=too-few-public-methods
  target = '/tmp'
  granularity = 8
  fig_size = (16, 6)
  timelimit = '25H'
  count = 1000
  callsign = os.getenv("CALLSIGN", '').upper()
  key = os.getenv("KEY")


class WsprData(object):
  """Structure storing WSPR data"""
  # pylint: disable=too-few-public-methods
  __slot__ = ["distance", "tx_call", "timestamp", "drift", "tx_grid", "rx_call", "power_dbm",
              "rx_grid", "azimuth", "snr", "freq"]

  def __init__(self, *_, **kwargs):
    for key, val, in kwargs.items():
      setattr(self, key, val)


def grid2latlong(maiden):
  """
  Transform a maidenhead grid locator to latitude & longitude
  """
  maiden = maiden.strip().upper()
  assert len(maiden) in [2, 4, 6, 8], 'Locator length error: 2, 4, 6 or 8 char_acters accepted'
  char_a = ord('A')

  multipliers = [20, 10, 2, 1, 5./60, 2.5/60, 5./600, 2.5/600]
  maiden = [ord(n) - char_a if n >= 'A' else int(n) for n in maiden]
  lon = -180.
  lat = -90.
  for idx in range(0, len(maiden), 2):
    lon += maiden[idx] * multipliers[idx]
  for idx in range(1, len(maiden), 2):
    lat += maiden[idx] * multipliers[idx]

  return lon, lat


def readfile(filename):
  """Read WSPR data file"""
  try:
    with open(filename, 'rb') as fdi:
      data = json.load(fdi)
  except (ValueError, IOError) as err:
    logging.error(err)
    sys.exit(os.EX_OSFILE)

  return [WsprData(**d) for d in data]


def download(band):
  """Download WSPR data from the dxplorer website"""
  params = dict(callsign=Config.callsign,
                band=band,
                key=Config.key,
                count=Config.count,
                timelimit=Config.timelimit)
  try:
    resp = requests.get(url=DXPLORER_URL, params=params)
    data = resp.json()
  except Exception as err:
    logging.error(err)
    raise

  if not data:
    logging.error('Empty data')
    sys.exit(os.EX_OSFILE)
  if 'Error' in data:
    logging.error(data['Error'])
    sys.exit(os.EX_OSFILE)

  logging.info('Downloaded %d records', len(data))
  return [WsprData(**d) for d in data]


def reject_outliers(data, magnitude=1.5):
  """Reject the statistical outliers from a list"""
  q25, q75 = np.percentile(data, [25, 75])
  iqr = q75 - q25

  qmin = q25 - (iqr * magnitude)
  qmax = q75 + (iqr * magnitude)

  return [x for x in data if qmin <= x <= qmax]


def azimuth(wspr_data):
  """Display the contacts azimut / distance."""
  filename = os.path.join(Config.target, 'azimuth.png')
  logging.info('Drawing azimuth to %s', filename)

  data = collections.defaultdict(set)
  for node in wspr_data:
    key = int(node.azimuth/Config.granularity) * Config.granularity
    data[key].add(node.distance)

  elements = []
  for azim, dists in data.items():
    for dist in reject_outliers(list(dists)):
      elements.append((azim, dist))

  fig = plt.figure()
  fig.text(.01, .02, 'http://github.com/0x9900/wspr - Time span: %s' % Config.timelimit)
  fig.suptitle('[{}] Azimuth x Distance'.format(Config.callsign), fontsize=14, fontweight='bold')

  ax_ = fig.add_subplot(111, polar=True)
  ax_.set_theta_zero_location("N")
  ax_.set_theta_direction(-1)
  ax_.scatter(*zip(*elements))

  plt.savefig(filename)
  plt.close()


def dist_plot(wspr_data):
  """Show the maximum distances"""

  filename = os.path.join(Config.target, 'distplot.png')
  logging.info('Drawing distplot to %s', filename)

  collection = collections.defaultdict(list)
  for val in wspr_data:
    key = 600 * (val.timestamp / 600)
    collection[key].append(float(val.distance))

  data = []
  for key, values in sorted(collection.items()):
    data.append((datetime.utcfromtimestamp(key), max(values)))

  fig, ax_ = plt.subplots(figsize=Config.fig_size)
  fig.text(.01, .02, 'http://github.com/0x9900/wspr - Time span: %s' % Config.timelimit)
  fig.suptitle('[{}] Distances'.format(Config.callsign), fontsize=14, fontweight='bold')
  fig.autofmt_xdate()

  ax_.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
  ax_.grid(True, linestyle='dotted')
  ax_.set_xlabel('UTC Time')
  ax_.set_ylabel('Km')
  ax_.plot(*zip(*data))

  plt.savefig(filename)
  plt.close()

def box_plot(wspr_data):
  """Box plot graph show the median, 75 and 25 percentile of the
  distance. It also show the outliers."""

  filename = os.path.join(Config.target, 'boxplot.png')
  logging.info('Drawing boxplot to %s', filename)

  collection = collections.defaultdict(list)
  for val in wspr_data:
    key = 3600 * (val.timestamp / 3600)
    collection[key].append(val.distance)

  data = []
  for key, values in sorted(collection.items()):
    data.append((datetime.utcfromtimestamp(key), values))

  fig, ax_ = plt.subplots(figsize=Config.fig_size)
  fig.text(.01, .02, 'http://github.com/0x9900/wspr - Time span: %s' % Config.timelimit)
  fig.suptitle('[{}] Distances'.format(Config.callsign), fontsize=14, fontweight='bold')
  fig.autofmt_xdate()

  labels, values = zip(*data)
  labels = ['{}'.format(h.strftime('%R')) for h in labels]

  ax_.grid(True, linestyle='dotted')
  ax_.set_xlabel('UTC Time')
  ax_.set_ylabel('Km')

  bplot = ax_.boxplot(values, sym="b.", patch_artist=True, autorange=True, labels=labels)
  for patch in bplot['boxes']:
    patch.set(color='silver', linewidth=1)

  plt.savefig(filename)
  plt.close()


def violin_plot(wspr_data):
  """After removing the outliers draw violin plot. This graph show where
  is the highest contact distances probabilities."""

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

  labels, values = zip(*data)
  labels = ['{}'.format(h.strftime('%R')) for h in labels]

  fig, ax_ = plt.subplots(figsize=Config.fig_size)
  fig.text(.01, .02, 'http://github.com/0x9900/wspr - Time span: %s' % Config.timelimit)
  fig.suptitle('[{}] Distances'.format(Config.callsign), fontsize=14, fontweight='bold')

  ax_.xaxis.set_ticks_position('bottom')
  ax_.set_xticks(np.arange(1, len(labels) + 1))
  ax_.set_xticklabels(labels)
  ax_.set_xlim(0.25, len(labels) + 0.75)
  ax_.set_xlabel('UTC Time')

  ax_.grid(True, linestyle='dotted')
  ax_.set_ylabel('Km')

  ax_.violinplot(values, showmeans=False, showmedians=True)

  plt.savefig(filename)
  plt.close()


def contact_map(wspr_data):

  """Show all the contacts on a map"""
  filename = os.path.join(Config.target, 'contactmap.png')
  logging.info('Drawing connection map to %s', filename)

  fig = plt.figure(figsize=(10, 10))
  fig.text(.01, .02, 'http://github/com/0x9900/wspr - Time span: %s' % Config.timelimit)
  fig.suptitle('[{}] Contact Map'.format(Config.callsign), fontsize=14, fontweight='bold')

  slon, slat = grid2latlong(wspr_data[0].tx_grid)

  logging.info("lat: %f / lon: %f", slat, slon)
  bmap = Basemap(projection='cyl', lon_0=slon, lat_0=slat, resolution='c')
  # bmap = Basemap(projection='cyl', lon_0=slon, lat_0=slat, resolution='l',
  #                llcrnrlat=0, urcrnrlat=90,
  #                llcrnrlon=-150, urcrnrlon=-60)
  # bmap.drawstates()

  bmap.fillcontinents(color='silver', lake_color='aqua')
  bmap.drawcoastlines()
  bmap.drawmapboundary(fill_color='aqua')
  bmap.drawparallels(np.arange(-90., 90., 45.))
  bmap.drawmeridians(np.arange(-180., 180., 45.))

  # draw great circle route between NY and London
  _calls = set([])
  for data in wspr_data:
    if data.rx_call in _calls:
      continue
    _calls.add(data.rx_call)
    slon, slat = grid2latlong(data.tx_grid)
    dlon, dlat = grid2latlong(data.rx_grid)
    bmap.drawgreatcircle(slon, slat, dlon, dlat, linewidth=.25, color='navy')
    x, y = bmap(dlon, dlat)
    bmap.plot(x, y, 'go', markersize=3, alpha=.5, color='navy')

  plt.savefig(filename)
  plt.close()

def main():
  """Every good program start with a main function"""
  parser = argparse.ArgumentParser(description='WSPR Stats.', usage=__doc__)
  parser.add_argument('-D', '--debug', action='store_true', default=False,
                      help='Print information useful for debugging')
  parser.add_argument('-t', '--target-dir', default='/tmp',
                      help=('Target directory where the images will be '
                            'saved [default: %(default)s]'))
  parser.add_argument('-f', '--file', help='JSON file from DXPlorer.net')
  parser.add_argument('-b', '--band', type=int, default=14,
                      help=('Band to download, in Mhz [default: %(default)s]'))
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
    wspr_data = download(pargs.band)

  box_plot(wspr_data)
  violin_plot(wspr_data)
  azimuth(wspr_data)
  dist_plot(wspr_data)

  if Basemap:
    contact_map(wspr_data)
  else:
    logging.warning('Install matplotlib.Basemap to generate the maps')

if __name__ == "__main__":
  main()
