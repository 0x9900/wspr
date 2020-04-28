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
import os
import sys

from datetime import datetime

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
  # pylint: disable=too-few-public-methods
  target = '/tmp'
  granularity = 5
  fig_size = (16, 6)
  timelimit = '25h'
  callsign = os.getenv("CALLSIGN", '').upper()
  key = os.getenv("KEY")


class WsprData(object):
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
  try:
    with open(filename, 'rb') as fdi:
      data = json.load(fdi)
  except (ValueError, IOError) as err:
    logging.error(err)
    sys.exit(os.EX_OSFILE)

  return [WsprData(**d) for d in data]


def download(timelimit, band):
  params = dict(callsign=Config.callsign,
                band=band,
                key=Config.key,
                timelimit=timelimit)
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


def smooth(y, box_pts):
  box = np.ones(box_pts)/box_pts
  y_smooth = np.convolve(y, box, mode='full')
  return y_smooth


def reject_outliers(data, magnitude=1.5):
  q25, q75 = np.percentile(data, [25, 75])
  iqr = q75 - q25

  qmin = q25 - (iqr * magnitude)
  qmax = q75 + (iqr * magnitude)

  return [x for x in data if qmin <= x <= qmax]


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

  azim, elems = zip(*elements)

  fig = plt.figure()
  fig.text(.01, .02, 'http://github.com/0x9900/wspr - Time span: %s' % Config.timelimit)
  fig.suptitle('[{}] Azimuth x Distance'.format(Config.callsign),  fontsize=14, fontweight='bold')

  ax_ = fig.add_subplot(111, polar=True)
  ax_.set_theta_zero_location("N")
  ax_.set_theta_direction(-1)
  ax_.scatter(azim, elems)

  plt.savefig(filename)
  plt.close()


def dist_plot(wspr_data):
  bucket_size = 120
  filename = os.path.join(Config.target, 'distplot.png')
  logging.info('Drawing distplot to %s', filename)

  data = collections.defaultdict(list)
  for val in wspr_data:
    key = bucket_size * (val.timestamp / bucket_size)
    data[key].append(float(val.distance))

  _, values = zip(*sorted(data.items()))
  values = smooth([np.percentile(v, 90, interpolation='midpoint') for v in values], 5)

  fig, ax_ = plt.subplots(figsize=Config.fig_size)
  fig.text(.01, .02, 'http://github.com/0x9900/wspr - Time span: %s' % Config.timelimit)
  fig.suptitle('[{}] Distances'.format(Config.callsign),  fontsize=14, fontweight='bold')

  ax_.plot(values)
  ax_.grid(True)

  plt.savefig(filename)
  plt.close()


def box_plot(wspr_data):
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
  fig, ax_ = plt.subplots(figsize=Config.fig_size)
  fig.text(.01, .02, 'http://github.com/0x9900/wspr - Time span: %s' % Config.timelimit)
  fig.suptitle('[{}] Distances'.format(Config.callsign), fontsize=14, fontweight='bold')

  bplot = ax_.boxplot(values, sym="b.", patch_artist=True)
  for patch in bplot['boxes']:
    patch.set(color='lightblue', linewidth=1)

  ax_.grid(True)
  ax_.set_xticklabels(['{}'.format(h.strftime('%R')) for h in hours])
  ax_.set_ylabel('Km')

  plt.grid(linestyle='dotted')
  plt.savefig(filename)
  plt.close()


def violin_plot(wspr_data):
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
  fig, ax_ = plt.subplots(figsize=Config.fig_size)
  fig.text(.01, .02, 'http://github.com/0x9900/wspr - Time span: %s' % Config.timelimit)

  fig.suptitle('[{}] Distances'.format(Config.callsign), fontsize=14, fontweight='bold')
  ax_.grid(True)
  ax_.set_xticklabels(['{}'.format(h.strftime('%d %R')) for h in hours])
  ax_.set_ylabel('Km')

  ax_.violinplot(values, showmeans=False, showmedians=True)

  plt.grid(linestyle='dotted')
  plt.savefig(filename)
  plt.close()


def contact_map(wspr_data):
  filename = os.path.join(Config.target, 'contactmap.png')
  logging.info('Drawing connection map to %s', filename)

  fig = plt.figure(figsize=(14, 10))
  fig.text(.01, .02, 'http://github/com/0x9900/wspr - Time span: %s' % Config.timelimit)
  fig.suptitle('[{}] Contact Map'.format(Config.callsign), fontsize=14, fontweight='bold')

  slon, slat = grid2latlong(wspr_data[0].tx_grid)

  logging.info("lat: %f / lon: %f", slat, slon)
  bmap = Basemap(projection='cyl', lon_0=slon, resolution='c')
  bmap.fillcontinents(color='linen', lake_color='aqua')
  bmap.drawcoastlines()
  bmap.drawmapboundary(fill_color='aqua')
  bmap.drawparallels(np.arange(-90., 90., 30.))
  bmap.drawmeridians(np.arange(-180., 180., 60.))

  # draw great circle route between NY and London
  for data in wspr_data:
    slon, slat = grid2latlong(data.tx_grid)
    dlon, dlat = grid2latlong(data.rx_grid)
    bmap.drawgreatcircle(slon, slat, dlon, dlat, linewidth=.5, color='g')
    x, y = bmap(dlon, dlat)
    bmap.plot(x, y, 'go', markersize=3, alpha=.5)
  plt.savefig(filename)
  plt.close()


def type_time(parg):
  errmsg = '--time: shoud be an integer followed by "H" for hour or "D" for days. Max 40 days.'
  parg = parg.upper()
  try:
    length, unit = float(parg[:-1]), parg[-1:]
  except ValueError:
    raise argparse.ArgumentTypeError(errmsg)

  if unit == 'H' and length <= 48:
    return int(length), unit

  if unit == 'H' and length > 48:
    length = round(length / 24.0)
    unit = 'D'

  if unit == 'D' and length < 40:
    return int(length), unit
  raise argparse.ArgumentTypeError(errmsg)


def main():
  parser = argparse.ArgumentParser(description='WSPR Stats.')
  parser.add_argument('-D', '--debug', action='store_true', default=False,
                      help='Print information useful for debugging')
  parser.add_argument('-t', '--target-dir', default='/tmp',
                      help=('Target directory where the images will be '
                            'saved [default: %(default)s]'))
  parser.add_argument('-f', '--file', help='JSON file from DXPlorer.net')
  parser.add_argument('-b', '--band', type=int, default=14,
                      help=('Band to download, in Mhz [default: %(default)s]'))
  parser.add_argument('-T', '--time', type=type_time, default='24h',
                      help=('Time: shoud be an integer followed by "H" for hour or '
                            '"D" for days [default: %(default)s]'))
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
    wspr_data = download('{}{}'.format(*pargs.time), pargs.band)

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
