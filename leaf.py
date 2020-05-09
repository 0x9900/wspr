#!/usr/bin/env python
"""The program leaf.py download the last 24 hours worth of data from WSPR
net and compute statistical analysis of your contacts.

To use leaf.py you need to set 2 environment variables one
with your call sign the second one with your wspr (dxplorer) key.

For example:
$ export CALLSIGN="W6BSD"
$ export KEY="aGAT9om5wASsmx8CIrH48MB8Dhh"

"""

import argparse
import collections
import logging
import math
import os
import sys

from datetime import datetime
from dateutil import tz

import json
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import requests

from scipy.interpolate import make_interp_spline

try:
  from mpl_toolkits.basemap import Basemap
except ImportError:
  Basemap = None

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

DXPLORER_URL = "http://dxplorer.net/wspr/tx/spots.json"

DEFAULT_BAND = "20m"

BANDS = collections.OrderedDict((
    ("160m", 1),
    ("80m", 3),
    ("60m", 5),
    ("40m", 7),
    ("30m", 10),
    ("20m", 14),
    ("17m", 18),
    ("15m", 21),
    ("12m", 24),
    ("10m", 28),
    ("6m", 50),
    ("4m", 70),
    ("2m", 144),
    ("70cm", 432),
    ("23cm", 1296),
))


class Config(object):
  """Store Configuration and global variables"""
  # pylint: disable=too-few-public-methods
  target = '/tmp'
  granularity = 8
  percentile = 90
  fig_size = (14, 6)
  count = 10000
  timespan = 24
  callsign = os.getenv("CALLSIGN", '').upper()
  key = os.getenv("KEY")
  band = 14
  file = None


class WsprData(object):
  """Structure storing WSPR data"""
  # pylint: disable=too-few-public-methods
  __slot__ = ["distance", "tx_call", "timestamp", "drift", "tx_grid", "rx_call", "power_dbm",
              "rx_grid", "azimuth", "snr", "freq", "rx_lat", "rx_long", "tx_lat", "tx_long"]

  def __init__(self, *_, **kwargs):
    for key, val, in kwargs.items():
      setattr(self, key, val)
      if key == 'tx_grid':
        lat, lon = grid2latlon(val)
        setattr(self, 'tx_lat', lat)
        setattr(self, 'tx_lon', lon)
      elif key == 'rx_grid':
        lat, lon = grid2latlon(val)
        setattr(self, 'rx_lon', lon)
        setattr(self, 'rx_lat', lat)

  def __repr__(self):
    pattern = "WsprData: {0.tx_call} / {0.rx_call}, distance: {0.distance}, snr: {0.snr}"
    return pattern.format(self)


def grid2latlon(maiden):
  """
  Transform a maidenhead grid locator to latitude & longitude
  """
  assert isinstance(maiden, basestring), "Maidenhead locator has to be a string"

  maiden = maiden.strip().upper()
  maiden_lg = len(maiden)
  assert len(maiden) in [2, 4, 6, 8], 'Locator length error: 2, 4, 6 or 8 characters accepted'

  char_a = ord("A")
  lon = -180.0
  lat = -90.0

  lon += (ord(maiden[0]) - char_a) * 20
  lat += (ord(maiden[1]) - char_a) * 10

  if maiden_lg >= 4:
    lon += int(maiden[2]) * 2
    lat += int(maiden[3]) * 1
  if maiden_lg >= 6:
    lon += (ord(maiden[4]) - char_a) * 5.0 / 60
    lat += (ord(maiden[5]) - char_a) * 2.5 / 60
  if maiden_lg >= 8:
    lon += int(maiden[6]) * 5.0 / 600
    lat += int(maiden[7]) * 2.5 / 600

  return lat, lon


def readfile():
  """Read WSPR data file"""
  try:
    with open(Config.file, 'rb') as fdi:
      data = json.load(fdi)
  except (ValueError, IOError) as err:
    logging.error(err)
    sys.exit(os.EX_OSFILE)

  return [WsprData(**d) for d in data]


def download():
  """Download WSPR data from the dxplorer website"""
  params = dict(callsign=Config.callsign,
                band=BANDS[Config.band],
                key=Config.key,
                count=Config.count,
                timelimit="24H")
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

  data = []
  for node in wspr_data:
    data.append((math.radians(int(node.azimuth/Config.granularity) * Config.granularity),
                 (node.distance / 50) * 50))

  dist_count = collections.defaultdict(int)
  for elem in data:
    dist_count[elem] += 1

  theta = []
  distance = []
  density = []
  for key, cnt in dist_count.items():
    theta.append(key[0])
    distance.append(key[1])
    density.append(cnt * 3)

  fig = plt.figure(figsize=(8, 8))
  fig.text(.01, .02, ('http://github.com/0x9900/wspr - Distance & direction - '
                      'Time span: %sH - Band: %s') % (Config.timespan, Config.band))
  fig.suptitle('[{}] Azimuth x Distance'.format(Config.callsign), fontsize=14, fontweight='bold')

  ax_ = fig.add_subplot(111, projection="polar")
  ax_.set_theta_zero_location("N")
  ax_.set_theta_direction(-1)
  ax_.scatter(theta, distance, s=density, c=theta, cmap='PiYG', alpha=0.8)

  plt.savefig(filename)
  plt.close()


def dist_plot(wspr_data):
  """Show the maximum distances"""

  filename = os.path.join(Config.target, 'distplot.png')
  logging.info('Drawing distplot to %s', filename)

  collection = collections.defaultdict(list)
  for data in wspr_data:
    collection[900 * (data.timestamp / 900)].append(data.distance)
  collection = {k: np.percentile(v, Config.percentile) for k, v in collection.items()}

  xval, yval = zip(*[(k, v) for k, v in sorted(collection.items())])
  xval = np.array(xval)
  yval = np.array(yval)

  xnew = np.linspace(xval.min(), xval.max(), len(xval) * 10)
  k_factor = 5 if len(xval) > 10 else 1
  spline = make_interp_spline(xval, yval, k=k_factor)
  smooth = spline(xnew)

  fig, ax_ = plt.subplots(figsize=Config.fig_size)
  fig.text(.01, .02, ('http://github.com/0x9900/wspr - Distance %sth percentile - Time span: '
                      '%sH - Band: %s') % (Config.percentile, Config.timespan, Config.band))
  fig.suptitle('[{}] Distances'.format(Config.callsign), fontsize=14, fontweight='bold')
  fig.autofmt_xdate()

  ax_.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
  ax_.grid(True, which="both", linestyle='dotted')
  ax_.set_xlabel('UTC Time')
  ax_.set_ylabel('Km')
  ax_.set_yscale('log')
  ax_.set_ylim(yval.min() / 2, yval.max()+1000)
  ax_.plot([datetime.utcfromtimestamp(x) for x in xnew], smooth)

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
  fig.text(.01, .02, ('http://github.com/0x9900/wspr - Distance quartile range - '
                      'Time span: %sH - Band: %s') % (Config.timespan, Config.band))
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
  fig.text(.01, .02, ('http://github.com/0x9900/wspr - Distance and contacts density - '
                      'Time span: %sH - Band: %s') % (Config.timespan, Config.band))
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

  __calls = []
  points = []
  for data in wspr_data:
    if data.rx_call in __calls:
      continue
    __calls.append(data.rx_call)
    points.append((data.rx_lon, data.rx_lat))

  points = np.array(points)
  right, upd = points.max(axis=0)
  left, down = points.min(axis=0)

  fig = plt.figure(figsize=(12, 8))
  fig.text(.01, .02, ('http://github/com/0x9900/wspr - Contacts map - '
                      'Time span: %sH - Band: %s') % (Config.timespan, Config.band))
  fig.suptitle('[{}] Contact Map'.format(Config.callsign), fontsize=14, fontweight='bold')

  logging.info("Origin lat: %f / lon: %f", wspr_data[0].tx_lat, wspr_data[0].tx_lon)
  bmap = Basemap(projection='cyl', lon_0=wspr_data[0].tx_lon, lat_0=wspr_data[0].tx_lat,
                 urcrnrlat=upd+5, urcrnrlon=right+15, llcrnrlat=down-15, llcrnrlon=left-5,
                 resolution='c')
  bmap.drawlsmask(land_color = "#ddaa66", ocean_color="#7777ff", resolution = 'l')
  bmap.drawcoastlines(color="#707070")
  bmap.drawparallels(np.arange(-90., 90., 45.))
  bmap.drawmeridians(np.arange(-180., 180., 45.))

  for lon, lat in points:
    bmap.drawgreatcircle(wspr_data[0].tx_lon, wspr_data[0].tx_lat, lon, lat,
                         linewidth=.5, color='navy')
    x, y = bmap(lon, lat)
    bmap.plot(x, y, 'v', markersize=4, alpha=.5, color='red')

  plt.savefig(filename)
  plt.close()


def band_select(argument):
  """Select and validate the band passed as argument"""
  argument = argument.lower()
  if argument not in BANDS:
    raise argparse.ArgumentTypeError("Possible bands are:", ",".join(BANDS))
  return argument


def main():
  """Every good program start with a main function"""
  parser = argparse.ArgumentParser(description='WSPR Stats.', usage=__doc__)
  parser.add_argument('-D', '--debug', action='store_true', default=False,
                      help='Print information useful for debugging')
  parser.add_argument('-t', '--target-dir', default='/tmp',
                      help=('Target directory where the images will be '
                            'saved [default: %(default)s]'))
  parser.add_argument('-f', '--file', help='JSON file from DXPlorer.net')
  parser.add_argument('-b', '--band', type=band_select, default=DEFAULT_BAND,
                      help=('Band to download, in Mhz [default: %(default)s]'))
  pargs = parser.parse_args()

  Config.target = pargs.target_dir
  Config.band = pargs.band
  Config.filename = pargs.file

  if pargs.debug:
    _logger = logging.getLogger()
    _logger.setLevel('DEBUG')
    del _logger

  if not pargs.file and not any([Config.callsign, Config.key]):
    logging.error('Call sign or key missing')
    sys.exit(os.EX_NOPERM)

  if pargs.file:
    wspr_data = readfile()
  else:
    wspr_data = download()

  timespan = np.array([datetime.utcfromtimestamp(w.timestamp) for w in wspr_data])
  Config.timespan  = np.timedelta64(timespan.max() - timespan.min(), 'h').astype(int)

  try:
    box_plot(wspr_data)
    violin_plot(wspr_data)
    azimuth(wspr_data)
    dist_plot(wspr_data)
    if Basemap:
      contact_map(wspr_data)
  except ValueError as err:
    logging.error(err)
    logging.error('Your dataset is to small. Run WSPR for a longer time and gather more data')
    sys.exit(os.EX_DATAERR)

if __name__ == "__main__":
  main()
