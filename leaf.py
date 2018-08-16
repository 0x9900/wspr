#!/usr/bin/env python

"""
Stem and leaf plotting.
http://en.wikipedia.org/wiki/Stem-and-leaf_display

$ leaf.py --help
usage: leaf.py [-h] [-s STEM_SIZE] filename

Stem-and-leaf display.

positional arguments:
  filename              File containing the values

optional arguments:
  -h, --help            show this help message and exit
  -s STEM_SIZE, --stem_size STEM_SIZE
                        length of the stem [default: 2]


Output example:

$ leaf.py -s 2 numbers.txt
0 | 16 26 39 47 52 57 62 72
1 | 0 8 57 58 77
2 | 10 11 36 56 63 66 77 84
3 | 4 15 19 45 67
4 | 24 71 78 90
5 | 18 31 56 75 82 83 90 93 94
6 | 11 14 24 94 99
7 | 12 25 27 28 34 61 77 78 93 98
8 | 32 37 53 56 60 72 74 96
9 | 2 20 34 53 54 70 73 95

"""

import argparse
import collections
import math
import sys

def printStemPlot(values, stem_size):
    """Prints a stem plot of the values."""
    stem_size = 10 ** stem_size
    stems = collections.defaultdict(list)
    for v in values:
        stems[v / stem_size].append(v % stem_size)

    smin, smax = min(stems), max(stems)
    try:
        padding = "{:<%d}|" % int(math.log10(smax) + 2)
    except ValueError:
        padding = "{:<2}|"

    for key in range(smin, smax + 1):
        print padding.format(key),
        print ' '.join(str(leaf) for leaf in stems.get(key, []))


def main():
    parser = argparse.ArgumentParser(description='Stem-and-leaf display.')
    parser.add_argument('-s', '--stem_size', type=int, default=2,
                        help='length of the stem [default: %(default)s]')
    parser.add_argument('filename', help='File containing the values')

    args = parser.parse_args()

    try:
        with open(args.filename, 'r') as fdi:
            values = [int(n) for n in fdi]
    except IOError as err:
        print err
        sys.exit(1)

    values.sort()
    printStemPlot(values, args.stem_size)

if __name__ == "__main__":
    main()

# Local Variables:
# python-indent-offset: 4
# tab-width: 4
# python-indent: 4
# End:
