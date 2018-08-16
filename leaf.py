#!/usr/bin/env python

"""
Stem and leaf plotting.
http://en.wikipedia.org/wiki/Stem-and-leaf_display

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
