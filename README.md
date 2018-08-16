# wspr

## Stem and leaf plotting.
http://en.wikipedia.org/wiki/Stem-and-leaf_display

## Usage

    $ leaf.py --help
    usage: leaf.py [-h] [-s STEM_SIZE] filename

    Stem-and-leaf display.

    positional arguments:
      filename              File containing the values

    optional arguments:
      -h, --help            show this help message and exit
      -s STEM_SIZE, --stem_size STEM_SIZE
                            length of the stem [default: 2]


## Example:

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
