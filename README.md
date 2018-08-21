# wspr

WSPR is a digital radio communication mode for probing potential
propagation paths or antenna performance with low-power transmissions.

The Weak Signal Propagation Reporter Network (WSPRnet) is a group of
amateurs radio operators using WSPR with very low power (QRP/QRPp)
transmissions.  They automatically upload their reception reports to a
central database called WSPRnet. This program downloads the data of
these transmission reports create several graphs useful to analyze how
propagation works or to optimize your antenna setting.

## Usage

    wspr[523]$ ./leaf.py --help
    usage: leaf.py [-h] [-d] [-t TARGET_DIR] [-f FILE]

    WSPR Stats.

    optional arguments:
      -h, --help            show this help message and exit
      -d, --debug           Print information useful for debugging
      -t TARGET_DIR, --target-dir TARGET_DIR
                            Target directory where the images will be saved
                            [default: /tmp]
      -f FILE, --file FILE  JSON file from DXPlorer.net

## Output example

The following boxplot graph show for each day at what distance the
bulk of your communication was heard. It also shows the distance
minima and maxima as well as the outliers.

![Distances](graphs/boxplot.png)

The violin graph shows how the station hearing your signal are
distributed in the IQR.

![Distribution](graphs/violin.png)

The azimuth graph shows what direction will show if your signal has
been heard more in a specific direction.

![Azimuth](graphs/azimuth.png)

## WSPRLite

The WSPRlite is a special test transmitter that sends a signal to a
Worldwide network of receiving stations.

![WSPR Picture](misc/wspr.jpg)

-- Fred C / [W6BSD](http://www.qrz.com/db/W6BSD) --
