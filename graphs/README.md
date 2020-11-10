# Graphs

This directory will store the `.gr` files, as well
as links from which they can be downloaded.

## Description of graphs used

* We use the following graphs from section 5.1 of [[1]](#1)
    - Web data commons hyperlink graph of pages from 2012 and from 2014 [[2]](#2), [[3]](#3)
    - From [[1]](#1): "rmat26, rmat28, and kron30 are randomized
      synthetically generated scale-free graphs using the rmat [[4]](#4)
      and kron [[5]](#5) generators (we used weights of 0.57, 0.19, 0.19,
      and 0.05, as suggested by graph500 [[6]](#6))".
      We construct these graphs using [SNAP](http://snap.stanford.edu/)

## Installing SNAP

We will use the [krongen](https://github.com/snap-stanford/snap/tree/master/examples/krongen)
and [graphgen](https://github.com/snap-stanford/snap/tree/master/examples/graphgen)
portions of SNAP to generate R-MAT and Kronecker graphs.
For ease of installation, we will use [SNAP.py](https://snap.stanford.edu/snappy/#setup).
Since we are on CentOS and using Python 2, here are instructions that for installation:
```bash
cd ~
curl -O https://snap.stanford.edu/snappy/release/snap-stanford-5.0.0-5.0-centos6.5-x64-py2.6.tar.gz
tar -xzvf snap-stanford-5.0.0-5.0-centos6.5-x64-py2.6.tar.gz
cd snap-stanford-5.0.0-5.0-centos6.5-x64-py2.6
python setup.py install --user
```

## Makefile Instructions

* Set the environment variable
`GALOIS_BUILD_DIR` to the top directory of
your Galois build.

TODO: Add the smaller graphs so that we can actually store things.

TODO: Remanage the Makefile into `download-small-edgelists`, medium, etc.

* Run `make download-edgelists` to run all the following targets
    - `make download-example-arcs-edgelists` downloads a graph with around 100 edges from http://webdatacommons.org/hyperlinkgraph/
    - `make download-wdc12-edgelist` downloads the *(331 GB when gzipped)*
      web data commons 2012 page-level graph from http://webdatacommons.org/hyperlinkgraph/2012-08/download.html

* Run `make convert-edgelists` to convert all downloaded edge lists into graphs (skipping
  any which have already been converted).

* Run `make clean` to run all the following targets
    - `make clean-edgelists` removes the directory of 
      downloaded edge lists and all its contents
    - `make clean-graphs` removes all `*.gr` files in this directory

Given the enormous size of the graphs, we recommend you regularly
call `make clean-edgelists`.

## References

<a id="1">[1]</a> Roshan Dathathri, Gurbinder Gill, Loc Hoang, Hoang-Vu Dang, Alex Brooks, Nikoli Dryden, Marc Snir, Keshav Pingali, [Gluon: a communication-optimizing substrate for distributed heterogeneous graph analytics.](https://dl.acm.org/doi/10.1145/3192366.3192404) PLDI 2018: 752-768,

<a id="2">[2]</a> Robert Meusel, Sebastiano Vigna, Oliver Lehmberg, and Christian Bizer. 2012. Web Data Commons - Hyperlink Graphs. http://webdatacommons.org/hyperlinkgraph/

<a id="3">[3]</a> Robert Meusel, Sebastiano Vigna, Oliver Lehmberg, and Christian Bizer. 2014. Graph Structure in the Web — Revisited: A Trick of the Heavy Tail. In Proceedings of the 23rd International Conference on World Wide Web (WWW ’14 Companion). ACM, New York, NY, USA, 427-432. https://doi.org/10.1145/2567948.2576928

<a id="4">[4]</a> Deepayan Chakrabarti, Yiping Zhan, and Christos Faloutsos. 2004.  R-MAT: A Recursive Model for Graph Mining. 442–446. https://doi.org/ 10.1137/1.9781611972740.43

<a id="5">[5]</a> Jure Leskovec, Deepayan Chakrabarti, Jon Kleinberg, Christos Faloutsos, and Zoubin Ghahramani. 2010. Kronecker Graphs: An Approach to Modeling Networks. J. Mach. Learn. Res. 11 (March 2010), 985–1042.  http://dl.acm.org/citation.cfm?id=1756006.1756039

<a id="6">[6]</a> 2010. Graph 500 Benchmarks. http://www.graph500.org
