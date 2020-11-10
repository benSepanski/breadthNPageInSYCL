This directory will store the `.gr` files, as well
as links from which they can be downloaded.


* Set the environment variable
`GALOIS_BUILD_DIR` to the top directory of
your Galois build.

* Run `make download-edgelists` to run all the following targets
    - `make download-example-arcs-edgelists` downloads a graph with around 100 edges from http://webdatacommons.org/hyperlinkgraph/
    - `make download-wdc12-edgelist` downloads the *(331 GB)* web data commons 2012 page-level graph from http://webdatacommons.org/hyperlinkgraph/2012-08/download.html

* Run `make convert-edgelists` to convert all downloaded edge lists into graphs (skipping
  any which have already been converted).

* Run `make clean` to run all the following targets
    - `make clean-edgelists` removes the directory of 
      downloaded edge lists and all its contents
    - `make clean-graphs` removes all `*.gr` files in this directory

Given the enormous size of the graphs, we recommend you regularly
call `make clean-edgelists`.
