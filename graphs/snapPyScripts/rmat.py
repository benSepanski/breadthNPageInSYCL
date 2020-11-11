# Based on https://snap.stanford.edu/snappy/doc/reference/GenRMat.html
#
# Usage:
#
# python rmat.py <outputFile> <scale>
# 
# Where |V| = 2^{scale}
# Note that we use the recommended probabilities and edgefactor
# from the graph500 benchmarks
# https://graph500.org/?page_id=12#alg:generator
#
# i.e.
# edgefactor = 16
# a=0.57, b=c=0.19, d=1-a-b-c=0.05


import sys
import snap

if len(sys.argv) != 3:
   raise ValueError("Expected exactly 2 command line arguments")

outFile = sys.argv[1]
scale = int(sys.argv[2])
edgefactor = 16
a = 0.57
b, c = 0.19, 0.19

nnodes = 2**scale
# In case scale <= 2, make sure nedges <= nnodes choose 2
nedges = min(nnodes * edgefactor, nnodes * (nnodes - 1) / 2)

Rnd = snap.TRnd()
graph = snap.GenRMat(nnodes, nedges, a, b, c, Rnd)
snap.SaveEdgeList(graph, outFile)
