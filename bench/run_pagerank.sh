# Run benchmark with device $1
for file in `cat graphsToRun.txt` ; do
    base=`basename "$file"`;
    outFile="results/pagerank${base%%.gr}${1}.out" ;
    if [ -f $outFile ] ; then
        echo "$outFile already exits" ;
    else
        touch $outFile ;
        echo "Running on $file"
        echo "INPUTGRAPH $file" >> $outFile
        for i in {1..3} ; do
            echo "Device $1" >> $outFile
            echo "SYCL Data-Driven" >> $outFile
            ../build/pagerank/pagerank-data-driven $file -g $1 -x 5000 >>$outFile 2>&1
        done
        for i in {1..3} ; do
            echo "Device $1" >> $outFile
            echo "SYCL Topology-Driven" >> $outFile
            ../build/pagerank/pagerank-topology-driven $file -g $1 -x 5000 >>$outFile 2>&1
        done
        for i in {1..3} ; do
            echo "Device $1" >> $outFile
            echo "Lonestar" >> $outFile
            ../build/extern/Galois/lonestar/analytics/gpu/pagerank/pagerank-gpu $file -g $1 -x 5000 >>$outFile 2>&1
        done
    fi
done
