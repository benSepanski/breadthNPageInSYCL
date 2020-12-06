# Run benchmark on graphs in file $1 on device $2 with block-sizes in file $3
MAXIT=5000;
for file in `cat $1` ; do
    base=`basename "$file"`;
    outFile="results/pagerank-${base%%.gr}-dev${2}.out" ;
    if [ -f $outFile ] ; then
        echo "$outFile already exits" ;
    else
        touch $outFile
        echo "Running on $file"
        for i in {1..3} ; do
            echo "Device $2" >> $outFile
            echo "Lonestar" >> $outFile
            ../build/extern/Galois/lonestar/analytics/gpu/pagerank/pagerank-gpu $file -g $2 -x $MAXIT >>$outFile 2>&1
        done
        for blocks in `cat $3` ; do
            echo "INPUTGRAPH $file" >> $outFile
            for i in {1..3} ; do
                echo "Device $2" >> $outFile
                echo "SYCL Data-Driven" >> $outFile
                ../build/pagerank/pagerank-data-driven $file -g $2 -b $blocks -x $MAXIT >>$outFile 2>&1
            done
        done
        for blocks in `cat $3` ; do
            for i in {1..3} ; do
                echo "Device $2" >> $outFile
                echo "SYCL Topology-Driven" >> $outFile
                ../build/pagerank/pagerank-topology-driven $file -g $2 -b $blocks -x $MAXIT >>$outFile 2>&1
            done
        done
    fi
done
