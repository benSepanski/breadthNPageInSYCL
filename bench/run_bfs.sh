# Run benchmark with device $1
for file in `cat graphsToRun.txt` ; do
    base=`basename "$file"`;
    outFile="results/bfs${base%%.gr}${1}.out" ;
    if [ -f $outFile ] ; then
        echo "$outFile already exits" ;
    else
        touch $outFile
        echo "Running on $file"
        source_node=`cat ${file%%.gr}.source`;
        echo "INPUTGRAPH $file" >> $outFile
        for i in {1..3} ; do
            echo "Device $1" >> $outFile
            echo "SYCL Data-Driven" >> $outFile
            ../build/bfs/bfs-data-driven $file -g $1 -s $source_node >>$outFile 2>&1
        done
        for i in {1..3} ; do
            echo "Device $1" >> $outFile
            echo "SYCL Topology-Driven" >> $outFile
            ../build/bfs/bfs-topology-driven $file -g $1 -s $source_node >>$outFile 2>&1
        done
        for i in {1..3} ; do
            echo "Device $1" >> $outFile
            echo "Lonestar" >> $outFile
            ../build/extern/Galois/lonestar/analytics/gpu/bfs/bfs-gpu $file -g $1 -s $source_node >>$outFile 2>&1
        done
    fi
done
