# Run benchmark on graphs in file $1 on device $2 with block-sizes in file $3
for file in `cat $1` ; do
    base=`basename "$file"`;
    outFile="results/bfs-${base%%.gr}-dev${2}.out" ;
    if [ -f $outFile ] ; then
        echo "$outFile already exits" ;
    else
        touch $outFile
        echo "Running on $file"
        source_node=`cat ${file%%.gr}.source`;
        for i in {1..3} ; do
            echo "Device $2" >> $outFile
            echo "Lonestar" >> $outFile
            ../build/extern/Galois/lonestar/analytics/gpu/bfs/bfs-gpu $file -g $2 -s $source_node >>$outFile 2>&1
        done
        for blocks in `cat $3` ; do
            echo "INPUTGRAPH $file" >> $outFile
            for i in {1..3} ; do
                echo "Device $2" >> $outFile
                echo "SYCL Data-Driven" >> $outFile
                ../build/bfs/bfs-data-driven $file -g $2 -b $blocks -s $source_node >>$outFile 2>&1
            done
        done
        for blocks in `cat $3` ; do
            for i in {1..3} ; do
                echo "Device $2" >> $outFile
                echo "SYCL Topology-Driven" >> $outFile
                ../build/bfs/bfs-topology-driven $file -g $2 -b $blocks -s $source_node >>$outFile 2>&1
            done
        done
    fi
done
