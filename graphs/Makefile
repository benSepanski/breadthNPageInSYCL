DOWNLOADED_EDGE_LISTS=downloadedEdgeLists # must be in this directory

make-graphs: 
	echo "***Making graphs" ; \
	for edgeList in $(DOWNLOADED_EDGE_LISTS) ; do \
		newFileName=$${edgeList%%.*}.gr ; \
		echo "***Converting graph file $$edgeList"
		./$$GALOIS_BUILD_DIR/tools/graph-convert/graph-convert --edgelist2gr $$edgeList $$newFileName ; \
	done

download-edge-lists:  # re-download all graphs
	echo "***Downloading edge lists" ; \
	mkdir -p $(DOWNLOADED_EDGE_LISTS) ; \
	for edgeListURL in `cat edgeListURLs.txt` ; do \
		cd $(DOWNLOADED_EDGE_LISTS) ; \
		curl -O $$edgeListURL ; \
		cd .. ; \
	done 

clean: clean-downloads clean-graphs 

clean-downloads:  # remove all downloaded graphs
	echo "***Removing downloads" ; \
	rm -rf $(DOWNLOADED_EDGE_LISTS) ;

clean-graphs:  # remove all *.gr files
	echo "***Removing *.gr files" ; \
	rm *.gr ;
