# must a subdirectory of this directory
DOWNLOADED_EDGE_LISTS=downloadedEdgeLists

############################################ CLEANING TARGETS ##########
clean: clean-edgelists clean-graphs 

clean-edgelists:  # remove all downloaded graphs
	@echo "***Removing downloads" ; \
	rm -rf $(DOWNLOADED_EDGE_LISTS) ;

clean-graphs:  # remove all *.gr files
	@echo "***Removing *.gr files" ; \
	rm *.gr ;

############################################ CONVERSION TO *.gr ##########
convert-edgelists: 
	@echo "***Converting edge lists" ; \
	for edgeList in $(DOWNLOADED_EDGE_LISTS)/* ; do \
		echo ; \
		edgeListBase=$$(basename -- $$edgeList) ; \
		newFileName=$${edgeListBase%%.*}.gr ; \
		if [[ -f $$newFileName ]] ; then \
			echo "*** $$newFileName already exists, skipping conversion of $$edgeList" ; \
		else \
		    echo "***Converting graph file $$edgeList" ; \
		    $$GALOIS_BUILD_DIR/tools/graph-convert/graph-convert --edgelist2gr $$edgeList $$newFileName ; \
		fi ; \
	done 

############################################ DOWNLOADS ##########
download-edgelists: download-example-arcs-edgelist

download-example-arcs-edgelist: 
	cd $(DOWNLOADED_EDGE_LISTS) ; \
	curl -O http://webdatacommons.org/hyperlinkgraph/data/example_arcs ; 
