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
make-downloaded-edge-list-dir:
	@mkdir -p $(DOWNLOADED_EDGE_LISTS)

download-edgelists: \
    download-example-arcs-edgelist \
    download-wdc12-edgelist

download-example-arcs-edgelist: make-downloaded-edge-list-dir
	@echo "*** Downloading example_arcs from webdatacommons.org" ; \
	echo ; \
	cd $(DOWNLOADED_EDGE_LISTS) ; \
	curl -O http://webdatacommons.org/hyperlinkgraph/data/example_arcs ; \
	cd .. ; \
	echo ; \
	echo ;

# This is around 331 GB!!!
download-wdc12-edgelist: make-downloaded-edge-list-dir
	@echo "*** Downloading wdc12 from webdatacommons.org... this is over 330GB" ; \
	echo ; \
	cd $(DOWNLOADED_EDGE_LISTS) ; \
	wget -i http://webdatacommons.org/hyperlinkgraph/2012-08/data/arc.list.txt ; \
	cd .. ; \
	echo ; \
	echo 
