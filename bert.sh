# !/bin/bash

sudo rm -rf /tmp/bert
allennlp train \
	bert.jsonnet \
	-s /tmp/bert \
	--include-package nlp

allennlp evaluate \
	/tmp/bert \
	data/test.txt \
	--include-package nlp
