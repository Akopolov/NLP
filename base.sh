# !/bin/bash

sudo rm -rf /tmp/base
allennlp train \
	base.jsonnet \
	-s /tmp/base \
	--include-package nlp

allennlp evaluate \
	/tmp/base \
	data/test.txt \
	--include-package nlp
