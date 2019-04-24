sudo rm -rf /tmp/elmo
allennlp train \
    elmo.jsonnet \
    -s /tmp/elmo \
    --include-package nlp

allennlp evaluate \
	/tmp/elmo \
	data/test.txt \
	--include-package nlp
