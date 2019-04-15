#! /bin/bash
train()
{
sudo rm -rf /tmp/serialization-dir 
allennlp train \
    Config.jsonnet \
    -s /tmp/serialization-dir \
    --include-package home
}

predict()
{
allennlp predict \
    /tmp/serialization-dir \
    data/Test.txt \
    --include-package home \
    --predictor lstm-predictor \
    --output-file data/Test_Results.txt
}
