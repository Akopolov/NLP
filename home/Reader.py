import sys
sys.tracebacklimit=0

from typing import Iterator, List, Dict
import shutil
import tempfile

import torch
import numpy as np

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

@DatasetReader.register('char-reader')
class PosDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        
    def text_to_instance(self, tokens: List[Token], tag: str = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        field = {"chars": sentence_field}

        if tag:
            label_field = LabelField(label = tag)
            field["label"] = label_field
        
        return Instance(field)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                name, tag = line.split("###")
                yield self.text_to_instance([Token(char) for char in name], tag)