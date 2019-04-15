import sys
sys.tracebacklimit=0

from typing import Iterator, List, Dict
import shutil
import tempfile

import torch
import numpy as np

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.tokenizers import Token

from allennlp.predictors.predictor import Predictor

@Predictor.register('lstm-predictor')
class Predictor(Predictor):
    results = {
        "Correct" : 0,
        "Incorrect" : 0,
        "Total": 0
    }
    
    @overrides
    def load_line(self, line: str) -> Dict[str, any]:
        name, tag = line.split("###")
        return {
            "name" : name,
            "tag" : tag
        }
    
    @overrides
    def predict_json(self, value: Dict[str, any]) -> JsonDict:
        instance = self._dataset_reader.text_to_instance([Token(char) for char in value["name"]])
        predictions = self.predict_instance(instance)["logits"]
        tag = self._model.vocab.get_token_from_index(np.argmax(predictions, axis=-1), "labels")
        
        if tag == value["tag"]:
            self.results["Correct"] += 1
        else:
            self.results["Incorrect"] += 1
        self.results["Total"] += 1
        
        print(self.results)
        
        return {"Actual" : value["tag"], "Prediction" : tag}