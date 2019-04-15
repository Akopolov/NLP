from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data.fields import TextField

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask

from allennlp.training.metrics import CategoricalAccuracy

torch.manual_seed(1)

@Model.register('lstm-tagger')
class LstmTagger(Model):
    def __init__(self, char_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        
        self.char_embeddings = char_embeddings
        
        self.encoder = encoder
        
        self.hidden2tag = torch.nn.Linear(
            in_features = encoder.get_output_dim(),
            out_features = vocab.get_vocab_size("labels"))
        
        self.accuracy = CategoricalAccuracy()
        
        self.loss_function = torch.nn.CrossEntropyLoss()
        
    def forward(self, chars: Dict[str, torch.Tensor], label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(chars)
        
        embeddings = self.char_embeddings(chars)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2tag(encoder_out)
        
        output = {"logits": logits}
        
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
            
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}