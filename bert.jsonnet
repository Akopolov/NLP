local embedding_dim = 768;
local hidden_dim = 512;

{
  "dataset_reader": {
    "type": "sst_tokens",
    "token_indexers": {
      "token": {
        "type": "bert-pretrained",
        "do_lowercase": true,
        "pretrained_model": "bert-base-uncased"
      }
    }
  },
  "train_data_path": "data/train.txt",
  "validation_data_path": "data/dev.txt",
  
  "model": {
    "type": "lstm-model",
    "word_embeddings": {
      "token": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-uncased",
          "requires_grad": true
      },
      "allow_unmatched_keys": true
    },
    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 32,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 100,
    "patience": 5
  }
}