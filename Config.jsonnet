local embedding_dim = 6;
local hidden_dim = 128;
local num_epochs = 200;
local patience = 10;
local batch_size = 200;
local learning_rate = 0.05;

{
    "train_data_path": 'data/Train.txt',
    "validation_data_path": 'data/Validate.txt',
    "dataset_reader": {
        "type": "char-reader"
    },
    "model": {
        "type": "lstm-tagger",
        "char_embeddings": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["chars", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
        "patience": patience
    }
}
