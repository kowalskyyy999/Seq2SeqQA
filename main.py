import os
from statistics import mode
import nltk

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from scripts.utils import *
from scripts.dataset import *
from scripts.model import *
from scripts.engine import *
from scripts.config import *

from dotenv import load_dotenv

config = load_dotenv('.env')

def main():
    path = os.getenv('DATA_PATH')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab =  Vocabulary()
    for w in nltk.corpus.words.words():
        vocab.add_word(w)

    train_dataset = QuACDataset(path, vocab)
    vocab = train_dataset.vocab
    
    val_dataset = QuACDataset(path, vocab, training=False)
    vocab = val_dataset.vocab

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    INPUT_SIZE = len(vocab.index2word)
    OUTPUT_SIZE = len(vocab.index2word)

    model = Seq2Seq(INPUT_SIZE, OUTPUT_SIZE, vocab)

    count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    engine = Engine(vocab, model, optimizer, criterion, EPOCHS, device)
    engine.fit(train_loader)

    checkpoints = {
        'vocab':vocab,
        'input_size':INPUT_SIZE,
        'output_size':OUTPUT_SIZE,
        'model':{
            'state_dict':engine.model.state_dict(),
            'architecture':model,
            'embedding_size':EMBEDDING_DIM,
            'hidden_size':HIDDEN_SIZE,
            'num_layers':NUM_LAYERS

        },
        'optimizer':{
            'state_dict':engine.optimizer.state_dict(),
            'optimizer':optimizer
        }

    }

    engine.save_model(checkpoints, '1.0.0.pth')

if __name__ == "__main__":
    main()