import os
import nltk

import torch
from torch.utils.data import DataLoader

from scripts.utils import *
from scripts.dataset import *
from scripts.model import *

from dotenv import dotenv_values

config = dotenv_values('.env')

def main():
    path = config['DATA_PATH']

    vocab =  Vocabulary()
    for w in nltk.corpus.words.words():
        vocab.add_word(w)

    train_dataset = QuACDataset(path, vocab)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    INPUT_SIZE = len(train_dataset.vocab.index2word)
    OUTPUT_SIZE = len(train_dataset.vocab.index2word)
 
    context, question, answer = next(iter(train_loader))

    model = Seq2Seq(INPUT_SIZE, OUTPUT_SIZE, train_dataset.vocab)
    print(model)
    count_parameters(model)
    outputs = model(context, question, answer)

if __name__ == "__main__":
    main()