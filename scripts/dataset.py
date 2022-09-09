from calendar import c
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader

from scripts.utils import *

class QuACDataset(Dataset):
    def __init__(self, path, vocab, training=True):
        if training:
            with open(os.path.join(path, 'train_v0.2.json'), 'r') as f:
                self.json = json.loads(f.read())

        else:
            with open(os.path.join(path, 'val_v0.2.json'), 'r') as f:
                self.json = json.loads(f.read())
        self.df = parseJson(self.json)
        self.quac_vocab = self.createVocab(self.json, vocab)
        self.vocab = self.quac_vocab.vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        payload = self.df.iloc[index]
        context = self._convertToken(payload['context'], 'c')
        question = self._convertToken(payload['question'], 'q')
        answer = self._convertToken(payload['answers'], 'a')

        context_tensor = torch.tensor(context, dtype=torch.int32)
        question_tensor = torch.tensor(question, dtype=torch.int32)
        answer_tensor = torch.tensor(answer, dtype=torch.int32)

        return context_tensor, question_tensor, answer_tensor  

    @staticmethod
    def createVocab(json_file, vocab):
        quac_vocab = QuAC_Vocab(vocab)
        for file in json_file['data']:
            quac_vocab(file)
        return quac_vocab

    def _convertToken(self, text, object='q'):
        tmp = []
        for x in text.lower().split():
            tmp.append(self.vocab.word2index[x])
        tmp.append(self.vocab.word2index['[EOS]'])
        tmp.insert(0, self.vocab.word2index['[SOS]'])
        return tmp


def collate_fn(batch):
    (contexts, questions, answers) = list(zip(*batch))
    contexts_pad = torch.nn.utils.rnn.pad_sequence(contexts)
    questions_pad = torch.nn.utils.rnn.pad_sequence(questions)
    answers_pad = torch.nn.utils.rnn.pad_sequence(answers)

    return contexts_pad, questions_pad, answers_pad

if __name__ == "__main__":
    main()