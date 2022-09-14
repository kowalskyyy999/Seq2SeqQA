import collections
import pandas as pd
from tqdm import tqdm

class Vocabulary(object):
    def __init__(self, specials=True):
        self.word2index= {}
        self.index2word = {}
        self.word2count= {}
        self.num_words = 6
        self.num_sentences = 0
        self.longest_sentences = 0
        if specials:
            for k, v in zip(['[PAD]', '[CLS]', '[MASK]', '[UNK]', '[SOS]', '[EOS]'], [0, 1, 2, 3, 4, 5]):
                self.index2word[v] = k
                self.word2index[k] = v

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentences(self, sentence):
        sentence_len = 0
        for word in sentence.split(" "):
            sentence_len += 1
            self.add_word(word)
        
        if sentence_len > self.longest_sentences:
            self.longest_sentences = sentence_len
        
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

class QuAC_Vocab(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.max_questions = 0
        self.max_answers = 0
        self.max_contexts = 0

    def title_vocab(self, file, vocab):

        if len(file['title'].lower().split(" ")) > 1:
            vocab.add_sentences(file['title'].lower())
        else:
            vocab.add_word(file['title'].lower())

        return vocab

    def bg_vocab(self, file, vocab):
        vocab.add_sentences(file['background'].lower())
        return vocab

    def section_vocab(self, file, vocab):
        vocab.add_sentences(file['section_title'].lower())
        return vocab

    def paragraphs_vocab(self, file, vocab):
        for parag in file['paragraphs']:
            if self.max_contexts < len(parag['context'].lower().split()):
                self.max_contexts = len(parag['context'].lower().split())
            vocab.add_sentences(parag['context'].lower())

            for q in parag['qas']:
                if self.max_questions < len(q['question'].lower().split()):
                    self.max_questions = len(q['question'].lower().split())
                vocab.add_sentences(q['question'].lower())

                for a in q['answers']:
                    if self.max_answers < len(a['text'].lower().split()):
                        self.max_answers = len(a['text'].lower().split())
                    vocab.add_sentences(a['text'].lower())
                    
        return vocab

    def __call__(self, file):
        self.vocab = self.title_vocab(file, self.vocab)
        self.vocab = self.bg_vocab(file, self.vocab)
        self.vocab = self.section_vocab(file, self.vocab)
        self.vocab = self.paragraphs_vocab(file, self.vocab)

def parseJson(json_file):
    datas = tqdm(json_file['data'], total=len(json_file['data']))
    df = collections.defaultdict(list)
    for file in datas:
        for paragraph in file['paragraphs']:
            for qa in paragraph['qas']:
                for a in qa['answers']:
                    df['context'].append(paragraph['context'])
                    df['question'].append(qa['question'])
                    df['answers'].append(a['text'])
    return pd.DataFrame(df)

def count_parameters(model):
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Size of Seq2Seq model parameters : {parameters/1000000:.2f} M")


def main():
    
    text = "Enak mantap sekali"
    vocab = Vocabulary()
    vocab.add_sentences(text)

if __name__ == "__main__":
    main()
