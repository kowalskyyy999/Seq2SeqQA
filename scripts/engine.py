import os
import torch
from tqdm import tqdm
import logging

class Engine(object):
    def __init__(self, vocab, model, optimizer, criterion, epochs, device='cpu'):
        self.model = model.to(device)
        self.vocab = vocab
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = device

    def fit(self, trainLoader, valLoader=None):
        for epoch in range(self.epochs):
            self.model.train()
            trainLoss = 0
            tk = tqdm(trainLoader, total=len(trainLoader))
            for c, q, a in tk:
                self.optimizer.zero_grad()

                out = self.model(c.to(self.device), q.to(self.device), a.to(self.device))

                out = out[1:].reshape(-1, out.shape[2])
                a = a[1:].reshape(-1)

                loss = self.criterion(out, a.to(self.device))
                loss.backward()

                self.optimizer.step()

                tk.set_postfix({'Epoch': epoch+1, 'Training Loss': loss.item()})

                trainLoss += loss.item() * c.size(1)

            trainLoss = trainLoss / len(trainLoader.dataset)

            if valLoader is not None:
                self.validation(valLoader, epoch)


    def validation(self, valLoader, epoch=None):
        valLoss = 0
        tk = tqdm(valLoader, total=len(valLoader))
        with torch.no_grad():
            self.model.eval()
            for c, q, a in tk:

                out = self.model(c.to(self.device), q.to(self.device), a.to(self.device))
                out = out[1:].reshape(-1, out.shape[2])
                a = a[1:].reshape(-1)

                loss = self.criterion(out, a.to(self.device))

                if epoch is not None:
                    tk.set_postfix({'Epoch': epoch + 1, 'Validation Loss': loss.item()})
                else:
                    tk.set_postfix({'Validation Loss': loss.item()})
                
                valLoss += loss.item() * c.size(1)

        valLoss = valLoss / len(valLoader.dataset)

    def save_model(self, checkpoint, nameModel, path='models'):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(checkpoint, os.path.join(path, nameModel))
        logging.info(f"Save the model with name '{nameModel}'")