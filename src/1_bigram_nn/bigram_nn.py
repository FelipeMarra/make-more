from typing import Dict, Tuple, List
import os

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

################### Constants ###################
NAMES_PATH = "../../data/names.txt"

START_TOKEN = '.'
END_TOKEN = '.'

SEED = 2147483647
EPOCHS = 200
LR = 50

class BigramNN():
    def __init__(self) -> None:
        self.words_list = self.read_words()
        self.char2idx, self.idx2char, self.n_chars = self.char2idx_idx2char()

    ################### Pre-processing ###################
    def read_words(self, path=NAMES_PATH) -> List[str]:
        return open(path, 'r').read().splitlines()

    def char2idx_idx2char(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        words_list = [START_TOKEN] + self.words_list + [END_TOKEN]
        char_list = sorted(list(set(''.join(words_list))))

        char2idx = {c:i for i,c in enumerate(char_list)}
        idx2char = {i:c for c,i in char2idx.items()}

        return char2idx, idx2char, len(char_list)

    def one_hot_bigram_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x, y = [], []

        for w in self.words_list:
            chrs = [START_TOKEN] + list(w) + [END_TOKEN]
            for ch1, ch2 in zip(chrs, chrs[1:]):
                idx1 = self.char2idx[ch1]
                idx2 = self.char2idx[ch2]

                x.append(idx1)
                y.append(idx2)

        x = torch.tensor(x)
        y = torch.tensor(y)
        num = x.nelement()

        x = F.one_hot(x, num_classes=self.n_chars).float()

        return x, y, num

    def train_model(self) -> torch.Tensor:
        x, y, num = self.one_hot_bigram_dataset()

        g = torch.Generator().manual_seed(SEED)
        W = torch.randn((self.n_chars, self.n_chars), 
                        generator=g, 
                        requires_grad=True)
        
        print(f"x: {x.shape}; y: {y.shape}; W: {W.shape}")

        for _ in range(EPOCHS):
            # Forward 
            logits = x @ W # predict "log-counts"
            ## Softmax:
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)

            # Backward
            ## Negative log likelihood
            regularization_loss = 0.01*(W**2).mean()
            loss = -probs[torch.arange(num), y].log().mean() + regularization_loss
            print(loss.item())

            ## Gradient
            W.grad = None
            loss.backward()
            ## Gradient descent
            W.data += -LR * W.grad

        self.model = W

    def sample_from_nn(self, num_samples:int=1) -> List[str]:
        generator = torch.Generator().manual_seed(2147483647)
        out = []

        for _ in range(num_samples):
            name = []
            sample_idx = self.char2idx[START_TOKEN] # Always start from start token
            
            while True:
                x = F.one_hot(torch.tensor(sample_idx), num_classes=27).float()
                logits = x @ self.model
                softmax = F.softmax(logits)

                sample_idx = torch.multinomial(softmax, num_samples=1, 
                                            replacement=True, generator=generator).item()
                sample = self.idx2char[sample_idx]

                # Stop generation if end token
                if sample_idx == self.char2idx[END_TOKEN]:
                    break

                name.append(sample)

            out.append(''.join(name))

        return out
