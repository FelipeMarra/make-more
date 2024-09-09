from typing import Dict, Tuple, List
import os

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

################### Constants ###################
NAMES_PATH = "../../data/names.txt"
START_TOKEN = '.'
END_TOKEN = '.'
MODEL_COUNT_SMOOTH = 1 # fake count to avoid -inf log likelihood

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

    def one_hot_bigram_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
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

        x = F.one_hot(x, num_classes=self.n_chars).float()
        y = F.one_hot(y, num_classes=self.n_chars).float()

        return x, y 
