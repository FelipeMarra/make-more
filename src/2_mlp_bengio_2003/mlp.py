from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F

################### Constants ###################
NAMES_PATH = "../../data/names.txt"

START_TOKEN = '.'
END_TOKEN = '.'

SEED = 2147483647
EPOCHS = 200
LR = 50
CONTEXT_SIZE = 3 # how many letters to predict the next one

class MLP():
    def __init__(self, verbose=False) -> None:
        self.words_list = self.read_words()
        self.char2idx, self.idx2char, self.n_chars = self.char2idx_idx2char()
        self.verbose = verbose

    ################### Pre-processing ###################
    def read_words(self, path=NAMES_PATH) -> List[str]:
        return open(path, 'r').read().splitlines()

    def char2idx_idx2char(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        words_list = [START_TOKEN] + self.words_list + [END_TOKEN]
        char_list = sorted(list(set(''.join(words_list))))

        char2idx = {c:i for i,c in enumerate(char_list)}
        idx2char = {i:c for c,i in char2idx.items()}

        return char2idx, idx2char, len(char_list)

    def get_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.char2idx[START_TOKEN]
        X, Y = [], []

        for word in self.words_list[:5]:
            if self.verbose: print('\n'+word)
            context = [start_idx] * CONTEXT_SIZE

            for char in word + END_TOKEN:
                idx = self.char2idx[char]
                X.append(context)
                Y.append(idx)

                if self.verbose:
                    print(''.join(self.idx2char[ctxt_idx] for ctxt_idx in context), '---->', char)

                context = context[1:] + [idx]

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y