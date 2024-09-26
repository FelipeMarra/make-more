from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F

################### Constants ###################
NAMES_PATH = "../../data/names.txt"

START_TOKEN = '.'
END_TOKEN = '.'

SEED = 2147483647
EPOCHS = 10000
LR = 1e-1
BATCH_SIZE = 512
CONTEXT_SIZE = 3 # how many letters will be used to predict the next one
EMBEDDING_DIMS = 2 # how many dims in our embedding space

class MLP():
    def __init__(self, verbose=False) -> None:
        self.generator = torch.Generator().manual_seed(SEED)
        self.words_list = self.read_words()
        self.char2idx, self.idx2char, self.n_chars = self.char2idx_idx2char()
        self.verbose = verbose

        self.X, self.Y = self.get_dataset()
        self.set_params()

    ################### Pre-processing #######################################################
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

        for word in self.words_list:
            #if self.verbose: print('\n'+word)
            context = [start_idx] * CONTEXT_SIZE

            for char in word + END_TOKEN:
                idx = self.char2idx[char]
                X.append(context)
                Y.append(idx)

                #if self.verbose:
                #    print(''.join(self.idx2char[ctxt_idx] for ctxt_idx in context), '---->', char)

                context = context[1:] + [idx]

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y

    ################### Model #######################################################
    def set_params(self):
        # The model params
        self.C = torch.randn((self.n_chars, EMBEDDING_DIMS)) # emmbedings lookup table

        # This layer receives the embeddings. 
        # Thus its first dim will be of size CONTEXT_SIZE*EMBEDDING_DIMS.
        # The second one is how many neurons we want in it
        self.W1 = torch.randn((CONTEXT_SIZE*EMBEDDING_DIMS, 100), generator=self.generator)
        self.b1 = torch.randn(100, generator=self.generator)

        self.W2 = torch.randn((100, self.n_chars), generator=self.generator)
        self.b2 = torch.randn(self.n_chars, generator=self.generator)

        self.params = [self.C, self.W1, self.b1, self.W2, self.b2]


        if self.verbose:
            n_params = sum(p.nelement() for p in self.params)
            print("Model params:", n_params)

        for p in self.params:
            p.requires_grad = True

    def forward(self):
        mini_batch_idxs = torch.randint(0, self.X.shape[0], (BATCH_SIZE,), generator=self.generator)

        embeddings = self.C[self.X[mini_batch_idxs]] # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_DIMS) 
        emb_view = embeddings.view(-1, CONTEXT_SIZE*EMBEDDING_DIMS) # (BATCH_SIZE, CONTEXT_SIZE*EMBEDDING_DIMS)

        hidden = torch.tanh(emb_view @ self.W1 + self.b1)

        logits = hidden @ self.W2 + self.b2

        # counts = logits.exp()
        # probs = counts / counts.sum(1, keepdims=True)
        # probs of the correct chars
        # chars_probs = probs[torch.arange(probs.shape[0]), Y]
        # loss = - chars_probs.log().mean()

        loss = F.cross_entropy(logits, self.Y[mini_batch_idxs])
        return loss

    ################### Train #######################################################
    def train(self):
        for epoch in range(EPOCHS):
            loss = self.forward()

            if self.verbose: print('Epoch:', epoch, 'Loss', loss.item())

            for p in self.params:
                p.grad = None

            loss.backward()

            for p in self.params:
                p.data += -LR * p.grad

        print('Final Loss', loss.item())

    def find_lr(self):
        exponents = torch.linspace(-3, 0, 1000)

        losses = []
        for exp in exponents:
            lr = 10**exp
            loss = self.forward()

            if self.verbose: print('LR:', lr, 'Loss', loss.item())
            losses.append(loss.item())

            for p in self.params:
                p.grad = None

            loss.backward()

            for p in self.params:
                p.data += -lr * p.grad

        # Reset the model
        self.set_params()

        return exponents, losses