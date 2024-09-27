import random
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F

################### Constants ###################
NAMES_PATH = "../../data/names.txt"

START_TOKEN = '.'
END_TOKEN = '.'

SEED = 2147483647
RANDOM_SEED = 42

# TODO: Scheduler?
# First
EPOCH_LIST = [50000, 10000]
LR = [10**-0.5, 1e-2]

BATCH_SIZE = 512
CONTEXT_SIZE = 3 # how many letters will be used to predict the next one
EMBEDDING_DIMS = 10 # how many dims in our embedding space

L1_NEURONS = 200

class MLP():
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        self.generator = torch.Generator('cuda').manual_seed(SEED)
        random.seed(RANDOM_SEED)

        self.words_list = self.read_words()
        random.shuffle(self.words_list)

        self.char2idx, self.idx2char, self.n_chars = self.char2idx_idx2char()

        # Get train, eval and test sets
        s1 = int(0.8 * len(self.words_list))
        s2 = int(0.9 * len(self.words_list))
        self.X_train, self.Y_train = self.get_dataset(self.words_list[:s1])
        self.X_eval, self.Y_eval = self.get_dataset(self.words_list[s1:s2])
        self.X_test, self.Y_test = self.get_dataset(self.words_list[s2:])

        # Init nn params
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

    def get_dataset(self, words_list) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.char2idx[START_TOKEN]
        X, Y = [], []

        for word in words_list:
            #if self.verbose: print('\n'+word)
            context = [start_idx] * CONTEXT_SIZE

            for char in word + END_TOKEN:
                idx = self.char2idx[char]
                X.append(context)
                Y.append(idx)

                #if self.verbose:
                #    print(''.join(self.idx2char[ctxt_idx] for ctxt_idx in context), '---->', char)

                context = context[1:] + [idx]

        X = torch.tensor(X).cuda()
        Y = torch.tensor(Y).cuda()
        print(f"Dataset size X:{X.shape}, {X.device}, Y:{Y.shape}, {Y.device}")

        return X, Y

    ################### Model #######################################################
    def set_params(self):
        # The model params
        self.C = torch.randn((self.n_chars, EMBEDDING_DIMS), device='cuda') # emmbedings lookup table

        # This layer receives the embeddings. 
        # Thus its first dim will be of size CONTEXT_SIZE*EMBEDDING_DIMS.
        # The second one is how many neurons we want in it
        self.W1 = torch.randn((CONTEXT_SIZE*EMBEDDING_DIMS, L1_NEURONS), generator=self.generator, device='cuda')
        self.b1 = torch.randn(L1_NEURONS, generator=self.generator, device='cuda')

        self.W2 = torch.randn((L1_NEURONS, self.n_chars), generator=self.generator, device='cuda')
        self.b2 = torch.randn(self.n_chars, generator=self.generator, device='cuda')

        self.params = [self.C, self.W1, self.b1, self.W2, self.b2]

        print("Model params:", sum(p.nelement() for p in self.params))

        for p in self.params:
            p.requires_grad = True

    def forward(self, X, Y):
        mini_batch_idxs = torch.randint(0, X.shape[0], (BATCH_SIZE,), generator=self.generator, device='cuda')

        embeddings = self.C[X[mini_batch_idxs]] # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_DIMS) 
        emb_view = embeddings.view(-1, CONTEXT_SIZE*EMBEDDING_DIMS) # (BATCH_SIZE, CONTEXT_SIZE*EMBEDDING_DIMS)

        hidden = torch.tanh(emb_view @ self.W1 + self.b1)

        logits = hidden @ self.W2 + self.b2

        # counts = logits.exp()
        # probs = counts / counts.sum(1, keepdims=True)
        # probs of the correct chars
        # chars_probs = probs[torch.arange(probs.shape[0]), Y]
        # loss = - chars_probs.log().mean()

        loss = F.cross_entropy(logits, Y[mini_batch_idxs])
        return loss

    ################### Train #######################################################
    def find_lr(self):
        exponents = torch.linspace(-3, 0, 1000)

        losses = []
        for exp in exponents:
            lr = 10**exp
            loss = self.forward(self.X_train, self.Y_train)

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

    def train(self):
        losses = []
        steps = 0

        for e_idx, epochs in enumerate(EPOCH_LIST):
            for epoch in range(epochs):
                loss = self.forward(self.X_train, self.Y_train)

                if self.verbose: print('Epoch:', epoch, 'Loss', loss.item())
                steps += 1
                losses.append(loss.log10().item())

                for p in self.params:
                    p.grad = None

                loss.backward()

                for p in self.params:
                    p.data += -LR[e_idx] * p.grad

        print('Final Train Loss', loss.item())
        return steps, losses

    ################### Sample #######################################################
    def sample(self, n_samples):
        generator = torch.Generator('cuda').manual_seed(SEED +10)

        words = []
        for _ in range(n_samples):
            word = []    
            context = [self.char2idx[START_TOKEN]] * CONTEXT_SIZE

            while True:
                embeddings = self.C[torch.tensor([context])] # (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_DIMS) 
                emb_view = embeddings.view(1, -1) # (BATCH_SIZE, CONTEXT_SIZE*EMBEDDING_DIMS)

                hidden = torch.tanh(emb_view @ self.W1 + self.b1)

                logits = hidden @ self.W2 + self.b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
                context = context[1:] + [ix]
                word.append(ix)

                if ix == self.char2idx[END_TOKEN]:
                    words.append(''.join(self.idx2char[i] for i in word))
                    break

        return words