################### Imports ###################
from typing import Dict, Tuple, List
import os

import torch

import matplotlib.pyplot as plt

################### Constants ###################
NAMES_PATH = os.path.join(os.path.pardir, "data/names.txt")
START_TOKEN = '.'
END_TOKEN = '.'

################### Pre-procSTART_END_TOKENessing ###################
def read_dataset(path=NAMES_PATH) -> list[str]:
    return open(path, 'r').read().splitlines()

def bigram_count_dict(words_list:list[str]):
    bigrams_dict = {}

    for w in words_list:
        chrs = [START_TOKEN] + list(w) + [END_TOKEN]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            bigram = (ch1, ch2)
            bigrams_dict[bigram] = bigrams_dict.get(bigram, 0) +1

    return bigrams_dict

def char2idx_idx2char(words_list:list[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    words_list = [START_TOKEN] + words_list + [END_TOKEN]
    char_list = sorted(list(set(''.join(words_list))))

    char2idx = {c:i for i,c in enumerate(char_list)}
    idx2char = {i:c for c,i in char2idx.items()}

    return char2idx, idx2char

def bigram_count_tensor(words_list:list[str], char2idx:Dict[str, int]) -> torch.Tensor:
    """
        A matrix where each cell have the form
            matrix[b, a] = 10
        mening that a comes after b 10 times
    """
    n_chars = len(char2idx)
    bigram_tensor = torch.zeros((n_chars, n_chars), dtype=torch.int32)

    for w in words_list:
        chrs = [START_TOKEN] + list(w) + [END_TOKEN]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            pos1 = char2idx[ch1]
            pos2 = char2idx[ch2]

            bigram_tensor[pos1, pos2] += 1

    return bigram_tensor

def plot_bigram_tensor(bigram_tensor:torch.Tensor, idx2char:Dict[int, str]) -> None:
    """
        Plot a matrix where each cell have the form
            ba
            10
        mening that a comes after b 10 times
    """
    # Diplays the array
    plt.figure(figsize=(16, 16))
    plt.imshow(bigram_tensor, cmap='Blues')

    # For each cell, display the bigram and its count
    shape = bigram_tensor.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            bigram_str = idx2char[i] + idx2char[j]
            bigram_count = bigram_tensor[i,j].item()

            plt.text(j, i, bigram_str, ha='center', va='bottom', color='gray')
            plt.text(j, i, bigram_count, ha='center', va='top', color='gray')

    plt.axis('off')

def get_bigram_probability_matrix(bigram_tensor:torch.Tensor):
    prob_matrix = bigram_tensor.float()
    # /= helps not creating new memory
    # keepdim=True because off broadcast 
    # https://pytorch.org/docs/stable/notes/broadcasting.html
    prob_matrix /= prob_matrix.sum(1, keepdim=True)

    return prob_matrix

def sample_from_bigram(bigram_tensor:torch.Tensor, idx2char:Dict[int, str], char2idx:Dict[int, str], num_samples:int=1) -> List[str]:
    generator = torch.Generator().manual_seed(2147483647)
    out = []

    probability_matrix = get_bigram_probability_matrix(bigram_tensor)

    for _ in range(num_samples):
        name = []
        sample_idx = char2idx[START_TOKEN] # Always start from start token
        
        while True:
            prob_distrib = probability_matrix[sample_idx]

            sample_idx = torch.multinomial(prob_distrib, num_samples=1, 
                                        replacement=True, generator=generator).item()
            sample = idx2char[sample_idx]

            # Stop generation if end token
            if sample_idx == char2idx[END_TOKEN]:
                break

            name.append(sample)

        out.append(''.join(name))

    return out