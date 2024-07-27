################### Imports ###################
from typing import Dict, Tuple
import os

import torch

import matplotlib.pyplot as plt

################### Constants ###################
PATH = os.path.join(os.path.pardir, "data/names.txt")
N_CHARS = 28 # 26 from the alphabet + start and end tokens
START_TOKEN = '<S>'
END_TOKEN = '<E>'

################### Pre-processing ###################
def read_dataset(path=PATH) -> list[str]:
    return open(path, 'r').read().splitlines()

def bigram_count_dict(words_list:list[str]):
    bigrams_dict = {}

    for w in words_list:
        chrs = ['<S>'] + list(w) + ['<E>']
        for ch1, ch2 in zip(chrs, chrs[1:]):
            bigram = (ch1, ch2)
            bigrams_dict[bigram] = bigrams_dict.get(bigram, 0) +1

    return bigrams_dict

def char2idx_idx2char(words_list:list[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    char_list = sorted(list(set(''.join(words_list))))
    char_list = char_list + [START_TOKEN, END_TOKEN]

    char2idx = {c:i for i,c in enumerate(char_list)}
    idx2char = {i:c for c,i in char2idx.items()}

    return char2idx, idx2char

def bigram_count_tensor(words_list:list[str], char2idx:Dict[str, int]) -> torch.Tensor:
    bigram_tensor = torch.zeros((N_CHARS, N_CHARS), dtype=torch.int32)

    for w in words_list:
        chrs = ['<S>'] + list(w) + ['<E>']
        for ch1, ch2 in zip(chrs, chrs[1:]):
            pos1 = char2idx[ch1]
            pos2 = char2idx[ch2]

            bigram_tensor[pos1, pos2] += 1

    return bigram_tensor

def plot_bigram_tensor(bigram_tensor:torch.Tensor, idx2char:Dict[int, str]) -> None:
    """
        Plot a matrix where each cell have the form
            xy
            10
        mening that x follows y 10 times
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