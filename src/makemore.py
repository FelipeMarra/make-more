################### Imports ###################
from typing import Dict, Tuple, List
import os

import torch

import matplotlib.pyplot as plt

################### Constants ###################
PATH = os.path.join(os.path.pardir, "data/names.txt")
N_CHARS = 27 # 26 from the alphabet + '.' that represents start and end tokens
START_END_TOKEN = '.'

################### Pre-processing ###################
def read_dataset(path=PATH) -> list[str]:
    return open(path, 'r').read().splitlines()

def bigram_count_dict(words_list:list[str]):
    bigrams_dict = {}

    for w in words_list:
        chrs = [START_END_TOKEN] + list(w) + [START_END_TOKEN]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            bigram = (ch1, ch2)
            bigrams_dict[bigram] = bigrams_dict.get(bigram, 0) +1

    return bigrams_dict

def char2idx_idx2char(words_list:list[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    char_list = sorted(list(set(''.join(words_list))))
    char_list = [START_END_TOKEN] + char_list

    char2idx = {c:i for i,c in enumerate(char_list)}
    idx2char = {i:c for c,i in char2idx.items()}

    return char2idx, idx2char

def bigram_count_tensor(words_list:list[str], char2idx:Dict[str, int]) -> torch.Tensor:
    """
        A matrix where each cell have the form
            matrix[i, j] = 10
        mening that j comes after i 10 times
    """
    bigram_tensor = torch.zeros((N_CHARS, N_CHARS), dtype=torch.int32)

    for w in words_list:
        chrs = [START_END_TOKEN] + list(w) + [START_END_TOKEN]
        for ch1, ch2 in zip(chrs, chrs[1:]):
            pos1 = char2idx[ch1]
            pos2 = char2idx[ch2]

            bigram_tensor[pos1, pos2] += 1

    return bigram_tensor

def plot_bigram_tensor(bigram_tensor:torch.Tensor, idx2char:Dict[int, str]) -> None:
    """
        Plot a matrix where each cell have the form
            ij
            10
        mening that j comes after i 10 times
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

def sample_from_bigram(bigram_tensor:torch.Tensor, idx2char:Dict[int, str], num_samples:int=1) -> List[str]:
    generator = torch.Generator().manual_seed(2147483647)
    out = []

    for _ in range(num_samples):
        name = []
        sample_idx = 0 # Always start from start token
        
        while True:
            prob_distrib = bigram_tensor[sample_idx].float()
            prob_distrib = prob_distrib / prob_distrib.sum()

            sample_idx = torch.multinomial(prob_distrib, num_samples=1, 
                                        replacement=True, generator=generator).item()
            sample = idx2char[sample_idx]

            # Stop generation if end token
            if sample_idx == 0:
                break

            name.append(sample)

        out.append(''.join(name))

    return out