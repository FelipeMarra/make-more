#%%
import os
import sys 
sys.path.append(os.path.pardir)

from makemore import *

# %%
words_list = read_dataset()
words_list[:10]

# %%
len(words_list)

# %%
min(len(w) for w in words_list)

# %%
max(len(w) for w in words_list)

#%%
# Bigram takes the last letter and predics the next, using pairs like:
for w in words_list[:5]:
    for ch1, ch2 in zip(w, w[1:]):
        print(ch1, ch2)
    print()

# %%
# Of course we need our start and end tokens represented by '.'
for w in words_list[:4]:
    chrs = [START_TOKEN] + list(w) + [END_TOKEN]
    for ch1, ch2 in zip(chrs, chrs[1:]):
        print(ch1, ch2)
    print()

# %%
# Now counting our bigrams
bigrams_dict = bigram_count_dict(words_list[:4])
print(bigrams_dict)

# count for the whole list
bigrams_dict = bigram_count_dict(words_list)

# %%
char2idx, idx2char = char2idx_idx2char(words_list)
char2idx

#%%
idx2char

# %%
bigram_tensor = bigram_count_tensor(words_list, char2idx)
bigram_tensor

# %%
plot_bigram_tensor(bigram_tensor, idx2char)

# %%
# Normalize the bigram tensor for the first letter
# to obtain it's probability distribution
prob_distr = bigram_tensor[0].float() 
prob_distr = prob_distr/torch.sum(prob_distr)
prob_distr

# %%
## Sampling from the distribution
generator = torch.Generator().manual_seed(2147483647)
sampled_idx = torch.multinomial(prob_distr, num_samples=1, 
                                replacement=True,generator=generator).item()
print(sampled_idx)
idx2char[sampled_idx]

# %%
samples = sample_from_bigram(bigram_tensor, idx2char, char2idx, 10)
samples

# %%
