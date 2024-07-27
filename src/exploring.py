#%%
# chaticie do python
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
# Of course we need our start and end tokens
for w in words_list[:4]:
    chrs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chrs, chrs[1:]):
        print(ch1, ch2)
    print()

# %%
# Now counting our gigrams
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
