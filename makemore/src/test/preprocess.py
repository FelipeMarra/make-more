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