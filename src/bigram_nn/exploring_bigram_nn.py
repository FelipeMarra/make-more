#%%
import matplotlib.pyplot as plt

from bigram_nn import BigramNN

#%%
bigram_nn = BigramNN()

#%%
x, y = bigram_nn.one_hot_bigram_dataset()

print(x[:5])
plt.imshow(x[:5])

# %%
