#%%
import matplotlib.pyplot as plt

from bigram_nn import BigramNN

#%%
bigram_nn = BigramNN()

#%%
x, y, num = bigram_nn.one_hot_bigram_dataset()

print(x.shape)
print(x[:5])
plt.imshow(x[:5])

# %%
model = bigram_nn.train_model()

#%% 
samples = bigram_nn.sample_from_nn(10)
samples
