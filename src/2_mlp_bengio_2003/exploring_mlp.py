#%%
import matplotlib.pyplot as plt

from mlp import MLP

#%%
mlp = MLP(verbose=True)

#%%
X, Y = mlp.get_dataset()

#%%
X.shape, X.dtype, Y.shape, Y.dtype

# %%
print('X:\n', X[:10, :])
print('Y:\n', Y[:10])

# %%
