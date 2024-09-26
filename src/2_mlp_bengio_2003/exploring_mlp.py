#%%
import torch
import matplotlib.pyplot as plt

from mlp import MLP

#%%
mlp = MLP(verbose=False)

#%%
X, Y = mlp.get_dataset()

#%%
X.shape, X.dtype, Y.shape, Y.dtype

# %%
print('X:\n', X[:10, :])
print('Y:\n', Y[:10])

# %%
C = torch.randn((27, 2))
# %%
C[
    torch.tensor(
        [
            [ 0,  0,  0],
            [ 0,  0,  5],
            [ 0,  5, 13],
            [ 5, 13, 13],
            [13, 13,  1],
            [ 0,  0,  0]
        ]
    )
]

# %%
print(C[X].shape, '\n', C[X][:10, :, :])

#%%
X[13, 2], C[1], C[X][13, 2]

#%%
exps, losses = mlp.find_lr()

plt.plot(exps, losses)

# %%
mlp.train()

# %%
