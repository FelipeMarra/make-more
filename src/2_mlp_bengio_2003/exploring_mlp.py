#%%
import torch
import matplotlib.pyplot as plt

from mlp import *

#%%
mlp = MLP(verbose=False)

#%%
X, Y = mlp.get_dataset(mlp.words_list)

#%%
X.shape, X.dtype, Y.shape, Y.dtype

# %%
print('X:\n', X[:10, :])
print('Y:\n', Y[:10])

# %%
C = torch.randn((27, 2)).cuda()
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

#plt.plot(exps, losses)

# %%
steps, losses = mlp.train()

#%%
#plt.plot(range(steps), losses)

# %%
# evall
mlp.forward(mlp.X_eval, mlp.Y_eval)

# %%
# test
mlp.forward(mlp.X_test, mlp.Y_test)

# %%
# Viz 2D embeddings
if EMBEDDING_DIMS == 2:
    plt.figure(figsize=(8,8))
    plt.scatter(mlp.C[:,0].data.detach().cpu(), mlp.C[:,1].data.detach().cpu(), s=200)
    for i in range(mlp.C.shape[0]):
        plt.text(mlp.C[i,0].item(), mlp.C[i,1].item(), mlp.idx2char[i], ha='center', va='center', color='white')
    plt.grid('minor')


# %%
