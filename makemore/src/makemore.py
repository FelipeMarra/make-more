################### Imports ###################
#%%
import os

################### Constantes ###################
PATH = os.path.join(os.path.pardir, "data/names.txt")

################### Pre-processing ###################
#### Read dataset

def read_dataset(path=PATH) -> list[str]:
    return open(path, 'r').read().splitlines()


# %%
