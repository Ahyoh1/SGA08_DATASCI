# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import datetime
import random


# %%
df= pd.DataFrame({'Year':np.random.randint(2010,2020, size=10),'Count':np.random.randint(1,10,size=10)})
df


# %%
for ind, row in df.iterrows():
    df.loc[ind,'Average Count']= row['Count']/row ['Count']


# %%
df


# %%
#Count=df.groupby('Year').count()

