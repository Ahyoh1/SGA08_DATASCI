# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import pandas as pd

# %% [markdown]
# # Import dataset

# %%
cars= pd.read_csv("mtcars.csv")

# %% [markdown]
# # To view the first five rows of the dataset

# %%
cars. head()

# %% [markdown]
# # To get summary of the dataset

# %%
cars.describe()

# %% [markdown]
# # To identify the data types in the dataset

# %%
cars.dtypes


# %%
import matplotlib.pyplot as plt

# %% [markdown]
# # Univariate analysis
# %% [markdown]
# ## Line graph showing the miles per gallon of each model

# %%
cars.plot("model","mpg", kind="line", legend= False, color="Purple", figsize=(12,8))
plt.sort=("model")
plt.xlabel(xlabel="Model")
plt.ylabel(ylabel="Mile per gallon")
plt.title ("Miles per gallon of each model")

# %% [markdown]
# # Bivariate analysis
# %% [markdown]
# ## Scatter plot showing the relationship between car weight and miles per gallon

# %%
cars.plot.scatter(x="wt",y="mpg", figsize=(8,6))
plt.title("Relationship between car weight and miles per gallon")

# %% [markdown]
# ## Scatter plot showing relationship between Miles per gallon and time of initial acceleration

# %%
cars.plot.scatter (x="mpg",y="qsec", figsize= (8,6))
plt.title ("Relationship between Miles per gallon and time of initial acceleration")

# %% [markdown]
# # Multivariate analysis

# %%
import seaborn as sns


# %%
sns.pairplot (cars,vars=["cyl","disp","hp","drat","gear","carb"])


# %%


