import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import (partial_dependence,
                                                 plot_partial_dependence)
from sklearn.preprocessing import Imputer

cols_to_use = ['Distance', 'Landsize', 'BuildingArea']
data = pd.read_csv('data/melb_house_pricing.csv')
# drop rows where target is NaN
data = data.loc[~(data['Price'].isna())]
y = data.Price
X = data[cols_to_use]
my_imputer = Imputer()
imputed_X = my_imputer.fit_transform(X)

print(f"Contains NaNs in training data: {np.isnan(imputed_X).sum()}")
print(f"Contains NaNs in target data: {np.isnan(y).sum()}")
print(f"Contains Infinity: {np.isinf(imputed_X).sum()}")
print(f"Contains Infinity: {np.isinf(y).sum()}")

my_model = GradientBoostingRegressor()
my_model.fit(imputed_X, y)

# Here we make the plot
my_plots = plot_partial_dependence(my_model,
                                   # column numbers of plots we want to show
                                   features=[0, 2],
                                   # raw predictors data.
                                   X=imputed_X,
                                   # labels on graphs
                                   feature_names=['Distance',
                                                  'Landsize', 'BuildingArea'],
                                   grid_resolution=10)  # number of values to plot on x axis
plt.show()
