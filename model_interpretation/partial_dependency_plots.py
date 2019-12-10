import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

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
                                   features=[0, 2], # column numbers of plots we want to show
                                   X=imputed_X,            # raw predictors data.
                                   feature_names=['Distance', 'Landsize', 'BuildingArea'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
plt.show()

