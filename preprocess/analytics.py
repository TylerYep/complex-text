import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_data import DataFeatures
import os, sys
sys.path.append('../')
import util
# import featexp

data_features = util.load_pkl('preprocess/weebit_features.pkl')
print(data_features.raw.loc[data_features.raw['level'] == 4])
#
# featexp.get_univariate_plots(data=data_features.nl_matrix, target_col='target',
#                      features_list=['word count'], bins=10)
