import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from read_data import DataFeatures
import os, sys
sys.path.append('../')
import util

x = util.load_pkl('weebitfeatures.pkl')
print(x.tfidf)
#TODO make this a jupyter notebook




