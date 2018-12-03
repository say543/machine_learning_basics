
#https://github.com/soheillll/Parts-of-Speech-Tagging/blob/master/HMM%20Tagger.ipynb


import numpy as np

import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
np.random.seed(123)

# install matplotlib
# this line for ipython mac only
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ? comment them at first since i do not think they are necessary 
# Jupyter "magic methods" -- only need to be run once per kernel restart
#%load_ext autoreload
#%aimport helpers, tests
#%autoreload 1

from itertools import chain
from collections import Counter, defaultdict
from helpers import show_model, Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

from IPython.core.display import HTML
