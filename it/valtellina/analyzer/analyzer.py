import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ucimlrepo import fetch_ucirepo

class Analyzer:

    def __init__(self, dataset_id=9):

        # download dataset
        dataset = fetch_ucirepo(id=dataset_id)

        # creo dataframe completo
        self.df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)




    def show_head(self):
        print(self.df.head())