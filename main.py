import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera

class Main:

    def __init__(self, df):
        self.df = df.copy()

    # ----------------------------
    # INFO BASE
    # ----------------------------
    def overview(self):
        print("Shape:", self.df.shape)
        print("\nInfo:")
        print(self.df.info())

    # ----------------------------
    # GESTIONE DUPLICATI
    # ----------------------------

    def remove_duplicates(self):
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]
        print(f"Duplicati trovati e rimossi: {before - after}")

    # ----------------------------
    # OUTLIERS (IQR METHOD)
    # ----------------------------
    def detect_outliers(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = self.df[
            (self.df[column] < lower) |
            (self.df[column] > upper)
            ]

        print(f"Outliers in {column}: {len(outliers)}")
        count_outliers = len(outliers)
        total = len(self.df)
        perc_outliers = (count_outliers / total) * 100
        print(f"Outliers percentage: {perc_outliers:.2f}%")

    def remove_outliers(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        self.df = self.df[
            (self.df[column] >= lower) &
            (self.df[column] <= upper)
            ]

        print(f"Outliers rimossi da {column}")

    # ----------------------------
    # DISTRIBUZIONE
    # ----------------------------
    def skewness(self):
        numeric = self.df.select_dtypes(include='number')
        print(numeric.skew().sort_values(ascending=False))

    def kurtosis(self):
        numeric = self.df.select_dtypes(include='number')
        print(numeric.kurtosis().sort_values(ascending=False))

    def jarque_bera_test(self):
        numeric = self.df.select_dtypes(include='number')
        results = {}
        for col in numeric.columns:
            stat, p = jarque_bera(numeric[col].dropna())
            results[col] = {
                "statistic": stat,
                "p_value": p,
                "normal": p > 0.05
            }
        return results

    # ----------------------------
    # PLOTS (per colonna)
    # ----------------------------
    def plot_distribution(self, column):
        sns.histplot(self.df[column], kde=True)
        plt.show()

mpg = Main(df)

mpg.overview()
mpg.remove_duplicates()

# mpg.detect_outliers("horsepower")

# mpg.skewness()
# mpg.kurtosis()
# mpg.jarque_bera_test()

# mpg.plot_distribution("horsepower")
