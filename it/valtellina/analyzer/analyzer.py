import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import jarque_bera
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import base64
import io

from ucimlrepo import fetch_ucirepo

class Analyzer:

    def __init__(self, dataset_id=9):

        # download dataset
        dataset = fetch_ucirepo(id=dataset_id)

        # creo dataframe completo
        self.df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    def show_head(self):
        print(self.df.head())


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


    # describe
    def info_describe(self,column):
        print(self.df[column].describe())

    def imputing_mediana(self, column):
        self.df[column] = self.df[column].fillna(self.df[column].median())


    # IQR: interquartile range. indica qaunto sono sparsi i dati centrali, da Q1 a Q3.
    # IQR = Q3-Q1
    # serve per trovare outliers --> un valore x è OUTLIER se x<Q1−1.5*IQR oppure x>Q3+1.5*IQR
    def count_outliers(self):
            # seleziono solo le colonne numeriche
            columns = self.df.select_dtypes(include=['number']).columns

            outlier_counts = {}

            for col in columns:
                # Calcolo dei parametri IQR
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Calcolo dei limiti
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Conteggio dei valori che stanno al di fuori dei limiti
                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_counts[col] = outliers.sum()

            return outlier_counts



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

    # skewness (asimmetria): misura quanto una distribuzione
    # è "spostata" rispetto a una distribuzione normale
    # = 0: simmetrica, > 0: ho tanti valori verso lo 0, < 0: ho tanti valori alti
    # tra -0.5 e +0.5 non faccio nulla
    # tra (-)0.5 e (-)1 possibile trasformazione
    # > 1 o < -1 trasformazione obbligatoria (es. log)

    def skewness(self):
        numeric = self.df.select_dtypes(include='number')
        print(numeric.skew().sort_values(ascending=False))

    # breve spiegazione kurtosis:
    # La curtosi dice quanto la distribuzione
    # è concentrata intorno alla media o
    # quanto invece "scappa" verso valori estremi (outliers).
    # risponde alla domanda: “I dati hanno molti valori estremi oppure no?”
    # valori: circa 0: normale, tra 0 e 3: abbastanza normale, > 3 molti outliers, < 0: distribuzione piatta (no forma a campana)

    def kurtosis(self):
        numeric = self.df.select_dtypes(include='number')
        print(numeric.kurtosis().sort_values(ascending=False))

        # Test di Shapiro-Wilk per verificare la normalità.
        # H0: i dati seguono una distribuzione normale.
        # p-value < 0.05 -> rifiuto H0 (dati non normali)

    def check_normality_shapiro(self):
        num_cols = self.df.select_dtypes(include=['number']).columns
        results = {}

        for col in num_cols:
            stat, p_value = stats.shapiro(self.df[col])

            # p-value > 0.05 significa che non ho evidenze
            # sufficienti per rifiutare la normalità
            is_normal = p_value > 0.05

            results[col] = {
                    "Shapiro_stat": round(stat, 4),
                    "p-value": round(p_value, 4),
                    "is_normal": is_normal
            }

        return results

    # ----------------------------
    # PLOTS (per colonna)
    # ----------------------------
    def plot_distribution(self, column):
        sns.histplot(self.df[column], kde=True)
        plt.show()

    def plot_outliers(self, column):
        sns.boxplot(self.df[column])
        plt.show()

    def stampa_outliers(self):
        for col in ["horsepower", "acceleration", "mpg"]:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            print(f"\n{col}")
            print(self.df[(self.df[col] < lower) | (self.df[col] > upper)][col].sort_values())


    def drop_columns(self,columns):
        for col in columns:
            self.df = self.df.drop(col, axis=1)

    def correlation_matrix(self):
        # selezione colonne numeriche
        num_df = self.df.select_dtypes(include=['number'])
        # matrice di correlazione
        corr = num_df.corr()
        # maschera triangolo superiore
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5
        )

        plt.title("Correlation Heatmap (Lower Triangle)")
        #plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        plt.close()

        return img_base64


    def pca(self,  X_train):

        # scaling robusto perche abbiamo outliers (riduciamo impatto)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train)

        # PCA - indico la varianza che voglio avere nel dataset
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca, scaler, pca

    # applica scaler pca anche al test set
    def transform_pca(self, X_test, scaler, pca):
        X_scaled = scaler.transform(X_test)
        X_pca = pca.transform(X_scaled)
        return X_pca

    def stampa_pca(self, X_pca):
        pca_df = pd.DataFrame(
            X_pca,
            columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])]
    )
        print(pca_df.head())

    def get_columns(self):
        return self.df.columns

    def get_pca_loadings_sorted(self, pca, feature_names):
        # estraggo i loadings (pesi delle variabili originali) per spiegare le componenti attuali
        loadings = pd.DataFrame(
            pca.components_,
            columns=feature_names,
            index=[f"PC{i + 1}" for i in range(pca.n_components_)]
        )

        sorted_loadings = {}

        for pc in loadings.index:
            sorted_loadings[pc] = loadings.loc[pc].sort_values(key=abs, ascending=False)

        return loadings, sorted_loadings


