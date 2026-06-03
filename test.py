from it.valtellina.analyzer.analyzer import Analyzer
from it.valtellina.regression_model.regression_model import RegressionModel
from it.valtellina.splitter_train_test.splitter_train_test import SplitterTrainTest

# creo oggetto che contiene il dataset e le sue funzioni di analisi
mpg = Analyzer()

# Analisi missing values
mpg.overview() # ---> 6 nan su horsepower

# controllo duplcati
# mpg.remove_duplicates() ---> nessun duplicato presente


# controllo distribuzione normale
mpg.skewness()
mpg.kurtosis()

normality = mpg.check_normality_shapiro()
print("normality: " + str(normality)) # ---> nessuna feature normale


# se percentuale missing values < 5%, imputiamo mediana perche distribuzione asimmetrica (skewness > 1)
# 6 missing su 400 righe --> sotto 5%
mpg.info_describe("horsepower")
mpg.imputing_mediana("horsepower")

#controllo outliers
# conto il numero di outliers
totale_outliers = mpg.count_outliers()
print("totale outliers: " + str(totale_outliers))

# outliers in: horsepower, acceleration, mpg
mpg.detect_outliers("horsepower")
mpg.detect_outliers("acceleration")
mpg.detect_outliers("mpg")

# stampa i boxplot
#mpg.plot_outliers("horsepower")
#mpg.plot_outliers("acceleration")
#mpg.plot_outliers("mpg")

# stampa gli istogrammi
'''mpg.plot_distribution("horsepower")
mpg.plot_distribution("acceleration")
mpg.plot_distribution("mpg")
mpg.plot_distribution("origin")
mpg.plot_distribution("displacement")
mpg.plot_distribution("weight")
mpg.plot_distribution("cylinders")
mpg.plot_distribution("model_year")'''


# stampiamo i valori degli outliers
mpg.stampa_outliers()
# valori presenti pochi e realistici ---> non vengono eliminati, verranno standardizzati


# le colonna origin non ci da informazioni, per cui la possiamo rimuovere
mpg.drop_columns(['origin'])

# analisi di correlazione
#mpg.correlation_matrix()

#corr. positiva tra cylinders-displacement, horsepower-displacement, weight-displacement
                    # horsepower-cylinders, weight-cylinders
                    # horsepower-weight
#corr. negativa tra mpg-displacement, mpg-cylinders, mpg-horsepower, mpg- weight


# Splitting train-test sets
splitter = SplitterTrainTest(mpg.df, "mpg")
X_train, X_test, y_train, y_test = splitter.split()

# applichiamo pca al train per gestire le correlazioni troppo elevate. alcune features possono essere
# spiegate da altre

X_train_pca, scaler, pca = mpg.pca(X_train)
X_test_pca = mpg.transform_pca(X_test, scaler, pca)

#stampa dataset pca
mpg.stampa_pca(X_train_pca)

# stampo i loadings per spiegare le componenti
columns = X_train.columns
loadings = mpg.get_pca_loadings_sorted(pca, columns)
print(loadings)

# otteniamo tre colonne:
# PC1: rappresenta potenza + dimensione auto
# PC2 : rappresenta accelerazione
# PC3: anno di produzione

# applico i modelli
reg_linear = RegressionModel()

reg_linear.select_model("linear")   # poi da fare con POST in Flask

reg_linear.fit(X_train_pca, y_train)

y_pred = reg_linear.predict(X_test_pca)

metrics = reg_linear.metrics(y_test, y_pred, X_train_pca.shape[1])
print("Linear:")
print(metrics)

reg_lasso = RegressionModel()

reg_lasso.select_model("lasso")   # poi da fare con POST in Flask

reg_lasso.fit(X_train_pca, y_train)

y_pred = reg_lasso.predict(X_test_pca)

metrics = reg_lasso.metrics(y_test, y_pred, X_train_pca.shape[1])
print("Lasso:")
print(metrics)

reg_ridge = RegressionModel()

reg_ridge.select_model("ridge")   # poi da fare con POST in Flask

reg_ridge.fit(X_train_pca, y_train)

y_pred = reg_ridge.predict(X_test_pca)

metrics = reg_ridge.metrics(y_test, y_pred, X_train_pca.shape[1])
print("Ridge:")
print(metrics)