from flask import Flask, jsonify, request
from it.valtellina.analyzer.analyzer import Analyzer
from it.valtellina.regression_model.regression_model import RegressionModel
from it.valtellina.splitter_train_test.splitter_train_test import SplitterTrainTest

app = Flask(__name__)

@app.route('/')
def intro():
    return 'Benvenuto!'

@app.route('/api/outliers')
def outliers():
    mpg = Analyzer()
    totale_outliers = mpg.count_outliers()
    # cast a int da int64
    totale_outliers = {k: int(v) for k, v in totale_outliers.items()}
    return jsonify({"totale outliers:": totale_outliers})


@app.route('/api/correlation-matrix')
def correlation_matrix():
    mpg = Analyzer()
    img = mpg.correlation_matrix()

    #return jsonify({"image": img})
    return f"""<img src="data:image/png;base64,{img}" />"""



@app.route("/api/select-model", methods=["POST"])
def select_model():
    data = request.json
    model = data.get("model") # --> model : linear

    if model not in ["linear", "ridge", "lasso"]:
        return jsonify({"message": "modello non valido"})

    # creo oggetto con il dataset in ingresso
    mpg = Analyzer()
    # imputing missing values
    mpg.imputing_mediana("horsepower")
    # elimino colonna useless
    mpg.drop_columns(['origin'])

    # splitting train e test set
    splitter = SplitterTrainTest(mpg.df, "mpg")
    X_train, X_test, y_train, y_test = splitter.split()

    # applicazione PCA per gestione feature
    X_train_pca, scaler, pca = mpg.pca(X_train)
    X_test_pca = mpg.transform_pca(X_test, scaler, pca)

    # addestro e applico il modello
    reg_linear = RegressionModel()
    reg_linear.select_model(model)
    reg_linear.fit(X_train_pca, y_train)
    y_pred = reg_linear.predict(X_test_pca)
    metrics = reg_linear.metrics(y_test, y_pred, X_train.shape[1])
    return jsonify({"metriche": metrics})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)