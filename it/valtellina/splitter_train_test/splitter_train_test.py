from sklearn.model_selection import train_test_split

class SplitterTrainTest():
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def split(self, test_size=0.2, random_state=42):

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test