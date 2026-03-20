class StandardScaler():
    def __str__(self):
        return "StandardScaler"

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # the standard deviation can be 0, which provokes
        # devision-by-zero errors; let's avoid that:
        self.std[self.std == 0] = 0.00001

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean