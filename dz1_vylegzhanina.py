import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLR # Для проверки 

class LinearRegression():

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            self.update_weights(X, Y)

    def update_weights(self, X, Y):
        n_samples = X.shape[0]

        Y_pred = np.dot(X, self.weights) + self.bias

        dw = (1 / n_samples) * np.dot(X.T, (Y_pred - Y))
        db = (1 / n_samples) * np.sum(Y_pred - Y)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


df = pd.read_csv("https://drive.google.com/uc?export=download&id=1jHFa9TyMxtE_r9tldzooC6NJ_e8X4tbq")
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression(learning_rate=0.01, iterations=5000)
model.fit(X_scaled, Y)
print("=== Моя модель ===")
print("Weights:", np.round(model.weights, 3))
print("Bias:", np.round(model.bias, 3))

# sklearn для проверки
sk_model = SklearnLR()
sk_model.fit(X_scaled, Y)
print("\n=== sklearn ===")
print("Weights:", np.round(sk_model.coef_, 3))
print("Bias:", np.round(sk_model.intercept_, 3))
