from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

def train_model(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    rmse = root_mean_squared_error(y_train, y_pred)

    print(f"Intercepto: {lr.intercept_:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return lr, rmse
