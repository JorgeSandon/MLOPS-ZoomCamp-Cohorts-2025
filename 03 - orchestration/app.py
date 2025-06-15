import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow
import mlflow.sklearn


def load_data():
    print("\n🔹 Cargando datos de marzo 2023...")
    url = 'data/yellow_tripdata_2023-03.parquet'
    df = pd.read_parquet(url)
    print(f"➡️ Registros cargados: {len(df)}")  # Q3

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    print(f"✅ Registros tras filtro de duración (1-60 min): {len(df)}")  # Q4

    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)

    return df


def transform_data(df):
    print("\n🔹 Transformando datos (one-hot encoding)...")
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    print(f"➡️ Shape X_train: {X_train.shape}")  # Q4

    return X_train, y_train, dv


def train_model(X_train, y_train):
    print("\n🔹 Entrenando modelo (LinearRegression)...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    rmse = root_mean_squared_error(y_train, y_pred)

    print(f"✅ Intercepto del modelo: {lr.intercept_:.2f}")  # Q5
    print(f"✅ RMSE en entrenamiento: {rmse:.2f}")

    return lr, rmse


def register_model(lr, dv, rmse):
    print("\n🔹 Registrando modelo en MLflow...")
    mlflow.set_tracking_uri("mlruns")


    with mlflow.start_run():
        mlflow.set_tag("developer", "jsandon")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("vectorizer_features", len(dv.feature_names_))
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(lr, artifact_path="model")
        print("✅ Modelo registrado con éxito.")


def main():
    df = load_data()
    X_train, y_train, dv = transform_data(df)
    lr, rmse = train_model(X_train, y_train)
    register_model(lr, dv, rmse)


if __name__ == "__main__":
    main()
