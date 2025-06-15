import mlflow
import mlflow.sklearn

def register_model(lr, dv, rmse):
    mlflow.set_tracking_uri("mlruns")

    with mlflow.start_run():
        mlflow.set_tag("developer", "jsandon")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("vectorizer_features", len(dv.feature_names_))
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(lr, artifact_path="model")
        print("✅ Modelo registrado con MLflow")
