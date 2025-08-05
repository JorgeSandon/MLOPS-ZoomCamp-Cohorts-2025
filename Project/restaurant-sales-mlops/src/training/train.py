import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import joblib
import os
import mlflow
import mlflow.sklearn
import numpy as np

DATA_PATH = "data/processed/sales_clean.csv"
MODEL_PATH = "data/outputs/best_model.pkl"

def load_data(path=DATA_PATH):
    """Carga los datos procesados"""
    if os.path.exists(path):
        print(f"📊 Cargando datos desde: {path}")
        return pd.read_csv(path)
    else:
        print(f"⚠️ Archivo no encontrado: {path}")
        # Buscar archivos alternativos
        processed_dir = "data/processed"
        if os.path.exists(processed_dir):
            csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
            if csv_files:
                alt_path = os.path.join(processed_dir, csv_files[0])
                print(f"📊 Usando archivo alternativo: {alt_path}")
                return pd.read_csv(alt_path)
        
        # Si no hay datos, crear datos de ejemplo
        print("🎲 Creando datos de ejemplo...")
        return create_sample_data()

def create_sample_data():
    """Crea datos de ejemplo para pruebas"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generar datos de ejemplo
    price = np.random.uniform(5, 50, n_samples)
    quantity = np.random.uniform(10, 200, n_samples)
    
    # Crear ventas totales con algo de lógica realista
    total_sales = price * quantity + np.random.normal(0, price * quantity * 0.1)
    total_sales = np.maximum(total_sales, 0)  # No ventas negativas
    
    df = pd.DataFrame({
        'Price': price,
        'Quantity': quantity,
        'Total_Sales': total_sales
    })
    
    return df

def train_and_log_models():
    """Entrena múltiples modelos y registra en MLflow"""
    print("🚀 Iniciando entrenamiento de modelos...")
    
    df = load_data()
    print(f"📊 Dataset cargado: {df.shape}")
    print(f"🔍 Columnas disponibles: {list(df.columns)}")

    # Verificar columnas necesarias
    required_cols = ["Price", "Quantity", "Total_Sales"]
    available_cols = df.columns.tolist()
    
    # Mapeo de columnas alternativas
    column_mapping = {}
    for req_col in required_cols:
        if req_col not in available_cols:
            # Buscar columnas similares
            similar_cols = [col for col in available_cols if req_col.lower() in col.lower()]
            if similar_cols:
                column_mapping[req_col] = similar_cols[0]
                print(f"📝 Mapeando '{req_col}' -> '{similar_cols[0]}'")
    
    # Aplicar mapeo si es necesario
    if column_mapping:
        df = df.rename(columns={v: k for k, v in column_mapping.items()})

    # Verificar si tenemos las columnas necesarias
    if not all(col in df.columns for col in required_cols):
        print("⚠️ Columnas requeridas no encontradas, usando todas las columnas numéricas...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            print("❌ No hay suficientes columnas numéricas")
            return None
        
        # Usar las últimas columnas como target y las demás como features
        X = df[numeric_cols[:-1]]
        y = df[numeric_cols[-1]]
        print(f"🎯 Features: {list(X.columns)}")
        print(f"🎯 Target: {numeric_cols[-1]}")
    else:
        # Features y target
        X = df[["Price", "Quantity"]]
        y = df["Total_Sales"]

    print(f"📈 Datos de entrenamiento: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"🔄 División completada:")
    print(f"   - Train: {X_train.shape[0]} muestras")
    print(f"   - Test: {X_test.shape[0]} muestras")

    # Modelos a probar
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    best_model = None
    best_score = -float("inf")
    best_model_name = ""

    try:
        # Definir experimento en MLflow
        mlflow.set_experiment("restaurant-sales")
        print("📊 MLflow configurado")
    except Exception as e:
        print(f"⚠️ MLflow no disponible: {e}")
        print("🔄 Continuando sin MLflow...")

    print("🤖 Entrenando modelos...")
    
    for name, model in models.items():
        print(f"   🔧 Entrenando {name}...")
        
        try:
            # Usar MLflow si está disponible
            if 'mlflow' in globals():
                with mlflow.start_run(run_name=name):
                    model, r2, rmse = train_single_model(model, name, X_train, X_test, y_train, y_test)
            else:
                model, r2, rmse = train_single_model(model, name, X_train, X_test, y_train, y_test)
            
            print(f"   ✅ {name} → R²: {r2:.3f}, RMSE: {rmse:.3f}")

            # Guardar mejor modelo
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name

        except Exception as e:
            print(f"   ❌ Error entrenando {name}: {e}")
            continue

    # Guardar mejor modelo como .pkl
    if best_model:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        print(f"💾 Mejor modelo guardado: {MODEL_PATH}")
        print(f"🏆 Mejor modelo: {best_model_name} con R²={best_score:.3f}")
        
        # Guardar información del modelo
        model_info = {
            'model_name': best_model_name,
            'r2_score': best_score,
            'features': list(X.columns),
            'target': y.name if hasattr(y, 'name') else 'target'
        }
        
        import json
        info_path = "data/outputs/model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"📋 Información del modelo guardada: {info_path}")
        
        return best_model
    else:
        print("❌ No se pudo entrenar ningún modelo")
        return None

def train_single_model(model, name, X_train, X_test, y_train, y_test):
    """Entrena un modelo individual"""
    # Entrenar
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Log en MLflow si está disponible
    try:
        mlflow.log_param("model_type", name)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)

        # Ejemplo de entrada para el modelo
        input_example = X_test.iloc[:1]

        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name=name,
            input_example=input_example,
        )

    except Exception as e:
        print(f"   ⚠️ No se pudo loggear en MLflow: {e}")

    return model, r2, rmse

def train_model():
    """Función principal para el pipeline de Prefect"""
    print("🎯 Ejecutando entrenamiento desde pipeline...")
    return train_and_log_models()

def load_best_model():
    """Carga el mejor modelo entrenado"""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

def predict(features):
    """Realiza predicciones con el mejor modelo"""
    model = load_best_model()
    return model.predict(features)

if __name__ == "__main__":
    print("🧪 Ejecutando entrenamiento directamente...")
    model = train_and_log_models()
    if model:
        print("✅ Entrenamiento completado exitosamente!")
    else:
        print("❌ Error en el entrenamiento")