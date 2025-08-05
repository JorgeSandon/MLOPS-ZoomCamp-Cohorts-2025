import pandas as pd
import os

RAW_PATH = "data/raw/Sales-Data-Analysis.csv"
PROCESSED_PATH = "data/processed/sales_clean.csv"

def load_data(path=RAW_PATH):
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame):
    # Quitar espacios extras en nombres de columnas y valores
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Convertir fecha
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    # Calcular ventas totales
    df["Total_Sales"] = df["Price"] * df["Quantity"]

    return df

def save_processed(df, path=PROCESSED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    save_processed(df_clean)
    print(f"Datos procesados guardados en {PROCESSED_PATH}")
