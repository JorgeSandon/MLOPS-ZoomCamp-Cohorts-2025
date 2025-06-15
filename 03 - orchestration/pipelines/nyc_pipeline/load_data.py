import pandas as pd

def load_data():
    print("📥 Cargando datos desde archivo local...")
    df = pd.read_parquet('data/yellow_tripdata_2023-03.parquet')
    print(f"Registros cargados: {len(df)}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)

    return df

