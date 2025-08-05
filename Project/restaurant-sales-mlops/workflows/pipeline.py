from prefect import flow, task
from src.data_processing.preprocess import load_data, clean_data, save_processed
from src.training.train import train_model

@task
def preprocessing():
    df = load_data()
    df_clean = clean_data(df)
    save_processed(df_clean)
    return "Preprocesamiento completado"

@task
def training():
    train_model()
    return "Entrenamiento completado"

@flow
def main_pipeline():
    preprocessing()
    training()

if __name__ == "__main__":
    main_pipeline()
