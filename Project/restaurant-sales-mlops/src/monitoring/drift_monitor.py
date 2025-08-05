import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfColumnsWithDrift

def detect_drift(new_data: pd.DataFrame, reference_data: pd.DataFrame):
    # Crear un reporte de Evidently para drift
    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(reference_data=reference_data, current_data=new_data)

    # Resultado en JSON
    results = report.as_dict()

    # Extraer métricas clave
    drift_detected = results["metrics"][0]["result"]["dataset_drift"]
    n_drifted = results["metrics"][0]["result"]["n_drifted_features"]

    return {
        "dataset_drift": drift_detected,
        "n_drifted_features": n_drifted,
        "drift_by_feature": results["metrics"][0]["result"]["drift_by_columns"]
    }

if __name__ == "__main__":
    ref = pd.read_csv("data/processed/sales_clean.csv").sample(500)
    new = pd.read_csv("data/raw/Sales-Data-Analysis.csv").sample(500)

    results = detect_drift(new, ref)
    print("🔎 Resultados de Evidently:")
    print(results)
