import pandas as pd
from src.data_processing.preprocess import clean_data

def test_clean_data():
    raw = pd.DataFrame({
        "Date": ["07-11-2022"],
        "Price": [10],
        "Quantity": [2]
    })
    clean = clean_data(raw)
    assert "Total_Sales" in clean.columns
    assert clean["Total_Sales"].iloc[0] == 20
