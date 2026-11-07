import pandas as pd


def load_data(file):
    """
    Load CSV or Excel data into a Pandas DataFrame.
    """

    file_name = file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(file)

    elif file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    else:
        raise ValueError("Unsupported file format")