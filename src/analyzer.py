import pandas as pd


def analyze_data(df):
    """
    Generate a statistical summary of the dataset.
    """

    analysis = {}

    analysis["rows"] = len(df)

    analysis["columns"] = len(df.columns)

    analysis["column_names"] = list(df.columns)

    analysis["missing_values"] = (
        df.isnull().sum().to_dict()
    )

    analysis["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    analysis["statistics"] = (
        df.describe(include="all")
        .fillna("")
        .to_dict()
    )

    return analysis


def get_numeric_correlations(df):
    """
    Find relationships between numerical columns.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return None

    return numeric_df.corr()