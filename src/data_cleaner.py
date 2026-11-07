import pandas as pd


def clean_data(df):
    """
    Clean the dataset by removing duplicates
    and handling missing values.
    """

    cleaned_df = df.copy()

    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()

    # Fill missing numeric values with median
    numeric_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        cleaned_df[column] = cleaned_df[column].fillna(
            cleaned_df[column].median()
        )

    # Fill missing text values
    text_columns = cleaned_df.select_dtypes(
        include="object"
    ).columns

    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].fillna(
            "Unknown"
        )

    return cleaned_df