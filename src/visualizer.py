import matplotlib.pyplot as plt
import seaborn as sns


def create_visualization(df, chart_type, column):
    """
    Create charts from dataset columns.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    if chart_type == "Histogram":

        sns.histplot(
            data=df,
            x=column,
            kde=True,
            ax=ax
        )

    elif chart_type == "Bar Chart":

        df[column].value_counts().head(15).plot(
            kind="bar",
            ax=ax
        )

    elif chart_type == "Box Plot":

        sns.boxplot(
            data=df,
            y=column,
            ax=ax
        )

    ax.set_title(f"{chart_type}: {column}")

    plt.tight_layout()

    return fig