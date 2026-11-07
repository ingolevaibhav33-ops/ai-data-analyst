import streamlit as st

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.analyzer import analyze_data, get_numeric_correlations
from src.visualizer import create_visualization
from src.ai_analyst import ask_ai


st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst")

st.write(
    "Upload your dataset and let AI help you "
    "understand your data."
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)


if uploaded_file:

    try:

        df = load_data(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(df.head(20))

        # Data cleaning

        cleaned_df = clean_data(df)

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", len(cleaned_df))

        col2.metric("Columns", len(cleaned_df.columns))

        col3.metric(
            "Duplicates",
            int(cleaned_df.duplicated().sum())
        )

        # Statistical analysis

        st.subheader("Statistical Analysis")

        analysis = analyze_data(cleaned_df)

        st.json(analysis["missing_values"])

        st.write(cleaned_df.describe(include="all"))

        # Visualization

        st.subheader("Data Visualization")

        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            selected_column = st.selectbox(
                "Select column",
                numeric_columns
            )

            chart_type = st.selectbox(
                "Chart type",
                ["Histogram", "Box Plot"]
            )

            fig = create_visualization(
                cleaned_df,
                chart_type,
                selected_column
            )

            st.pyplot(fig)

        # Correlation

        correlation = get_numeric_correlations(cleaned_df)

        if correlation is not None:

            st.subheader("Correlation Analysis")

            st.dataframe(correlation)

        # AI Analyst

        st.subheader("🤖  AI Data Analyst")

        question = st.text_area(
            "Ask anything about your dataset",
            placeholder="Why are sales decreasing?"
        )

        if st.button("Ask NVIDIA AI"):

            if question:

                with st.spinner("NVIDIA AI is thinking..."):

                    answer = ask_ai(
                        question,
                        cleaned_df
                    )

                st.markdown(answer)

                # Download cleaned data

                csv_data = cleaned_df.to_csv(index=False)

                st.download_button(
                    "Download Cleaned Dataset",
                    csv_data,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )

    except Exception as error:

        st.error(f"Something went wrong: {error}")

else:

    st.info("Upload a dataset to begin.")