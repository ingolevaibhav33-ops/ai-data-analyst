
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables from the exact file
load_dotenv(
    dotenv_path=ENV_FILE,
    override=True
)


# --------------------------------------------------
# 2. NVIDIA API CONNECTION
# --------------------------------------------------

def get_nvidia_client():

    if not ENV_FILE.is_file():
        raise FileNotFoundError(
            f".env file not found at: {ENV_FILE}"
        )

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key or not api_key.strip():
        raise ValueError(
            "NVIDIA_API_KEY is missing or empty. "
            "Check your root .env file."
        )

    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key.strip(),
        timeout=180.0
    )


# --------------------------------------------------
# 3. DATASET CONTEXT
# --------------------------------------------------

def get_dataset_context(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    context = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),

        "data_types": df.dtypes.astype(str).to_dict(),

        "missing_values": (
            df.isnull().sum().to_dict()
        ),

        "sample_data": (
            df.head(10).to_dict(orient="records")
        ),

        "statistics": (
            numeric_df.describe().to_dict()
        )
    }

    return json.dumps(
        context,
        default=str
    )


# --------------------------------------------------
# 4. NVIDIA AI DATA ANALYST
# --------------------------------------------------

def ask_ai(question, df):

    if df is None or df.empty:
        return "Please upload a valid dataset."

    if not question or not question.strip():
        return "Please enter a question."

    try:

        client = get_nvidia_client()

        model = os.getenv(
            "NVIDIA_MODEL",
            "nvidia/nemotron-3-ultra-550b-a55b"
        )

        dataset_context = get_dataset_context(df)

        messages = [

            {
                "role": "system",
                "content": """
                You are an expert AI Data Analyst.

                Your expertise includes:
                Python, Pandas, Statistics,
                Machine Learning and Business Intelligence.

                Analyze the supplied dataset.

                Requirements:

                1. Answer the user's question directly.
                2. Explain findings in simple language.
                3. Never invent numerical results.
                4. Separate facts from assumptions.
                5. Identify important patterns.
                6. Suggest actionable recommendations.
                7. Mention limitations in the data.
                8. Use structured Markdown formatting.
                """
            },

            {
                "role": "user",
                "content": f"""
                DATASET INFORMATION:

                {dataset_context}

                USER QUESTION:

                {question}

                Give a detailed and useful analysis.
                """
            }
        ]

        response = client.chat.completions.create(

            model=model,

            messages=messages,

            temperature=0.3,

            max_tokens=4096
        )

        answer = response.choices[0].message.content

        if not answer:
            return "NVIDIA returned an empty response."

        return answer

    except Exception as error:

        return (
            f"NVIDIA API Error: {type(error).__name__}\n\n"
            f"Details: {str(error)}"
        )
