import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
groq_api_key = os.getenv("GROQ_API_KEY")  # Set this in your environment or hardcode for testing

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

def extract_table_from_pdf(pdf_file):
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
                except Exception as e:
                    st.warning(f"Skipping a table due to error: {e}")
    return tables


def suggest_chart_via_llm(df):
    prompt = f"""
You are a data visualization expert. Given the table below, suggest the best chart to visualize it:

{df.head().to_markdown(index=False)}

Use these rules:
- Use **line** charts for time-based data (like quarterly trends or yearly growth).
- Use **bar** charts for comparing categories.
- Use **pie** only when one column shows **percentages or proportions**.
- If unsure, prefer **bar** for simplicity.
- Choose chart types that **vary** if multiple tables are shown — don’t repeat the same type for all.
- Reply only in JSON format:
{{
  "chart_type": "bar",  # or "line", "pie"
  "x": "ColumnName",
  "y": ["Column1", "Column2"],
  "title": "Descriptive chart title"
}}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return json.loads(response.content)

def plot_dynamic_chart(df, config):
    fig, ax = plt.subplots()
    x = config["x"]
    y = config["y"]
    chart_type = config["chart_type"]

    df_clean = df.copy()
    df_clean[y] = df_clean[y].apply(pd.to_numeric, errors='coerce')

    if chart_type == "line":
        for idx, row in df_clean.iterrows():
            ax.plot(y, row[y], marker='o', label=row[x])
    elif chart_type == "bar":
        df_grouped = df_clean.groupby(x)[y].sum()
        df_grouped.plot(kind="bar", ax=ax)
    elif chart_type == "pie":
        df_grouped = df_clean.groupby(x)[y[0]].sum()
        df_grouped.plot(kind="pie", ax=ax, autopct='%1.1f%%')
    else:
        raise ValueError("Unsupported chart type")

    ax.set_title(config["title"])
    ax.set_xlabel(config["x"])
    ax.set_ylabel(", ".join(config["y"]))
    if chart_type != "pie":
        ax.legend()
    return fig

def run_visuals():
    st.title(" Report Visualizer")
    st.write("Upload a business report (PDF) with tabular data.")

    uploaded_file = st.file_uploader("Upload Report PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Extracting tables from report..."):
            try:
                tables = extract_table_from_pdf(uploaded_file)
            except Exception as e:
                st.error(f"Failed to extract tables: {e}")
                return

        if not tables:
            st.warning(" No tables detected in this PDF.")
            return

        for i, table in enumerate(tables):
            st.subheader(f" Table {i + 1}")
            st.dataframe(table)

            try:
                suggestion = suggest_chart_via_llm(table)
                st.markdown(f"** Chart Suggestion:** `{json.dumps(suggestion, indent=2)}`")

                fig = plot_dynamic_chart(table, suggestion)
                st.pyplot(fig)
            except Exception as e:
                try:
                    fig, ax = plt.subplots()
                    df_clean = table.copy()
                    df_clean.set_index(df_clean.columns[0], inplace=True)
                    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
                    df_clean.T.plot(ax=ax)
                    ax.set_title(f"Chart for Table {i + 1}")
                    st.pyplot(fig)
                except Exception as fallback_err:
                    st.error(f"chart failed: {fallback_err}")
