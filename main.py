# main.py

import streamlit as st
import app  # Document Q&A
import imgapp  # Image Insight
import reportvisuals  # Report Visuals

st.set_page_config(
    page_title="DocuMind: Understand Your Documents, Effortlessly",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("  Navigation")
page = st.sidebar.radio("Select a module:", ["Document Q&A", "Image Insight", "Report Visuals"])

# Page routing
if page == "Document Q&A":
    app.run_app()
