#!/usr/bin/env python
"""
Final verification script - Run this after pip install completes
"""

import sys
import os

print("\n" + "="*70)
print("✅ FINAL VERIFICATION - DocuMind Setup")
print("="*70 + "\n")

errors = []
warnings = []

# Test 1: Core imports
print("🔍 Testing core imports...")
try:
    from core import (
        get_llm,
        get_embedding_model,
        load_documents_from_files,
        build_conversational_chain,
        convert_csv_to_sqlite,
        generate_sql_query,
        SimpleChatHistory,
    )
    print("✅ core.py imports successful\n")
except Exception as e:
    print(f"❌ core.py import failed: {str(e)}\n")
    errors.append(f"core.py: {str(e)}")

# Test 2: Streamlit
print("🔍 Testing Streamlit...")
try:
    import streamlit as st
    print(f"✅ Streamlit {st.__version__} loaded\n")
except Exception as e:
    print(f"❌ Streamlit failed: {str(e)}\n")
    errors.append(f"Streamlit: {str(e)}")

# Test 3: LangChain
print("🔍 Testing LangChain...")
try:
    import langchain
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    print(f"✅ LangChain {langchain.__version__} loaded\n")
except Exception as e:
    print(f"❌ LangChain failed: {str(e)}\n")
    errors.append(f"LangChain: {str(e)}")

# Test 4: Document processing
print("🔍 Testing document processors...")
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    print("✅ Document loaders loaded\n")
except Exception as e:
    print(f"❌ Document loaders failed: {str(e)}\n")
    errors.append(f"Document loaders: {str(e)}")

# Test 5: Embeddings
print("🔍 Testing embeddings (this may take a moment)...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True}
    )
    print("✅ Embeddings initialized\n")
except Exception as e:
    print(f"❌ Embeddings failed: {str(e)}\n")
    errors.append(f"Embeddings: {str(e)}")

# Test 6: .env file
print("🔍 Checking .env configuration...")
if os.path.exists(".env"):
    print("✅ .env file found")
    with open(".env", "r") as f:
        content = f.read()
        if "GROQ_API_KEY" in content and "gsk_" in content:
            print("✅ GROQ_API_KEY configured\n")
        else:
            print("⚠️  GROQ_API_KEY might be empty or invalid\n")
            warnings.append("GROQ_API_KEY not properly configured")
else:
    print("⚠️  .env file not found")
    print("⚠️  Create .env with: GROQ_API_KEY=gsk_your_key_here\n")
    warnings.append(".env file missing")

# Test 7: Groq connection (optional)
print("🔍 Testing Groq API connection...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key == "gsk_your_key_here":
        print("⚠️  GROQ_API_KEY not set in .env\n")
        warnings.append("Groq API key not configured")
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.3
        )
        response = llm.invoke("Say 'OK'")
        if "OK" in str(response):
            print("✅ Groq API connection working\n")
        else:
            print(f"⚠️  Groq response unexpected: {str(response)[:50]}\n")
except Exception as e:
    print(f"⚠️  Groq test skipped (offline or no API key): {str(e)}\n")
    warnings.append("Groq connection not tested")

# Test 8: Data processing
print("🔍 Testing data processing...")
try:
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(f"✅ Pandas/NumPy working\n")
except Exception as e:
    print(f"❌ Data processing failed: {str(e)}\n")
    errors.append(f"Data processing: {str(e)}")

# Test 9: Required directories
print("🔍 Checking required directories...")
dirs_to_check = ["temp_files", "chroma_store"]
for dir_name in dirs_to_check:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"✅ Created {dir_name}/")
    else:
        print(f"✅ {dir_name}/ exists")
print()

# Summary
print("="*70)
if errors:
    print(f"❌ SETUP INCOMPLETE - {len(errors)} error(s) found:\n")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print("\n🔧 Fix the errors above before running the app.")
    sys.exit(1)
elif warnings:
    print(f"⚠️  SETUP MOSTLY COMPLETE - {len(warnings)} warning(s):\n")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print("\n💡 The app should work, but check warnings if issues occur.")
else:
    print("✅ SETUP COMPLETE - All tests passed!")
    print("\n🚀 Ready to run:")
    print("   streamlit run main.py")
    print("\n📱 Then open: http://localhost:8501")

print("="*70 + "\n")