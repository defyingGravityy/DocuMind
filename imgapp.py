import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import ollama
from langchain_groq import ChatGroq
import tempfile
import easyocr
import numpy as np

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Load BLIP captioning model ===
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_fast=False
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model
def extract_text(image):
	reader=easyocr.Reader(['en'],gpu=True)
	image_np=np.array(image)
	results=reader.readtext(image_np)
	extracted_text= "\n".join([item[1] for item in results])
	return extracted_text
# === Caption Generator ===
def generate_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# === Insight Generator with LLaVA ===
def generate_insight_with_llava(image):
    try:
        # Extract text using OCR
        ocr_text = extract_text(image)

        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image_path = tmp.name
            image.save(image_path)

        # Refined prompt with OCR context
        prompt = (
            "You are a visual analyst AI. Analyze the uploaded image, which is a chart, graph, or diagram. "
            "Use the following extracted text from the image to guide your analysis:\n\n"
            f"{ocr_text}\n\n"
            "Describe only what is clearly visible. Avoid assumptions. Focus on actual trends, labels, and data points. "
            "Provide a concise summary of the visual content."
        )

        # Call LLaVA via Ollama
        response = ollama.chat(
            model='llava:latest',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )

        return response['message']['content']

    except Exception as e:
        return f"[LLaVA Error] {str(e)}"


# === Fallback Insight Generator with Groq ===
def generate_insight_with_groq(caption):
    try:
        prompt = (
            "You are an intelligent assistant. Based on the following image caption, "
            "provide useful insights or summaries that help understand its content, "
            "such as trends, key data, or interpretation.\n\n"
            f"Image Caption: \"{caption}\"\n\n"
            "Respond with a concise explanation or inference."
        )
        client = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model="llama3-70b-8192"
        )
        return client.invoke(prompt).content
    except Exception:
        return "[Groq Fallback] Unable to generate insight."

# === Main Streamlit App ===
def run_imgapp():
    st.title("Image Insight Generator")
    st.markdown(
        "Upload a graph, table, or diagram image. "
        "Get AI-generated caption and insight."
    )

    uploaded_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        try:
            with st.spinner(" Generating caption..."):
                processor, model = load_blip_model()
                caption = generate_caption(image, processor, model)

            st.success("✅ Caption Generated:")
            st.markdown(f"> **Caption:** {caption}")

            with st.spinner(" Generating insight via LLaVA..."):
                insight = generate_insight_with_llava(image)

            if insight:
                st.success(" Insights:")
                st.markdown(f"> **{insight}**")
            else:
                with st.spinner(" LLaVA failed, using Groq fallback..."):
                    fallback_insight = generate_insight_with_groq(caption)
                st.warning("Fallback Insight:")
                st.markdown(f"> **{fallback_insight}**")

        except Exception as e:
            st.error(f" An error occurred: {str(e)}")


