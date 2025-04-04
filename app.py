import streamlit as st
import tempfile
from encoder import encode_query
from utils import query_pinecone
from heatmap import draw_similarity_grid
from rerank import rerank_with_gpt4
from PIL import Image
import os
import zipfile
import requests

print("App launched")

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAMESPACE = os.getenv("NAMESPACE", "ns1")

# Data
AID_ZIP_URL = "https://drive.google.com/file/d/1VUJZn7Rv7oyZ8iotQPjouPw43xOr8NvO/view?usp=drive_link"
zip_path = "data/AID.zip"


def download_and_extract_aid():
    os.makedirs("data", exist_ok=True)
    print("Downloading AID dataset from Google Drive...")
    with requests.get(AID_ZIP_URL, stream=True) as r:
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Extracting AID.zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data")
    os.remove(zip_path)
    print("Dataset ready.")


download_and_extract_aid()

# Search
st.set_page_config("AID Multimodal Search", layout="wide")
st.title("Satellite Image Search (Multimodal)")
st.caption("Search by text, image, or both using Pinecone + CLIP")
st.write("Streamlit UI loaded")

query_text = st.text_input(
    "Enter a text prompt", placeholder="e.g., an aerial view of a harbor"
)
uploaded_file = st.file_uploader("Upload a query image (optional)", type=["jpg", "png"])
search_mode = st.radio("Search Mode", ["Text Only", "Image Only", "Text + Image"])

if st.button("Search"):
    with st.spinner("Processing query and retrieving results..."):
        image_path = None
        if uploaded_file:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        query_vector = encode_query(
            text=query_text if "Text" in search_mode else None,
            image_path=image_path if "Image" in search_mode else None,
        )

        results = query_pinecone(vector=query_vector, namespace=NAMESPACE, top_k=6)

        st.markdown("## Top Results")
        cols = st.columns(3)
        vectors = []
        paths = []

        for i, match in enumerate(results):
            vectors.append(match["values"] if "values" in match else query_vector)
            paths.append(match["metadata"]["image_path"])
            with cols[i % 3]:
                st.image(
                    match["metadata"]["image_path"],
                    caption=f"{match['metadata']['caption']}\n(score={match['score']:.2f})",
                    use_column_width=True,
                )

        st.markdown("## Similarity Heatmap")
        draw_similarity_grid(query_vector, vectors, paths)

        with st.expander("GPT-4o Reranking & Explanation"):
            if st.button("Run GPT-4o Reranker"):
                st.info("Calling GPT-4o...")
                result = rerank_with_gpt4(query_text, paths)
                st.markdown(result)
