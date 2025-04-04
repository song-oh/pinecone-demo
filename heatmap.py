import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import streamlit as st


def draw_similarity_grid(query_vector, result_vectors, result_paths):
    sim_scores = cosine_similarity([query_vector], result_vectors)[0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, score, img_path in zip(axes.flat, sim_scores, result_paths):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"Similarity: {score:.2f}")
        ax.axis("off")
    st.pyplot(fig)
