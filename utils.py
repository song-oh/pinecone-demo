from pinecone import Pinecone
import os

INDEX_NAME = "clip-vit-base-patch32"
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)


def query_pinecone(vector, namespace, top_k=6):
    return index.query(
        namespace=namespace, vector=vector, top_k=top_k, include_metadata=True
    )["matches"]
