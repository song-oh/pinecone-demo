import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)


def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.squeeze().cpu().numpy()


def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    return embedding.squeeze().cpu().numpy()


def encode_query(text=None, image_path=None):
    image_emb = encode_image(image_path) if image_path else None
    text_emb = encode_text(text) if text else None
    if image_emb is not None and text_emb is not None:
        return ((image_emb + text_emb) / 2).tolist()
    elif image_emb is not None:
        return image_emb.tolist()
    elif text_emb is not None:
        return text_emb.tolist()
    else:
        raise ValueError("At least one of text or image must be provided.")
