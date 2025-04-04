import base64
import openai
import os

api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = api_key


def rerank_with_gpt4(query, image_paths):
    imgs = []
    for path in image_paths:
        with open(path, "rb") as f:
            img = base64.b64encode(f.read()).decode("utf-8")
            imgs.append(img)

    prompt = (
        f"The user asks: '{query}'. You are shown 6 satellite images.\n"
        f"Please rank them from most relevant to least, and explain your top choice."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                    for img in imgs
                ],
            ],
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o", messages=messages, max_tokens=300
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"
