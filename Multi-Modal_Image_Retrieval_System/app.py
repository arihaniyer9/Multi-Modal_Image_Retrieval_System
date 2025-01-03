import gradio as gr
import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = CLIPModel.from_pretrained("./CLIP-VIT").to(device)
processor = CLIPProcessor.from_pretrained("./CLIP-VIT")
print("Model loaded!")

client = chromadb.PersistentClient("img_db/")


def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        return image_features.to(torch.float16).cpu().squeeze(0).numpy().tolist()


def search(query=None, image=None):
    collection = client.get_collection("images")

    if query:  # Text-based search
        with torch.no_grad():
            text_emb = model.get_text_features(
                **processor(text=query, return_tensors="pt").to(device)
            )
        query_embedding = text_emb.cpu().squeeze(0).tolist()

    elif image:  # Image-based search
        img = Image.open(image).convert("RGB")
        query_embedding = extract_features_clip(img)

    # Perform the query using either text or image embeddings
    results = collection.query(
        query_embeddings=query_embedding, n_results=4
    )

    gallery_images = [Image.open(doc) for doc in results["documents"][0]]
    return gallery_images


if __name__ == "__main__":
    demo = gr.Interface(
        fn=search,
        inputs=[
            gr.Textbox(placeholder="Enter a text query (optional)"),
            gr.Image(type="filepath", label="Upload an image (optional)"),
        ],
        outputs=gr.Gallery(label="Results", selected_index=0, preview=True),
        title="Multi-Modal Image Retrieval System",
        description="An image search engine powered by CLIP! Search by text or by image or both.",
        theme=gr.themes.Default(primary_hue="purple"),
    )

    demo.launch(share=True)
