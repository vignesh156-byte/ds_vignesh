import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


# Define model
def get_vgg16():
    model = models.vgg16(weights=None)

    # same classifier as training
    model.classifier[6] = nn.Linear(4096, 2)

    return model


# Load model
@st.cache_resource
def load_model():
    model = get_vgg16()
    model.load_state_dict(
        torch.load("vgg16_model.pth", map_location="cpu")
    )
    model.eval()
    return model


model = load_model()


# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# UI
st.title("🦅 Bird / 🚁 Drone Classifier")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader(
    "Choose image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    classes = ["Bird", "Drone"]

    st.success(
        f"Prediction: {classes[pred.item()]}"
    )

    st.info(
        f"Confidence: {conf.item()*100:.2f}%"
    )