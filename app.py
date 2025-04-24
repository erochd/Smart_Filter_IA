# app.py
import streamlit as st
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import cv2

# Titre de l'app
st.set_page_config(page_title="Filtration Anomaly Detection", layout="centered")
st.title("🔍 Détection de mauvaise filtration")
st.markdown("Chargez une image de filtration et laissez l'IA vous dire si elle est correcte ou anormale.")

# Chargement du modèle
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    # Charger le modèle depuis Hugging Face
    model_url = "https://huggingface.co/erochd/Smart_Filter_IA/resolve/main/resnet50_filtration_model.pt"
    response = requests.get(model_url)
    model.load_state_dict(torch.load(BytesIO(response.content), map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Upload de l'image
uploaded_file = st.file_uploader("📤 Choisissez une image de filtration", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Image chargée", use_column_width=True)

    original_image = pil_image.resize((224, 224))
    input_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])(pil_image).unsqueeze(0).to(device)

    # Prédiction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, 1).item()
        probs = probabilities.cpu().numpy()[0]

    # Résultats
    classes = ["Bonne filtration", "Mauvaise filtration"]
    st.subheader(f"✅ Résultat : **{classes[prediction]}**")
    st.write("📊 **Probabilités :**")
    st.write(f"- Bonne filtration : {probs[0]*100:.2f}%")
    st.write(f"- Mauvaise filtration : {probs[1]*100:.2f}%")

    # Grad-CAM si mauvaise filtration détectée
    if prediction == 1:
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        img_np = np.array(original_image).astype(np.float32) / 255.0
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        st.markdown("### 🔥 Zone d’influence détectée (Grad-CAM)")
        st.image(visualization, use_column_width=True)
