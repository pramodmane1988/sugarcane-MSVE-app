import os
import torch
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from msve import SugarcaneMSVE

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Sugarcane Disease Analysis",
    layout="wide"
)

st.title("🌿 Sugarcane Multi-Scale Vision Encoder")
st.write("Upload a sugarcane leaf image.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model = SugarcaneMSVE()

    model.load_state_dict(
        torch.load(
            "models/msve_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# Upload Image
# -----------------------------
uploaded = st.file_uploader(
    "Choose Image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.image(image, width=300)

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(tensor)

    scale1 = outputs["scale1"]
    scale2 = outputs["scale2"]
    scale3 = outputs["scale3"]
    attention = outputs["attention"]
    fused = outputs["fused"]

    st.success("Feature Extraction Completed")

    st.write("### Feature Shapes")

    st.write(scale1.shape)
    st.write(scale2.shape)
    st.write(scale3.shape)
    st.write(attention.shape)
    st.write(fused.shape)

    # -----------------------------
    # Function
    # -----------------------------
    def plot_feature_maps(feature_tensor,title):

        feature_maps = feature_tensor.squeeze(0)

        n = min(16,feature_maps.shape[0])

        fig,axes = plt.subplots(4,4,figsize=(8,8))

        axes=axes.flatten()

        for i in range(16):

            axes[i].axis("off")

            if i<n:

                img = feature_maps[i].cpu().numpy()

                img=(img-img.min())/(img.max()-img.min()+1e-8)

                axes[i].imshow(img,cmap="viridis")

                axes[i].set_title(f"{title}-{i+1}")

        st.pyplot(fig)

    st.header("Scale-1")

    plot_feature_maps(scale1,"S1")

    st.header("Scale-2")

    plot_feature_maps(scale2,"S2")

    st.header("Scale-3")

    plot_feature_maps(scale3,"S3")

    st.header("Fused Feature")

    plot_feature_maps(fused,"F")

    st.header("Attention Maps")

    att = attention.squeeze(0)

    fig,axes=plt.subplots(1,3,figsize=(12,4))

    titles=[
        "Scale1",
        "Scale2",
        "Scale3"
    ]

    for i in range(3):

        img=att[i].cpu().numpy()

        img=(img-img.min())/(img.max()-img.min()+1e-8)

        axes[i].imshow(img,cmap="inferno")

        axes[i].set_title(titles[i])

        axes[i].axis("off")

    st.pyplot(fig)