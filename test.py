import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from msve import SugarcaneMSVE

# ============================================
# Create Output Folder
# ============================================

os.makedirs("results", exist_ok=True)

# ============================================
# Load Model
# ============================================

model = SugarcaneMSVE()

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model (ONLY for testing)
torch.save(model.state_dict(), "models/msve_model.pth")

model.eval()

# ============================================
# Image Transformation
# ============================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ============================================
# Load Sugarcane Image
# ============================================

image_path = "D:/Sugarcane_MSVE1/images/rust.jpeg"

original_image = Image.open(image_path).convert("RGB")

image = transform(original_image)

image = image.unsqueeze(0)

# ============================================
# Display Original Image
# ============================================

plt.figure(figsize=(5,5))
plt.imshow(original_image)
plt.title("Original Sugarcane Leaf")
plt.axis("off")
plt.savefig("results/Original_Image.png", dpi=300)
plt.show()

# ============================================
# Forward Pass
# ============================================

with torch.no_grad():

    outputs = model(image)

scale1 = outputs["scale1"]
scale2 = outputs["scale2"]
scale3 = outputs["scale3"]
attention = outputs["attention"]
fused = outputs["fused"]

print("\nFinal Feature Shape")
print(fused.shape)

# ============================================
# Function to Display Feature Maps
# ============================================

def show_feature_maps(feature_tensor, title, filename):

    feature_maps = feature_tensor.squeeze(0)

    fig, axes = plt.subplots(4,4, figsize=(8,8))

    for i, ax in enumerate(axes.flat):

        ax.imshow(
            feature_maps[i].detach().cpu().numpy(),
            cmap="gray"
        )

        ax.set_title(f"{title}-{i+1}")

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"results/{filename}",
        dpi=300
    )

    plt.show()


# ============================================
# Scale-1 Feature Maps
# ============================================

show_feature_maps(
    scale1,
    "S1",
    "Scale1_Features.png"
)

# ============================================
# Scale-2 Feature Maps
# ============================================

show_feature_maps(
    scale2,
    "S2",
    "Scale2_Features.png"
)

# ============================================
# Scale-3 Feature Maps
# ============================================

show_feature_maps(
    scale3,
    "S3",
    "Scale3_Features.png"
)

# ============================================
# Attention Maps
# ============================================

attention_maps = attention.squeeze(0)

fig, axes = plt.subplots(1,3, figsize=(12,4))

titles = [
    "Scale-1 Weight",
    "Scale-2 Weight",
    "Scale-3 Weight"
]

for i in range(3):

    axes[i].imshow(
        attention_maps[i].detach().cpu().numpy(),
        cmap="jet"
    )

    axes[i].set_title(titles[i])

    axes[i].axis("off")

plt.tight_layout()

plt.savefig(
    "results/Attention_Maps.png",
    dpi=300
)

plt.show()

# ============================================
# Final Fused Feature Maps
# ============================================

show_feature_maps(
    fused,
    "F",
    "Fused_Features.png"
)

print("\n================================")
print("All Images Saved Successfully")
print("================================")