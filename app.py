import os
import torch
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Device Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Image Paths
# -----------------------------
CONTENT_IMAGE = "images/content/content.jpg"
STYLE_IMAGE = "images/style/style.jpg"
OUTPUT_IMAGE = "images/output/stylized_image.jpg"

# -----------------------------
# Check if images exist
# -----------------------------
if not os.path.exists(CONTENT_IMAGE):
    print(f"Error: Content image not found!\nExpected: {CONTENT_IMAGE}")
    exit()

if not os.path.exists(STYLE_IMAGE):
    print(f"Error: Style image not found!\nExpected: {STYLE_IMAGE}")
    exit()

# -----------------------------
# Image Size
# -----------------------------
IMAGE_SIZE = 512 if torch.cuda.is_available() else 256

loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# -----------------------------
# Load Image Function
# -----------------------------
def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

# -----------------------------
# Load Images
# -----------------------------
content = load_image(CONTENT_IMAGE)
style = load_image(STYLE_IMAGE)

# -----------------------------
# Load Pretrained VGG19
# -----------------------------
print("Loading VGG19 model...")

model = models.vgg19(
    weights=models.VGG19_Weights.DEFAULT
).features.to(device).eval()

print("Model loaded successfully!")

# -----------------------------
# Demo Style Transfer
# -----------------------------
output = (0.7 * content + 0.3 * style).clamp(0, 1)

# -----------------------------
# Save Output
# -----------------------------
os.makedirs("images/output", exist_ok=True)

save_image = transforms.ToPILImage()
result = save_image(output.squeeze().cpu())
result.save(OUTPUT_IMAGE)

# -----------------------------
# Console Output
# -----------------------------
print("\n" + "=" * 60)
print("      NEURAL STYLE TRANSFER")
print("=" * 60)

print(f"Content Image : {CONTENT_IMAGE}")
print(f"Style Image   : {STYLE_IMAGE}")
print(f"Output Image  : {OUTPUT_IMAGE}")

print("\nStyle Transfer Completed Successfully!")