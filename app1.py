import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

# -----------------------------------------
# 1. Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="Cat vs Dog - ViT",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱🐶 Cat vs Dog Image Classification")
st.write("Vision Transformer (ViT) + PyTorch")

# -----------------------------------------
# 2. Device
# -----------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.info(f"Running on: {device}")

# -----------------------------------------
# 3. Classes
# -----------------------------------------

class_names = ["cats", "dogs"]

# -----------------------------------------
# 4. Load ViT model
# -----------------------------------------

@st.cache_resource
def load_model():

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=2
    )

    checkpoint = torch.load(
        "best_vit_cat_dog.pth",
        map_location=device
    )

    # Handle checkpoint format
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove module. prefix if present
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


model = load_model()

# -----------------------------------------
# 5. Image preprocessing
# -----------------------------------------

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------------
# 6. Upload image
# -----------------------------------------

uploaded_file = st.file_uploader(
    "Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------------
# 7. Prediction
# -----------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert image
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to GPU/CPU
    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = class_names[
        predicted.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )

    # -----------------------------------------
    # 8. Display result
    # -----------------------------------------

    st.success(
        f"Prediction: {predicted_class.upper()}"
    )

    st.metric(
        "Confidence",
        f"{confidence_value:.2f}%"
    )

    # -----------------------------------------
    # 9. Probability
    # -----------------------------------------

    st.subheader(
        "Prediction Probabilities"
    )

    for i, class_name in enumerate(
        class_names
    ):

        probability = (
            probabilities[0][i].item() * 100
        )

        st.write(
            f"{class_name}: {probability:.2f}%"
        )

        st.progress(
            int(probability)
        )