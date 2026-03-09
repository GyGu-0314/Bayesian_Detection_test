import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import timm
import numpy as np

# ===========================
# 1. Device Configuration
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# 2. Model Definition
#    (Must match training architecture exactly)
# ===========================
class MCDropoutMLP(nn.Module):
    def __init__(self, d_in=2048, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


# ===========================
# 3. Load Models
# ===========================
@st.cache_resource
def load_models():
    # --- A. Feature Extractor (Xception) ---
    # We use timm to load the pretrained Xception model.
    # This acts as the "eyes" to convert the image into a 2048-dim vector.
    feature_extractor = timm.create_model("xception", pretrained=True, num_classes=0)
    feature_extractor.to(DEVICE)
    feature_extractor.eval()  # Freeze feature extractor

    # --- B. Classifier (MC Dropout Head) ---
    # This is the "brain" you trained.
    classifier = MCDropoutMLP(d_in=2048, hidden=256, p=0.3)

    # Load the weights from your .pt file
    # Ensure 'model.pt' is in the same directory as app.py
    try:
        checkpoint = torch.load("model.pt", map_location=DEVICE)

        # Handle the dictionary structure based on your saving code
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            classifier.load_state_dict(checkpoint["state_dict"])
        else:
            classifier.load_state_dict(checkpoint)

    except FileNotFoundError:
        st.error("Error: 'model.pt' not found. Please upload your model weights.")
        st.stop()

    classifier.to(DEVICE)
    return feature_extractor, classifier


# ===========================
# 4. Image Preprocessing
# ===========================
def process_image(image):
    # Standard ImageNet normalization required by Xception
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    return transform(image).unsqueeze(0).to(DEVICE)


# ===========================
# 5. MC Dropout Inference Logic
# ===========================
def mc_predict(feature_extractor, classifier, img_tensor, n_samples=50):
    # 1. Extract features (Deterministic)
    with torch.no_grad():
        features = feature_extractor(img_tensor)  # Shape: [1, 2048]

    # 2. Enable Dropout for Monte Carlo sampling
    classifier.train()

    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = classifier(features)
            p = torch.sigmoid(logits).item()
            probs.append(p)

    probs = np.array(probs)
    # Return mean probability and standard deviation (uncertainty)
    return probs.mean(), probs.std()


# ===========================
# 6. Streamlit UI Layout
# ===========================
st.set_page_config(page_title="Deepfake Probabilistic Detector")

st.title("Deepfake Detector (MC Dropout)")
st.markdown("""
This tool uses Monte Carlo Dropout to analyze images.
Instead of a simple Yes/No, it provides the **probability** of the image being manipulated by AI tools 
and the **uncertainty** of the model's prediction.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    # Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Input Image', width=350)

    # Load Models
    with st.spinner('Initializing models...'):
        feature_extractor, classifier = load_models()

    # Run Inference
    if st.button('Analyze Image'):
        with st.spinner('Running Monte Carlo sampling (50 passes)...'):
            img_tensor = process_image(image)
            mean_prob, std = mc_predict(feature_extractor, classifier, img_tensor)

        # --- Results Display ---
        st.divider()
        st.subheader("Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.info("**Probability Score**")
            st.markdown(f"# {mean_prob:.2%}")

            # Visual progress bar for probability
            st.progress(mean_prob)

        with col2:
            st.warning("**Model Uncertainty (Std Dev)**")
            st.markdown(f"# {std:.4f}")

            # Contextual help based on uncertainty
            if std < 0.05:
                st.caption("The model is very confident in this score.")
            elif std < 0.15:
                st.caption("Moderate uncertainty.")
            else:
                st.caption("High uncertainty. The model is confused.")

        # Optional: Raw Data Dropdown
        with st.expander("See Interpretation Guide"):
            st.markdown("""
            - **Probability**: The likelihood that the image is manipulated.
            - **Uncertainty**: How much the model 'wavered' during analysis. High uncertainty often means the image contains artifacts the model hasn't seen before (Out-of-Distribution).
            """)