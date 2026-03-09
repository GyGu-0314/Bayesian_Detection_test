import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import timm
import numpy as np
import os
import random

# ===========================
# 1. Configuration & Styling
# ===========================
st.set_page_config(page_title="Deepfake Probabilistic Detector", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2. Model Architectures
#    (Must match your training code exactly)
# ===========================

# --- A. MC Dropout Architecture ---
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

# --- B. Bayesian/VI Architecture ---
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=0.5):
        super().__init__()
        self.w_mu  = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.b_mu  = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.full((out_features,), -3.0))
        self.prior_std = prior_std

    def forward(self, x):
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)
        # Reparameterization trick
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        return F.linear(x, w, b)

class VIModel(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.layer = BayesianLinear(in_dim, 1)

    def forward(self, x):
        return self.layer(x)

# ===========================
# 3. Utility Functions
# ===========================

@st.cache_resource
def load_feature_extractor():
    """Loads the Xception backbone once."""
    # Using 'legacy_xception' to suppress timm warnings if necessary, 
    # but 'xception' works fine usually.
    model = timm.create_model("xception", pretrained=True, num_classes=0)
    model.to(DEVICE)
    model.eval()
    return model

def load_head_model(filepath, architecture_type):
    """Loads a specific classification head based on type."""
    # 1. Initialize the correct architecture
    if architecture_type == "mc_dropout":
        model = MCDropoutMLP(d_in=2048, hidden=256, p=0.3)
    elif architecture_type == "vi":
        model = VIModel(in_dim=2048)
    else:
        return None

    # 2. Load weights
    try:
        # FIX: weights_only=False needed for PyTorch 2.6+ compatibility with older/complex checkpoints
        checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
        
        # Handle state_dict key wrapper if present
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
             # Try loading dict directly if it matches keys
             try:
                 model.load_state_dict(checkpoint)
             except:
                 # If checkpoint has extra keys (like optimizer), filter or warn
                 pass 
        else:
            # Fallback for raw state_dict
            model.load_state_dict(checkpoint)
            
        model.to(DEVICE)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None

def process_image(image):
    """Preprocessing pipeline."""
    # FIX: Convert RGBA/Grayscale to RGB to prevent channel mismatch errors
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = T.Compose([
        T.Resize(342),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def predict_uncertainty(feature_extractor, head_model, img_tensor, model_type, n_samples=50):
    """Generic inference function for both MC Dropout and VI."""
    
    # 1. Extract Features
    with torch.no_grad():
        features = feature_extractor(img_tensor)

    # 2. Prepare Model Mode
    if model_type == "mc_dropout":
        head_model.train() # Force dropout on
    else:
        head_model.eval()  # VI model has stochastic forward pass by default

    # 3. Sampling Loop
    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = head_model(features)
            p = torch.sigmoid(logits).item()
            probs.append(p)

    probs = np.array(probs)
    return probs.mean(), probs.std()

# ===========================
# 4. App Logic
# ===========================

# --- Sidebar: Model Selection ---
st.sidebar.header("Model Configuration")
st.sidebar.write("Select models to run:")

# Define your available models here
model_options = {
    "MC Dropout": {"file": "mc_dropout.pt", "type": "mc_dropout"},
    "Bayesian Linear": {"file": "bayesian_linear.pt", "type": "vi"},
    "Variational Inference": {"file": "variational_inference.pt", "type": "vi"}
}

selected_models = st.sidebar.multiselect(
    "Active Models", 
    options=list(model_options.keys()),
    default=["MC Dropout"]
)

# --- Main Layout ---
st.title("Bayesian Deepfake Detection")
st.markdown("Analyze images using uncertainty quantification to detect AI-generated content.")

# Initialize Session State for Sample Image
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

# --- Sample Images Section ---
with st.expander("📂 Try a Sample Image", expanded=True):
    sample_dir = "sample_pics"
    if os.path.exists(sample_dir):
        # Get list of valid images
        valid_ext = ('.png', '.jpg', '.jpeg', '.webp')
        all_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(valid_ext)]
        
        if all_files:
            # Randomly select up to 4 images
            if 'random_samples' not in st.session_state:
                st.session_state.random_samples = random.sample(all_files, min(len(all_files), 4))
            
            cols = st.columns(len(st.session_state.random_samples))
            for idx, file_name in enumerate(st.session_state.random_samples):
                file_path = os.path.join(sample_dir, file_name)
                with cols[idx]:
                    # FIX: Force RGB conversion immediately on load
                    img = Image.open(file_path).convert('RGB')
                    # FIX: Use use_container_width instead of deprecated use_column_width
                    st.image(img, use_container_width=True)
                    if st.button(f"Analyze Sample {idx+1}", key=f"btn_{idx}"):
                        st.session_state.selected_image = img
        else:
            st.warning("No images found in 'sample_pics' folder.")
    else:
        st.warning("Folder 'sample_pics' not found.")

# --- File Uploader ---
st.divider()
uploaded_file = st.file_uploader("Or upload your own image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    # FIX: Force RGB conversion on upload
    st.session_state.selected_image = Image.open(uploaded_file).convert('RGB')

# --- Analysis Pipeline ---
if st.session_state.selected_image is not None:
    # Display the active image
    col_img, col_res = st.columns([1, 2])
    
    with col_img:
        st.image(st.session_state.selected_image, caption="Target Image", use_container_width=True)
    
    with col_res:
        st.subheader("Inference Results")
        
        if not selected_models:
            st.error("Please select at least one model from the sidebar.")
        else:
            # Load Backbone
            with st.spinner("Loading Feature Extractor..."):
                feature_extractor = load_feature_extractor()
            
            img_tensor = process_image(st.session_state.selected_image)

            # Loop through selected models
            for model_name in selected_models:
                config = model_options[model_name]
                
                # Load specific head
                head = load_head_model(config["file"], config["type"])
                
                if head:
                    # Run Inference
                    mean_p, std_dev = predict_uncertainty(
                        feature_extractor, head, img_tensor, config["type"]
                    )
                    
                    # Display metrics
                    with st.container():
                        st.markdown(f"### 🧠 {model_name}")
                        m_col1, m_col2, m_col3 = st.columns(3)
                        
                        m_col1.metric("Fake Probability", f"{mean_p:.2%}")
                        m_col2.metric("Uncertainty (Std)", f"{std_dev:.4f}")
                        
                        # Interpretation
                        if mean_p > 0.5:
                            status = "FAKE"
                            color = "red"
                        else:
                            status = "REAL"
                            color = "green"
                            
                        m_col3.markdown(f"Prediction: :{color}[**{status}**]")
                        
                        # Visual Bar
                        st.progress(mean_p)
                        st.divider()
                else:
                    st.error(f"Could not load **{config['file']}**. Ensure the file exists in the root directory.")
