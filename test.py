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
# 1. 基础配置
# ===========================
st.set_page_config(page_title="Deepfake Probabilistic Detector", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2. 模型架构定义 (严格对应 Notebook)
# ===========================

# --- A. MC Dropout 架构 ---
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

# --- B. 贝叶斯/变分推断 (VI) 架构 ---
class BayesianLinear(nn.Module):
    # Notebook 中定义 prior_std=0.5
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
        # 重参数化技巧 (Reparameterization trick)
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
# 3. 核心工具函数
# ===========================

@st.cache_resource
def load_feature_extractor():
    """加载 Xception 特征提取器"""
    try:
        # 使用 'legacy_xception' 或 'xception'，忽略弃用警告
        model = timm.create_model("xception", pretrained=True, num_classes=0)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load feature extractor: {e}")
        return None

def safe_load_checkpoint(filepath):
    """
    兼容性加载函数：
    1. 自动处理 PyTorch 2.6+ 的 weights_only 参数报错
    2. 自动处理 CPU/GPU 设备映射
    """
    try:
        # 尝试使用 PyTorch 2.6+ 的新参数 weights_only=False
        return torch.load(filepath, map_location=DEVICE, weights_only=False)
    except TypeError:
        # 如果是旧版 PyTorch (不支持 weights_only)，则回退到普通加载
        return torch.load(filepath, map_location=DEVICE)
    except Exception as e:
        raise e

def load_head_model(filepath, architecture_type):
    """加载特定的分类头模型"""
    # 1. 初始化对应的模型结构
    if architecture_type == "mc_dropout":
        model = MCDropoutMLP(d_in=2048, hidden=256, p=0.3)
    elif architecture_type == "vi":
        model = VIModel(in_dim=2048)
    else:
        return None

    # 2. 加载权重
    if not os.path.exists(filepath):
        # 静默失败，在 UI 中提示即可
        return None

    try:
        checkpoint = safe_load_checkpoint(filepath)
        
        # 提取 state_dict
        state_dict = None
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # === 关键修复：处理 VI 模型的键值前缀不匹配问题 ===
        if architecture_type == "vi":
            # 检查权重是否包含 'layer.' 前缀
            has_layer_prefix = any(k.startswith("layer.") for k in state_dict.keys())
            # 如果模型定义有 self.layer 但权重里没有 (例如只保存了 BayesianLinear)，则手动添加前缀
            if not has_layer_prefix:
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 为所有键添加 'layer.' 前缀，使其匹配 VIModel 的定义
                    new_key = f"layer.{k}" if not k.startswith("layer.") else k
                    new_state_dict[new_key] = v
                state_dict = new_state_dict
        # ===============================================

        model.load_state_dict(state_dict, strict=False) # strict=False 允许一定的容错
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading {filepath}: {str(e)}")
        return None

def process_image(image):
    """
    图片预处理管线
    对应 Notebook 中的:
    T.Resize(342),
    T.CenterCrop(299),
    ...
    """
    # 修复：强制转换为 RGB，防止 PNG 透明通道或灰度图导致崩溃
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
    """通用的不确定性预测函数"""
    
    # 1. 提取特征
    with torch.no_grad():
        features = feature_extractor(img_tensor)

    # 2. 准备模型模式
    if model_type == "mc_dropout":
        head_model.train() # MC Dropout 需要开启训练模式以激活 Dropout
    else:
        head_model.eval()  # VI 模型自带随机前向传播

    # 3. 蒙特卡洛采样循环
    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = head_model(features)
            p = torch.sigmoid(logits).item()
            probs.append(p)

    probs = np.array(probs)
    return probs.mean(), probs.std()

# ===========================
# 4. Streamlit 应用程序逻辑
# ===========================

# --- 侧边栏：模型选择 ---
st.sidebar.header("🛠️ Model Configuration")
st.sidebar.write("Select models to run:")

# 定义可用模型列表
# 这里包含了您提到的所有三个模型文件
model_options = {
    "MC Dropout": {"file": "mc_dropout.pt", "type": "mc_dropout"},
    "Bayesian Linear": {"file": "bayesian_linear.pt", "type": "vi"}, # 假设这也是 VI 结构
    "Variational Inference": {"file": "variational_inference.pt", "type": "vi"}
}

# 默认全选，或者根据文件是否存在动态选择
available_defaults = [name for name, cfg in model_options.items() if os.path.exists(cfg["file"])]
if not available_defaults:
    available_defaults = ["MC Dropout"] # Fallback

selected_models = st.sidebar.multiselect(
    "Active Models", 
    options=list(model_options.keys()),
    default=available_defaults
)

# --- 主界面布局 ---
st.title("🕵️‍♂️ Bayesian Deepfake Detection")
st.markdown("Analyze images using **uncertainty quantification** to detect AI-generated content.")

# 初始化 Session State
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

# --- 示例图片选择区 ---
with st.expander("📂 Try a Sample Image", expanded=True):
    sample_dir = "sample_pics"
    
    # 检查文件夹是否存在
    if os.path.exists(sample_dir):
        valid_ext = ('.png', '.jpg', '.jpeg', '.webp')
        all_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(valid_ext)]
        
        if all_files:
            # 随机选择最多 4 张图片
            if 'random_samples' not in st.session_state:
                st.session_state.random_samples = random.sample(all_files, min(len(all_files), 4))
            
            # 创建列布局显示图片
            cols = st.columns(len(st.session_state.random_samples))
            for idx, file_name in enumerate(st.session_state.random_samples):
                file_path = os.path.join(sample_dir, file_name)
                with cols[idx]:
                    try:
                        # 立即转换为 RGB 防止预览报错
                        img = Image.open(file_path).convert('RGB')
                        st.image(img, use_container_width=True)
                        if st.button(f"Analyze Sample {idx+1}", key=f"btn_{idx}"):
                            st.session_state.selected_image = img
                    except Exception as e:
                        st.error(f"Bad image: {file_name}")
        else:
            st.warning(f"No valid images found in '{sample_dir}'.")
    else:
        st.info(f"Note: Create a folder named '{sample_dir}' and verify your images are inside.")

# --- 文件上传区 ---
st.divider()
uploaded_file = st.file_uploader("Or upload your own image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    try:
        st.session_state.selected_image = Image.open(uploaded_file).convert('RGB')
    except Exception as e:
        st.error(f"Error reading image: {e}")

# --- 分析执行区 ---
if st.session_state.selected_image is not None:
    # 布局：左侧显示图，右侧显示结果
    col_img, col_res = st.columns([1, 2])
    
    with col_img:
        st.image(st.session_state.selected_image, caption="Target Image", use_container_width=True)
    
    with col_res:
        st.subheader("Inference Results")
        
        if not selected_models:
            st.warning("Please select at least one model from the sidebar to start analysis.")
        else:
            # 1. 加载特征提取器 (只加载一次)
            with st.spinner("Loading Feature Extractor..."):
                feature_extractor = load_feature_extractor()
            
            if feature_extractor:
                # 2. 预处理图片
                img_tensor = process_image(st.session_state.selected_image)

                # 3. 循环运行选中的模型
                for model_name in selected_models:
                    config = model_options[model_name]
                    
                    # 加载模型头
                    head = load_head_model(config["file"], config["type"])
                    
                    if head:
                        # 运行推理
                        mean_p, std_dev = predict_uncertainty(
                            feature_extractor, head, img_tensor, config["type"]
                        )
                        
                        # 显示结果卡片
                        with st.container():
                            st.markdown(f"### 🧠 {model_name}")
                            m_col1, m_col2, m_col3 = st.columns(3)
                            
                            m_col1.metric("Fake Probability", f"{mean_p:.2%}")
                            m_col2.metric("Uncertainty (Std)", f"{std_dev:.4f}")
                            
                            # 结果解释
                            if mean_p > 0.5:
                                status = "FAKE"
                                color = "red"
                            else:
                                status = "REAL"
                                color = "green"
                                
                            m_col3.markdown(f"Prediction:\n#### :{color}[{status}]")
                            
                            # 可视化进度条
                            st.progress(float(mean_p))
                            
                            # 不确定性解释
                            if std_dev > 0.1:
                                st.caption("High uncertainty: The model is not sure about this image.")
                            else:
                                st.caption("Low uncertainty: The model is confident.")
                            
                            st.divider()
                    else:
                        # 仅当文件确实不存在时才报错
                        if os.path.exists(config["file"]):
                             st.error(f"❌ Failed to load **{model_name}**. Check logs for details.")
                        else:
                             st.warning(f"⚠️ File **{config['file']}** not found. Please upload it.")

