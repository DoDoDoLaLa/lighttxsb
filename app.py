
import os
import json
import base64
import io
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

# OpenAI SDK (OpenAI-compatible) for Volcengine Ark / Doubao
from openai import OpenAI


# ----------------------------
# Configuration
# ----------------------------
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
DEFAULT_MODEL = os.getenv("ARK_MODEL", "ep-20260212130909-5kf5x")  # user-specified endpoint/model id

st.set_page_config(
    page_title="SpectraMind — 光谱智脑 (Demo)",
    page_icon="🔬",
    layout="wide"
)

# ----------------------------
# Helpers
# ----------------------------

@dataclass
class SimParams:
    int_time: int          # integration time (ms)
    laser_power: int       # 0-100
    gain: int              # 0-30 (noise scale)
    wavelength: int        # nm
    mode: str              # RGB / NIR
    denoise: bool

def _to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_simulation(img_rgb: np.ndarray, p: SimParams) -> np.ndarray:
    """
    Lightweight "hardware simulation" pipeline:
    - Integration time -> brightness (beta)
    - Laser power -> contrast (alpha)
    - Gain -> additive gaussian noise
    - Wavelength/mode -> channel filtering / NIR pseudo-color
    """
    img = img_rgb.copy()
    img_bgr = _to_bgr(img)

    # (1) Integration time -> brightness
    # baseline 1000ms, range 10-5000
    beta = (p.int_time - 1000) / 20.0  # simple mapping
    img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=beta)

    # (2) Laser power -> contrast
    alpha = 1.0 + (p.laser_power - 50) / 100.0  # 0.5 ~ 1.5
    img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=0)

    # Optional denoise (to make it feel "instrument-grade")
    if p.denoise:
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)

    # (3) Gain -> noise (higher gain, more noise)
    if p.gain > 0:
        noise = np.random.normal(0, p.gain, img_bgr.shape).astype(np.float32)
        img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # (4) Spectral filtering / mode
    if p.mode == "NIR":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # pseudo-thermal colormap for "NIR heatmap"
        img_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    else:
        # Simple channel gating for "wavelength"
        # 450nm -> emphasize blue; 650nm -> red; 850nm -> treat as near-IR-ish (grayscale)
        if p.wavelength <= 500:
            # keep blue channel mainly
            b, g, r = cv2.split(img_bgr)
            img_bgr = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])
        elif 600 <= p.wavelength <= 700:
            b, g, r = cv2.split(img_bgr)
            img_bgr = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
        elif p.wavelength >= 800:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return _to_rgb(img_bgr)

def roi_stats(img_rgb: np.ndarray, roi: Tuple[int,int,int,int]) -> Dict[str, float]:
    x1, y1, x2, y2 = roi
    x1, x2 = sorted([max(0, x1), max(0, x2)])
    y1, y2 = sorted([max(0, y1), max(0, y2)])
    x2 = min(img_rgb.shape[1]-1, x2)
    y2 = min(img_rgb.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        patch = img_rgb
    else:
        patch = img_rgb[y1:y2, x1:x2]

    gray = cv2.cvtColor(_to_bgr(patch), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = float(np.mean(gray))
    std = float(np.std(gray) + 1e-6)
    # Simple SNR proxy (mean/std), convert to dB
    snr = mean / std
    snr_db = 20.0 * np.log10(max(snr, 1e-6))
    # Michelson contrast proxy
    mn, mx = float(np.min(gray)), float(np.max(gray))
    contrast = (mx - mn) / (mx + mn + 1e-6)
    return {
        "roi_mean": mean,
        "roi_std": std,
        "snr_db": float(snr_db),
        "contrast": float(contrast),
        "roi_min": mn,
        "roi_max": mx,
        "roi_area_px": float(gray.size),
    }

def synth_spectrum(stats: Dict[str, float], p: SimParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fake-but-coherent spectrum:
    - Peak position tied to selected wavelength
    - Peak height tied to roi_mean & laser_power
    - Noise tied to gain & roi_std
    """
    wl = np.linspace(400, 900, 501)
    peak = float(p.wavelength)
    height = (stats["roi_mean"] / 255.0) * (0.5 + p.laser_power/200.0)
    width = 35.0 + 0.6 * stats["roi_std"]
    baseline = 0.08 + 0.15 * (stats["contrast"])
    y = baseline + height * np.exp(-0.5*((wl-peak)/width)**2)
    noise_scale = 0.02 + 0.002 * p.gain
    y = y + np.random.normal(0, noise_scale, size=wl.shape)
    y = np.clip(y, 0, None)
    return wl, y

def encode_image_data_url(img_rgb: np.ndarray, fmt: str = "png") -> str:
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format=fmt.upper())
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.lower() == "png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def build_payload(p: SimParams, stats: Dict[str, float]) -> Dict[str, Any]:
    return {
        "mode": p.mode,
        "params": {
            "integration_time_ms": p.int_time,
            "laser_power_pct": p.laser_power,
            "gain_level": p.gain,
            "wavelength_nm": p.wavelength,
            "denoise": p.denoise
        },
        "roi_stats": stats
    }

def call_ark_llm(model: str, api_key: str, base_url: str, img_rgb: np.ndarray, payload: Dict[str, Any]) -> str:
    """
    Calls Volcengine Ark (OpenAI-compatible) via openai-python SDK.
    Uses Responses API with multimodal input (image + text).
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    img_url = encode_image_data_url(img_rgb, fmt="png")

    prompt = f"""
你是“光谱智脑 SpectraMind”的物理分析引擎。下面给你：1) 仿真处理后的图像；2) 仪器参数与ROI统计（JSON）。
请输出一份**结构化的实验分析报告**，必须包含以下字段（用Markdown标题）：
1) Summary（3条要点）
2) Observations（基于给定JSON指标逐条解释，不要凭空编造）
3) Hypotheses（给出2-3条可能解释/材料类别/现象机制，每条标注置信度：Low/Med/High，并说明依据来自哪些指标）
4) Next Steps（3-5条可执行的下一步实验/参数扫描建议）
5) Limitations（说明这是软件仿真/非真实定量测量，避免夸大）

输入JSON如下：
```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```
"""

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": img_url},
                {"type": "input_text", "text": prompt}
            ]
        }]
    )

    # openai-python Responses returns .output_text in recent versions
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # Fallback: try to extract from output list
    try:
        out = resp.output[0].content[0].text
        return out
    except Exception:
        return str(resp)

# ----------------------------
# UI
# ----------------------------

st.title("🔬 SpectraMind（光谱智脑）— 端云协同光电物质分析（Demo）")
st.caption("目标：**看起来像科研仪器软件** + **可调用方舟/豆包 OpenAI 兼容 API**，实现低成本高观感演示。")

with st.sidebar:
    st.subheader("Virtual Hardware Console")
    st.markdown("调参即“操作仪器”。中间画面与曲线会实时响应。")

    int_time = st.slider("Integration Time (ms)", 10, 5000, 1200, step=10)
    laser_power = st.slider("Laser Power (%)", 0, 100, 65, step=1)
    gain = st.slider("Detector Gain (noise level)", 0, 30, 12, step=1)
    wavelength = st.select_slider("Wavelength (nm)", options=[450, 650, 850], value=850)
    mode = st.radio("Mode", ["RGB", "NIR"], index=1, horizontal=True)
    denoise = st.toggle("Denoise (NLMeans)", value=True)

    st.divider()
    st.subheader("AI / API")
    model = st.text_input("Model / Endpoint", value=DEFAULT_MODEL)
    base_url = st.text_input("Base URL", value=ARK_BASE_URL)
    api_key = st.text_input("ARK_API_KEY", value=os.getenv("ARK_API_KEY", ""), type="password")
    st.caption("建议把 KEY 放在环境变量：`export ARK_API_KEY=...`")

p = SimParams(int_time=int_time, laser_power=laser_power, gain=gain, wavelength=wavelength, mode=mode, denoise=denoise)

col1, col2, col3 = st.columns([1.1, 1.1, 1.0], gap="large")

# Data source
with col1:
    st.subheader("Input")
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("请上传一张图片作为“样品图像”。也可以随便用一张风景/物体图，演示效果依然很强。")
        st.stop()

    file_bytes = np.frombuffer(up.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("无法读取图片，请换一张。")
        st.stop()
    img_rgb = _to_rgb(img_bgr)

    st.image(img_rgb, caption="原始输入图像", use_container_width=True)

# Process & ROI
with col2:
    st.subheader("Instrument View")
    sim_rgb = apply_simulation(img_rgb, p)
    st.image(sim_rgb, caption="仿真处理后（仪器输出）", use_container_width=True)

    st.markdown("**ROI（感兴趣区域）**")
    roi = (0, 0, sim_rgb.shape[1], sim_rgb.shape[0])

    if HAS_CANVAS:
        st.caption("拖拽画框选择ROI（若画不出来，可能是依赖未安装；会自动回退全图ROI）。")
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=2,
            stroke_color="rgba(255, 0, 0, 0.9)",
            background_image=Image.fromarray(sim_rgb),
            update_streamlit=True,
            height=min(500, sim_rgb.shape[0]),
            width=min(700, sim_rgb.shape[1]),
            drawing_mode="rect",
            key="canvas",
        )
        if canvas.json_data and len(canvas.json_data.get("objects", [])) > 0:
            obj = canvas.json_data["objects"][-1]
            x1 = int(obj["left"])
            y1 = int(obj["top"])
            x2 = int(x1 + obj["width"])
            y2 = int(y1 + obj["height"])
            roi = (x1, y1, x2, y2)
    else:
        st.caption("未检测到 streamlit-drawable-canvas，ROI 默认使用全图。")

    stats = roi_stats(sim_rgb, roi)

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Intensity", f'{stats["roi_mean"]:.1f}')
    m2.metric("SNR (dB)", f'{stats["snr_db"]:.1f}')
    m3.metric("Contrast", f'{stats["contrast"]:.3f}')

# Spectrum + AI report
with col3:
    st.subheader("Live Signal Oscilloscope")
    wl, y = synth_spectrum(stats, p)
    fig = plt.figure()
    plt.plot(wl, y)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Intensity (a.u.)")
    plt.title("Synthesized Spectrum (ROI-driven)")
    st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("AI Physics Report")

    payload = build_payload(p, stats)
    with st.expander("Payload (JSON)", expanded=False):
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

    if st.button("Start Analysis", type="primary", use_container_width=True):
        if not api_key:
            st.error("缺少 ARK_API_KEY。请在侧边栏填写，或设置环境变量。")
        else:
            with st.status("Running pipeline…", expanded=True) as s:
                st.write("Acquire → Simulate → ROI Stats → Spectrum → LLM Report")
                try:
                    report = call_ark_llm(model=model, api_key=api_key, base_url=base_url, img_rgb=sim_rgb, payload=payload)
                    s.update(label="Done", state="complete")
                    st.markdown(report)
                except Exception as e:
                    s.update(label="Failed", state="error")
                    st.exception(e)

st.divider()
st.caption("SpectraMind Demo • Streamlit + OpenCV/NumPy + Matplotlib + OpenAI SDK (Ark OpenAI-compatible)")
