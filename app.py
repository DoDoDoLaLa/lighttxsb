import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import plotly.graph_objects as go
from PIL import Image
import cv2
import io

# ==========================================
# 1. 核心修复补丁 (Monkey Patch)
# ==========================================
try:
    from streamlit.elements import image as st_image
    if not hasattr(st_image, 'image_to_url'):
        from streamlit.elements.lib import image_utils
        st_image.image_to_url = image_utils.image_to_url
except ImportError:
    pass

from streamlit_drawable_canvas import st_canvas
from openai import OpenAI

# ==========================================
# 2. 页面与状态初始化
# ==========================================
st.set_page_config(
    page_title="SpectraMind 光电分析平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'spectral_data' not in st.session_state:
    st.session_state.spectral_data = None
if 'roi_stats' not in st.session_state:
    st.session_state.roi_stats = {}

# ==========================================
# 3. 安全鉴权模块
# ==========================================
def check_authentication():
    def password_entered():
        if st.session_state["password_input"] == st.secrets["general"]["app_password"]:
            st.session_state.authenticated = True
            del st.session_state["password_input"]
        else:
            st.session_state.authenticated = False
            st.error("密码错误，请重试。")
    if st.session_state.authenticated:
        return True
    st.markdown("## 🔐 系统访问受限")
    st.text_input(
        "请输入访问密钥", 
        type="password", 
        on_change=password_entered, 
        key="password_input"
    )
    return False

if not check_authentication():
    st.stop()

# ==========================================
# 4. 科学仿真算法库
# ==========================================
def generate_pseudo_spectrum_curve(n_points=1000, n_peaks=5, noise_level=0.05):
    wavelengths = np.linspace(400, 1000, n_points)
    intensity = np.zeros_like(wavelengths)
    intensity += np.linspace(0.1, 0.3, n_points)
    intensity += 0.05 * np.sin(wavelengths / 100)

    peaks_info = []  # ✅ 修复这里

    np.random.seed(int(time.time()))
    for _ in range(n_peaks):
        center = np.random.uniform(450, 950)
        width = np.random.uniform(5, 15)
        amplitude = np.random.uniform(0.3, 1.0)
        peak_signal = amplitude * np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))
        intensity += peak_signal
        peaks_info.append(f"{center:.1f}nm")

    noise = np.random.normal(0, noise_level, n_points)
    intensity += noise
    intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

    return pd.DataFrame({"Wavelength": wavelengths, "Intensity": intensity}), peaks_info

def create_synthetic_diffraction_image(width=800, height=600):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        hue = int((i / width) * 180)
        cv2.line(img, (i, 100), (i, 500), (hue, 255, 255), 1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.GaussianBlur(img, (51, 51), 0)

    lines = [(200, (0, 255, 255)), (450, (0, 255, 0)), (600, (0, 0, 255))]
    for x, color in lines:
        cv2.line(img, (x, 50), (x, 550), color, 4)
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ==========================================
# 5. LLM 报告生成
# ==========================================
def get_ai_physical_report(stats_json):
    client = OpenAI(
        api_key=st.secrets["llm"]["api_key"],
        base_url=st.secrets["llm"]["base_url"]
    )

    report_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "spectroscopy_analysis_report",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "sample_quality": {"type": "string"},
                    "detected_elements": {"type": "array", "items": {"type": "string"}},
                    "physical_interpretation": {"type": "string"},
                    "anomaly_detected": {"type": "boolean"},
                    "confidence_score": {"type": "number"}
                },
                "required": ["sample_quality", "detected_elements", "physical_interpretation", "anomaly_detected", "confidence_score"],
                "additionalProperties": False
            }
        }
    }

    prompt = f"""
你是一名资深光谱分析专家。请根据以下 ROI 统计数据生成物理分析报告:
{json.dumps(stats_json)}
"""

    try:
        completion = client.chat.completions.create(
            model=st.secrets["llm"]["model"],
            messages=[
                {"role": "system", "content": "请严格按照 JSON Schema 输出，不要额外文字。"},
                {"role": "user", "content": prompt}
            ],
            response_format=report_schema
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 6. UI 界面与交互
# ==========================================
with st.sidebar:
    st.title("SpectraMind 🔬")
    st.caption("光电分析 仿真 & 报告 系统")
    st.markdown("---")
    st.subheader("⚙️ 模拟参数")
    integration_time = st.slider("积分时间 (ms)", 10, 2000, 100)
    laser_power = st.number_input("激光功率 (mW)", 0, 500, 50)
    gain_level = st.select_slider("增益等级", ["Low", "Medium", "High", "Ultra"])
    st.markdown("---")
    st.subheader("🎨 图形配置")
    plot_theme = st.selectbox("图表主题", ["plotly_dark", "plotly_white", "ggplot2"])
    show_peaks = st.checkbox("显示峰值标签", True)

tab_monitor, tab_vision, tab_report = st.tabs(["📈 光谱监测", "📸 ROI分析", "📝 AI报告"])

# --- TAB 光谱监测 ---
with tab_monitor:
    st.markdown("### 📊 光谱采集")
    if st.button("采集单帧光谱"):
        df_spec, peaks = generate_pseudo_spectrum_curve(noise_level=0.1 if gain_level == "High" else 0.02)
        st.session_state.spectral_data = df_spec
        st.success(f"采集完成！检测到 {len(peaks)} 个特征峰")

    if st.session_state.spectral_data is not None:
        df = st.session_state.spectral_data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Wavelength"], y=df["Intensity"], mode='lines', name='光谱'))
        fig.update_layout(template=plot_theme, xaxis_title="波长 (nm)", yaxis_title="强度 (a.u.)")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB ROI 分析 ---
with tab_vision:
    st.markdown("### 📍 ROI 区域选择")
    st.info("在左侧图像上绘制矩形以选取分析区域")

    original_image = create_synthetic_diffraction_image(800, 600)
    CANVAS_W = 500
    orig_w, orig_h = original_image.size
    CANVAS_H = int(CANVAS_W * orig_h / orig_w)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=original_image,
        update_streamlit=True,
        height=CANVAS_H,
        width=CANVAS_W,
        drawing_mode="rect",
        key="canvas"
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][-1]
        scale_x = orig_w / CANVAS_W
        scale_y = orig_h / CANVAS_H

        rect_x = int(obj["left"] * scale_x)
        rect_y = int(obj["top"] * scale_y)
        rect_w = int(obj["width"] * obj["scaleX"] * scale_x)
        rect_h = int(obj["height"] * obj["scaleY"] * scale_y)

        x1, y1 = rect_x, rect_y
        x2, y2 = x1 + rect_w, y1 + rect_h

        img_arr = np.array(original_image)

        if x2 > x1 and y2 > y1:
            roi_slice = img_arr[y1:y2, x1:x2]
            st.image(roi_slice, caption=f"ROI {rect_w}×{rect_h} px")

            mean_val = float(np.mean(roi_slice))
            std_val = float(np.std(roi_slice))
            max_val = float(np.max(roi_slice))

            st.session_state.roi_stats = {
                "roi_position": [x1, y1, x2, y2],
                "mean_intensity": mean_val,
                "std_deviation": std_val,
                "max_intensity": max_val,
                "pixel_count": int(rect_w * rect_h)
            }
            st.write(st.session_state.roi_stats)

# --- TAB AI 报告 ---
with tab_report:
    st.markdown("### 🧠 AI 物理分析报告")

    if not st.session_state.roi_stats:
        st.warning("请先在 ROI 分析中选择一个区域")
    else:
        if st.button("生成 AI 报告"):
            report = get_ai_physical_report(st.session_state.roi_stats)
            st.json(report)
