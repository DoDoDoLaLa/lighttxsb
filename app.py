import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import plotly.graph_objects as go
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas
from openai import OpenAI

# ==========================================
# 1. 页面与状态初始化
# ==========================================
st.set_page_config(
    page_title="SpectraMind 光电分析平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'spectral_data' not in st.session_state:
    st.session_state.spectral_data = None
if 'roi_stats' not in st.session_state:
    st.session_state.roi_stats = {}

# ==========================================
# 2. 科学仿真算法库
# ==========================================
def generate_pseudo_spectrum_curve(n_points=1000, n_peaks=5, noise_level=0.05):
    wavelengths = np.linspace(400, 1000, n_points)
    intensity = np.zeros_like(wavelengths)
    intensity += np.linspace(0.1, 0.3, n_points)
    intensity += 0.05 * np.sin(wavelengths / 100)

    peaks_info = []

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
# 3. AI 物理分析报告
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
                    "detected_elements": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "physical_interpretation": {"type": "string"},
                    "anomaly_detected": {"type": "boolean"},
                    "confidence_score": {"type": "number"}
                },
                "required": ["sample_quality","detected_elements","physical_interpretation","anomaly_detected","confidence_score"],
                "additionalProperties": False
            }
        }
    }

    prompt = f"""
请根据以下 ROI 统计数据生成物理分析报告:
{json.dumps(stats_json)}
"""

    try:
        completion = client.chat.completions.create(
            model=st.secrets["llm"]["model"],
            messages=[
                {"role":"system","content":"你是光谱分析专家，必须严格按 JSON Schema 输出结果。"},
                {"role":"user","content":prompt}
            ],
            response_format=report_schema
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 4. UI 布局与交互
# ==========================================
with st.sidebar:
    st.title("SpectraMind 仿真分析平台")
    st.markdown("---")
    integration_time = st.slider("积分时间 (ms)", 10, 2000, 100)
    laser_power = st.number_input("激光功率 (mW)", 0, 500, 50)
    gain_level = st.select_slider("增益等级", ["Low","Medium","High","Ultra"])
    st.markdown("---")
    st.subheader("绘图设置")
    plot_theme = st.selectbox("Plotly主题", ["plotly_dark","plotly_white","ggplot2"])
    show_peaks = st.checkbox("显示峰值标签", True)

tab_monitor, tab_vision, tab_report = st.tabs(["📈 光谱监测","📸 ROI分析","🧠 AI报告"])

# --- TAB 光谱监测 ---
with tab_monitor:
    st.markdown("### 光谱采集")
    if st.button("采集单帧光谱"):
        df_spec, peaks = generate_pseudo_spectrum_curve(noise_level=0.1 if gain_level=="High" else 0.02)
        st.session_state.spectral_data = df_spec
        st.success(f"采集完成！检测到 {len(peaks)} 个峰")

    if st.session_state.spectral_data is not None:
        df = st.session_state.spectral_data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Wavelength"], y=df["Intensity"], mode="lines"))
        fig.update_layout(template=plot_theme, xaxis_title="波长 (nm)", yaxis_title="强度 (a.u.)")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB ROI 分析 ---
with tab_vision:
    st.markdown("### ROI 可视化分析")
    st.info("在左侧图像上绘制矩形选取区域")
    original_image = create_synthetic_diffraction_image(800,600)
    CANVAS_W = 500
    orig_w,orig_h = original_image.size
    CANVAS_H = int(CANVAS_W*orig_h/orig_w)

    canvas_result = st_canvas(
        fill_color="rgba(255,0,0,0.2)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=original_image,
        update_streamlit=True,
        height=CANVAS_H,
        width=CANVAS_W,
        drawing_mode="rect",
        key="roi_canvas",
        display_toolbar=True
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][-1]
        scale_x = orig_w / CANVAS_W
        scale_y = orig_h / CANVAS_H

        x1 = int(obj["left"]*scale_x)
        y1 = int(obj["top"]*scale_y)
        w  = int(obj["width"]*obj["scaleX"]*scale_x)
        h  = int(obj["height"]*obj["scaleY"]*scale_y)
        x2,y2 = x1+w, y1+h

        arr = np.array(original_image)
        roi_slice = arr[y1:y2, x1:x2]
        st.image(roi_slice)

        mean_val = float(np.mean(roi_slice))
        std_val = float(np.std(roi_slice))
        st.session_state.roi_stats = {
            "mean_intensity": mean_val,
            "std_deviation": std_val,
            "pixel_count": int(w*h)
        }
        st.write(st.session_state.roi_stats)

# --- TAB AI 报告 ---
with tab_report:
    st.markdown("### AI 物理分析报告")
    if not st.session_state.roi_stats:
        st.warning("请先选择 ROI 区域")
    else:
        if st.button("生成报告"):
            report = get_ai_physical_report(st.session_state.roi_stats)
            st.json(report)
