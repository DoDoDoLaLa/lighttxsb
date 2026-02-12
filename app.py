import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import cv2
import io
import base64

# ==========================================
# 1. 强力修复补丁 (必须放在最前面)
# ==========================================
# 修复 Streamlit 1.35+ 导致 st_canvas 崩溃的问题
try:
    from streamlit.elements import image as st_image
    from streamlit.elements.lib import image_utils
    
    # 强制覆盖 Streamlit 内部的 image_to_url 函数
    def custom_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="JPEG", image_id=None):
        # 将图片转为 Base64 字符串，绕过 Streamlit 的内部检查
        if isinstance(image, (Image.Image, np.ndarray)):
            # 如果是 numpy 数组，先转 PIL
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            buffered = io.BytesIO()
            fmt = output_format if output_format else "PNG"
            try:
                image.save(buffered, format=fmt)
            except Exception:
                image.save(buffered, format="PNG")
                fmt = "PNG"
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/{fmt.lower()};base64,{img_str}"
        return ""

    # 应用补丁
    st_image.image_to_url = custom_image_to_url
    if hasattr(image_utils, 'image_to_url'):
        image_utils.image_to_url = custom_image_to_url
except ImportError:
    pass

# 补丁必须在导入 st_canvas 之前执行
from streamlit_drawable_canvas import st_canvas
from openai import OpenAI

# ==========================================
# 2. 页面配置
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
# 3. 辅助函数：灵活读取 Secrets (修复图2/3的问题)
# ==========================================
def get_secret(key):
    """
    尝试从 st.secrets 中读取配置，兼容有 [llm] 和没有 [llm] 的情况
    """
    # 1. 尝试从 [llm] 节点读取 (代码里的标准写法)
    if "llm" in st.secrets and key in st.secrets["llm"]:
        return st.secrets["llm"][key]
    # 2. 尝试从根节点读取 (兼容您截图中的写法)
    if key in st.secrets:
        return st.secrets[key]
    return None

# ==========================================
# 4. 算法与仿真
# ==========================================
def generate_pseudo_spectrum_curve(n_points=1000, n_peaks=5, noise_level=0.05):
    wavelengths = np.linspace(400, 1000, n_points)
    intensity = np.zeros_like(wavelengths)
    intensity += np.linspace(0.1, 0.3, n_points) 
    intensity += 0.05 * np.sin(wavelengths / 100)
    
    np.random.seed(int(time.time()))
    peaks_info = []
    
    for _ in range(n_peaks):
        center = np.random.uniform(450, 950)
        width = np.random.uniform(5, 15)
        amplitude = np.random.uniform(0.3, 1.0)
        peak_signal = amplitude * np.exp(-((wavelengths - center)**2) / (2 * width**2))
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
        overlay = img.copy()
        cv2.line(overlay, (x, 50), (x, 550), color, 20)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ==========================================
# 5. LLM 接口
# ==========================================
def get_ai_physical_report(stats_json):
    api_key = get_secret("api_key")
    base_url = get_secret("base_url")
    model = get_secret("model")

    # 如果读取不到配置，返回错误
    if not api_key or not base_url or not model:
        return {"error": "未读取到 Secrets 配置。请检查 Streamlit Cloud 设置。", "anomaly_detected": True, "physical_interpretation": "配置缺失", "confidence_score": 0.0, "detected_elements": [], "sample_quality": "Critical"}

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    report_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "spectroscopy_analysis_report",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "sample_quality": {"type": "string", "enum": ["Excellent", "Good", "Noisy", "Critical"]},
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
    你是一名资深光谱分析专家。请根据以下从 ROI 区域提取的统计数据，生成一份物理分析报告。
    输入数据: {json.dumps(stats_json)}
    """

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful scientific assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format=report_schema
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "anomaly_detected": True, "physical_interpretation": "API Error", "confidence_score": 0.0, "detected_elements": [], "sample_quality": "Critical"}

# ==========================================
# 6. UI 布局
# ==========================================
with st.sidebar:
    st.title("SpectraMind 🔬")
    st.caption("v1.2.1 | 修复 Secrets & Columns") 
    st.markdown("---")
    integration_time = st.slider("积分时间 (ms)", 10, 2000, 100)
    gain_level = st.select_slider("增益等级", options=["Low", "Medium", "High", "Ultra"])
    st.markdown("---")
    plot_theme = st.selectbox("图表主题", ["plotly_dark", "plotly_white"])

tab_monitor, tab_vision, tab_report = st.tabs(["📈 光谱监测", "👁️ ROI 视觉分析", "📝 智能报告"])

# --- TAB 1 ---
with tab_monitor:
    st.markdown("### 实时光谱采集流")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("峰值波长", "656.3 nm")
    col2.metric("信噪比 (SNR)", "45.2 dB")
    col3.metric("系统温度", "23.1 °C")
    col4.metric("采集帧率", "12 fps")
    
    if st.button("采集单帧光谱", type="primary") or st.session_state.spectral_data is None:
        with st.spinner("读取数据..."):
            time.sleep(0.3)
            df_spec, peaks = generate_pseudo_spectrum_curve(noise_level=0.1 if gain_level=="High" else 0.02)
            st.session_state.spectral_data = df_spec
            st.toast(f"采集完成！检测到 {len(peaks)} 个特征峰", icon="✅")
    
    if st.session_state.spectral_data is not None:
        df = st.session_state.spectral_data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Wavelength'], y=df['Intensity'], mode='lines', fill='tozeroy'))
        fig.update_layout(template=plot_theme, height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2 ---
with tab_vision:
    st.markdown("### 高光谱图像 ROI 提取")
    
    uploaded_file = st.file_uploader("📤 上传实验图像 (支持 PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            st.toast("图像加载成功！", icon="🖼️")
        except Exception as e:
            st.error(f"图像解析失败: {e}")
            original_image = create_synthetic_diffraction_image(800, 600)
    else:
        original_image = create_synthetic_diffraction_image(800, 600)
    
    col_canvas, col_result = st.columns([1.5, 1])
    
    with col_canvas:
        CANVAS_WIDTH = 500
        orig_w, orig_h = original_image.size
        CANVAS_HEIGHT = int(CANVAS_WIDTH * orig_h / orig_w)
        
        st.markdown(f"**画布视图** ({orig_w}x{orig_h})")
        
        # 调用绘图组件
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=original_image,
            update_streamlit=True,
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            drawing_mode="rect",
            key="roi_canvas",
            display_toolbar=True
        )

    with col_result:
        st.markdown("**ROI 统计分析**")
        
        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][-1]
            
            scale_x = orig_w / CANVAS_WIDTH
            scale_y = orig_h / CANVAS_HEIGHT
            
            rect_x = int(obj["left"] * scale_x)
            rect_y = int(obj["top"] * scale_y)
            rect_w = int(obj["width"] * obj["scaleX"] * scale_x)
            rect_h = int(obj["height"] * obj["scaleY"] * scale_y)
            
            img_arr = np.array(original_image)
            y1, y2 = max(0, rect_y), min(orig_h, rect_y + rect_h)
            x1, x2 = max(0, rect_x), min(orig_w, rect_x + rect_w)
            
            if x2 > x1 and y2 > y1:
                roi_slice = img_arr[y1:y2, x1:x2]
                st.image(roi_slice, caption=f"ROI ({x2-x1}x{y2-y1})")
                
                mean_val = float(np.mean(roi_slice))
                std_val = float(np.std(roi_slice))
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("均值", f"{mean_val:.1f}")
                m_col2.metric("标准差", f"{std_val:.1f}")
                
                st.session_state.roi_stats = {
                    "roi_position": [x1, y1, x2, y2],
                    "mean_intensity": mean_val,
                    "std_deviation": std_val,
                    "max_intensity": float(np.max(roi_slice)),
                    "pixel_count": int((x2-x1)*(y2-y1))
                }
            else:
                st.warning("选区太小或超出边界")
        else:
            st.info("请在左图绘制矩形")

# --- TAB 3 ---
with tab_report:
    st.markdown("### 📝 AI 物理分析报告")
    
    if not st.session_state.roi_stats:
        st.warning("⚠️ 请先在 'ROI 视觉分析' 页面选取区域")
    else:
        st.json(st.session_state.roi_stats, expanded=False)
        
        if st.button("🚀 生成报告", type="primary"):
            with st.status("AI 正在分析物理特征...", expanded=True) as status:
                report_data = get_ai_physical_report(st.session_state.roi_stats)
                if "error" in report_data:
                    status.update(label="失败", state="error")
                    st.error(report_data["error"])
                else:
                    status.update(label="完成", state="complete")
                    st.divider()
                    # 修复: 明确指定列数为2，解决 TypeError
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("置信度", f"{report_data.get('confidence_score', 0)*100:.1f}%")
                        st.info(f"质量评估: {report_data.get('sample_quality')}")
                    with c2:
                        st.markdown(f"**结论:** {report_data.get('physical_interpretation')}")
                        st.write(f"**元素:** {', '.join(report_data.get('detected_elements', []))}")
                    with st.expander("Raw JSON"):
                        st.json(report_data)