import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import cv2
import io

# ==========================================
# 1. 核心修复补丁 (Monkey Patch)
# ==========================================
# 解决 Streamlit 1.32+ 中 'image_to_url' 属性缺失导致的 Crash
# 必须在导入 streamlit_drawable_canvas 之前执行
try:
    from streamlit.elements import image as st_image
    if not hasattr(st_image, 'image_to_url'):
        from streamlit.elements.lib import image_utils
        st_image.image_to_url = image_utils.image_to_url
except ImportError:
    pass  # 针对旧版本的防御性编程

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

# 初始化 Session State
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'spectral_data' not in st.session_state:
    st.session_state.spectral_data = None
if 'roi_stats' not in st.session_state:
    st.session_state.roi_stats = {}

# ==========================================
# 3. 安全鉴权模块 (基于 st.secrets)
# ==========================================
def check_authentication():
    """
    鉴权守门人：验证用户密码是否与 secrets.toml 中一致。
    """
    def password_entered():
        # 从 st.secrets 读取预设密码
        if st.session_state["password_input"] == st.secrets["general"]["app_password"]:
            st.session_state.authenticated = True
            del st.session_state["password_input"]  # 立即清除内存中的明文密码
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
    st.stop()  # 阻断后续代码执行

# ==========================================
# 4. 科学仿真算法库
# ==========================================
def generate_pseudo_spectrum_curve(n_points=1000, n_peaks=5, noise_level=0.05):
    """
    生成高保真伪光谱曲线：基线 + 高斯峰 + 噪声
    """
    wavelengths = np.linspace(400, 1000, n_points)
    intensity = np.zeros_like(wavelengths)
    
    # 1. 添加线性与非线性基线漂移
    intensity += np.linspace(0.1, 0.3, n_points)  # 线性
    intensity += 0.05 * np.sin(wavelengths / 100) # 低频波动
    
    # 2. 随机生成高斯特征峰
    np.random.seed(int(time.time())) # 确保每次点击生成不同光谱
    peaks_info =
    for _ in range(n_peaks):
        center = np.random.uniform(450, 950)
        width = np.random.uniform(5, 15)   # FWHM 相关
        amplitude = np.random.uniform(0.3, 1.0)
        
        # 高斯函数
        peak_signal = amplitude * np.exp(-((wavelengths - center)**2) / (2 * width**2))
        intensity += peak_signal
        
        peaks_info.append(f"{center:.1f}nm")
    
    # 3. 添加加性高斯白噪声 (AWGN)
    noise = np.random.normal(0, noise_level, n_points)
    intensity += noise
    
    # 4. 归一化处理 (0-1)
    intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    
    return pd.DataFrame({"Wavelength": wavelengths, "Intensity": intensity}), peaks_info

def create_synthetic_diffraction_image(width=800, height=600):
    """
    生成模拟的衍射/干涉光谱图像，用于 ROI 演示。
    """
    # 创建黑色背景
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 模拟光谱条纹 (彩虹色渐变)
    for i in range(width):
        hue = int((i / width) * 180) # OpenCV Hue 范围 0-179
        # 在图像中间画一条亮带
        cv2.line(img, (i, 100), (i, 500), (hue, 255, 255), 1)
        
    # 转换回 BGR 并添加高斯模糊模拟光晕
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.GaussianBlur(img, (51, 51), 0)
    
    # 添加几条明显的发射线 (模拟特征峰)
    lines = [(200, (0, 255, 255)), (450, (0, 255, 0)), (600, (0, 0, 255))]
    for x, color in lines:
        cv2.line(img, (x, 50), (x, 550), color, 4)
        # 添加光晕
        overlay = img.copy()
        cv2.line(overlay, (x, 50), (x, 550), color, 20)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # 添加模拟 Sensor 噪声
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ==========================================
# 5. LLM 智能接口 (结构化输出)
# ==========================================
def get_ai_physical_report(stats_json):
    """
    调用 LLM 生成结构化物理分析报告。
    使用 JSON Schema 强制模型输出稳定格式。
    """
    client = OpenAI(
        api_key=st.secrets["llm"]["api_key"],
        base_url=st.secrets["llm"]["base_url"]
    )
    
    # 定义严格的 JSON Schema
    report_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "spectroscopy_analysis_report",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "sample_quality": {
                        "type": "string",
                        "enum": ["Excellent", "Good", "Noisy", "Critical"],
                        "description": "基于信噪比对样本质量的评估"
                    },
                    "detected_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "推测的化学元素列表，例如 ['Hydrogen', 'Helium']"
                    },
                    "physical_interpretation": {
                        "type": "string",
                        "description": "对光谱特征的物理学解释，不超过50字"
                    },
                    "anomaly_detected": {
                        "type": "boolean",
                        "description": "是否存在异常信号"
                    },
                    "confidence_score": {
                        "type": "number",
                        "description": "AI对分析结果的置信度 (0.0 - 1.0)"
                    }
                },
                "required": ["sample_quality", "detected_elements", "physical_interpretation", "anomaly_detected", "confidence_score"],
                "additionalProperties": False
            }
        }
    }

    prompt = f"""
    你是一名资深光谱分析专家。请根据以下从 ROI 区域提取的统计数据，生成一份物理分析报告。
    
    输入数据:
    {json.dumps(stats_json)}
    
    背景知识：
    - 高均值强度通常意味着强发射信号。
    - 高标准差可能意味着信号波动大或存在多个峰。
    - 这里的像素区域代表光谱仪的感光面积。
    """

    try:
        completion = client.chat.completions.create(
            model=st.secrets["llm"]["model"],
            messages=,
            response_format=report_schema
        )
        return json.loads(completion.choices.message.content)
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 6. UI 布局与交互逻辑
# ==========================================

# 6.1 Sidebar: 全局控制
with st.sidebar:
    st.title("SpectraMind 🔬")
    st.caption(f"v1.0.2 | 已认证: {st.secrets['general']['app_password'][:2]}***")
    st.markdown("---")
    
    st.subheader("⚙️ 硬件参数模拟")
    integration_time = st.slider("积分时间 (ms)", 10, 2000, 100, help="模拟 CCD 曝光时间")
    laser_power = st.number_input("激光功率 (mW)", 0, 500, 50)
    gain_level = st.select_slider("增益等级", options=["Low", "Medium", "High", "Ultra"])
    
    st.markdown("---")
    st.subheader("🎨 绘图配置")
    plot_theme = st.selectbox("图表主题", ["plotly_dark", "plotly_white", "ggplot2"])
    show_peaks = st.checkbox("标记特征峰", True)

# 6.2 主界面: Tabs 结构
tab_monitor, tab_vision, tab_report = st.tabs()

# --- TAB 1: 光谱监测 ---
with tab_monitor:
    st.markdown("### 实时光谱采集流")
    
    # 顶部 Metrics 指标卡
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("峰值波长", "656.3 nm", "H-α")
    col2.metric("信噪比 (SNR)", "45.2 dB", "-1.2 dB", delta_color="inverse")
    col3.metric("系统温度", "23.1 °C", "0.2 °C")
    col4.metric("采集帧率", "12 fps", "Stable")
    
    # 生成/更新数据
    if st.button("采集单帧光谱", type="primary") or st.session_state.spectral_data is None:
        with st.spinner("正在从虚拟光谱仪读取数据..."):
            time.sleep(0.3) # 模拟 IO 延迟
            df_spec, peaks = generate_pseudo_spectrum_curve(noise_level=0.1 if gain_level=="High" else 0.02)
            st.session_state.spectral_data = df_spec
            st.toast(f"采集完成！检测到 {len(peaks)} 个特征峰", icon="✅")
    
    # 绘制 Plotly 曲线
    if st.session_state.spectral_data is not None:
        df = st.session_state.spectral_data
        fig = go.Figure()
        
        # 信号线
        fig.add_trace(go.Scatter(
            x=df, y=df['Intensity'],
            mode='lines', name='光谱信号',
            line=dict(color='#00ffcc' if plot_theme=='plotly_dark' else '#0066cc', width=2),
            fill='tozeroy'
        ))
        
        # 布局优化
        fig.update_layout(
            template=plot_theme,
            xaxis_title="波长 Wavelength (nm)",
            yaxis_title="归一化强度 Normalized Intensity (a.u.)",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: ROI 视觉分析 (核心难点) ---
with tab_vision:
    st.markdown("### 高光谱图像 ROI 提取")
    st.info("💡 提示：在左侧图像上绘制矩形，右侧将自动显示该区域的原始分辨率切片及统计数据。无需手动计算坐标缩放。")
    
    col_canvas, col_result = st.columns([1.5, 1])
    
    # 生成高分辨率原图 (例如 800x600)
    original_image = create_synthetic_diffraction_image(800, 600)
    
    with col_canvas:
        # 定义 Canvas 显示宽度 (例如固定 500px)
        CANVAS_WIDTH = 500
        # 计算显示的比例高度，保持纵横比
        orig_w, orig_h = original_image.size
        CANVAS_HEIGHT = int(CANVAS_WIDTH * orig_h / orig_w)
        
        st.markdown(f"**画布视图** (显示尺寸: {CANVAS_WIDTH}x{CANVAS_HEIGHT})")
        
        # 调用 Drawable Canvas
        # 注意：background_image 会被自动缩放到 Canvas 大小
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
        st.markdown(f"**分析结果** (原始分辨率: {orig_w}x{orig_h})")
        
        # 解析 ROI 数据
        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            # 获取最后一个绘制的对象
            obj = canvas_result.json_data["objects"][-1]
            
            # --- 关键：无痛坐标映射逻辑 ---
            # Canvas 返回的坐标是基于 CANVAS_WIDTH 的
            # 我们需要将其映射回 original_image 的尺寸
            scale_x = orig_w / CANVAS_WIDTH
            scale_y = orig_h / CANVAS_HEIGHT
            
            # 提取并映射坐标
            # 注意：用户可能在 Canvas 上对框进行了缩放，所以要乘 obj['scaleX']
            rect_x = int(obj["left"] * scale_x)
            rect_y = int(obj["top"] * scale_y)
            rect_w = int(obj["width"] * obj["scaleX"] * scale_x)
            rect_h = int(obj["height"] * obj * scale_y)
            
            # 边界保护
            img_arr = np.array(original_image)
            y1, y2 = max(0, rect_y), min(orig_h, rect_y + rect_h)
            x1, x2 = max(0, rect_x), min(orig_w, rect_x + rect_w)
            
            if x2 > x1 and y2 > y1:
                # Numpy 切片提取 ROI
                roi_slice = img_arr[y1:y2, x1:x2]
                
                # 显示提取的 ROI
                st.image(roi_slice, caption=f"ROI 切片 ({x2-x1}x{y2-y1} px)")
                
                # 计算统计量
                mean_val = float(np.mean(roi_slice))
                std_val = float(np.std(roi_slice))
                max_val = float(np.max(roi_slice))
                
                # 展示 Metrics
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("ROI 均值", f"{mean_val:.1f}")
                m_col2.metric("ROI 标准差", f"{std_val:.1f}")
                
                # 保存到 Session State 供 Report 使用
                st.session_state.roi_stats = {
                    "roi_position": [x1, y1, x2, y2],
                    "mean_intensity": mean_val,
                    "std_deviation": std_val,
                    "max_intensity": max_val,
                    "pixel_count": int((x2-x1)*(y2-y1))
                }
            else:
                st.warning("ROI 区域无效或超出边界")
        else:
            st.info("请在左侧图像上绘制矩形框以选取分析区域。")

# --- TAB 3: 智能报告 ---
with tab_report:
    st.markdown("### 📝 物理分析报告生成器")
    
    if not st.session_state.roi_stats:
        st.warning("⚠️ 请先在 'ROI 视觉分析' 标签页中选择一个区域。")
    else:
        st.markdown("#### 待分析数据包")
        st.json(st.session_state.roi_stats, expanded=False)
        
        btn_generate = st.button("🚀 调用 AI 生成报告", type="primary")
        
        if btn_generate:
            with st.status("正在进行物理建模与推理...", expanded=True) as status:
                st.write("正在连接 LLM 推理端点...")
                time.sleep(0.5)
                st.write("上传 ROI 统计特征...")
                
                # 调用 AI
                report_data = get_ai_physical_report(st.session_state.roi_stats)
                
                if "error" in report_data:
                    status.update(label="生成失败", state="error")
                    st.error(f"API 调用错误: {report_data['error']}")
                else:
                    status.update(label="报告生成完毕", state="complete")
                    
                    st.divider()
                    st.subheader("SpectraMind Analysis Report")
                    
                    # 渲染结构化报告
                    r_col1, r_col2 = st.columns()
                    
                    with r_col1:
                        # 状态卡片
                        if report_data["anomaly_detected"]:
                            st.error("❌ 检测到异常")
                        else:
                            st.success("✅ 样本正常")
                        
                        st.metric("置信度", f"{report_data['confidence_score']*100:.1f}%")
                        st.text_input("样本质量", report_data["sample_quality"], disabled=True)
                    
                    with r_col2:
                        st.markdown("**物理学解释:**")
                        st.info(report_data["physical_interpretation"])
                        
                        st.markdown("**检测到的元素痕迹:**")
                        st.write(", ".join([f"`{elm}`" for elm in report_data["detected_elements"]]))
                    
                    # 原始 JSON 折叠
                    with st.expander("查看原始 JSON 输出"):
                        st.json(report_data)