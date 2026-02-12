# SpectraMind Demo (Streamlit)

一个“看起来像科研仪器软件”的端云协同光电分析演示站点：
- 左侧：虚拟硬件控制台（Integration Time / Laser Power / Gain / Wavelength / Mode）
- 中间：仪器输出图像（亮度/对比度/噪声/伪彩NIR）+ ROI 框选 + 指标卡
- 右侧：伪光谱示波器曲线 + 一键调用方舟/豆包 OpenAI 兼容接口生成“物理分析报告”

## 1) 安装
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 配置 API Key
建议环境变量：
```bash
export ARK_API_KEY="你的key"
# 可选：
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export ARK_MODEL="ep-20260212130909-5kf5x"
```

## 3) 运行
```bash
streamlit run app.py
```

## 4) 说明
- 模型/endpoint 默认使用你提供的：`ep-20260212130909-5kf5x`
- base_url 默认：`https://ark.cn-beijing.volces.com/api/v3`
- 报告输出加了“Limitations”与“只基于JSON解释”的护栏，避免胡编导致穿帮
