import io
import os
import requests
import gradio as gr
from openai import AzureOpenAI
from dotenv import load_dotenv

# 환경 변수 로드 (.env 사용)
load_dotenv()

PREDICTION_URL = os.getenv("PREDICTION_URL")
PREDICTION_KEY = os.getenv("PREDICTION_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# Custom Vision으로 약 분류

def classify_pill(image):
    if image is None:
        return "이미지 없음", 0.0

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    headers = {
        "Content-Type": "application/octet-stream",
        "Prediction-Key": PREDICTION_KEY,
    }

    resp = requests.post(PREDICTION_URL, headers=headers, data=img_bytes)
    resp.raise_for_status()
    data = resp.json()

    preds = data.get("predictions", [])
    if not preds:
        return "분류 실패", 0.0

    best = max(preds, key=lambda x: x["probability"])
    tag_name = best["tagName"]
    prob = best["probability"] * 100
    return tag_name, prob

# Azure OpenAI로 약 설명 생성

def explain_pill_with_gpt(pill_name: str) -> str:
    if pill_name in ["이미지 없음", "분류 실패"]:
        return "이미지 인식이 제대로 되지 않아 약 정보를 생성할 수 없습니다. 다시 촬영해 주세요."

    system_msg = (
        "당신은 복약 안내를 도와주는 친절한 약사입니다. "
        "사용자가 복용하려는 약의 이름을 알려주면, "
        "1) 어떤 약인지, 2) 일반적인 효능, 3) 기본 복용 방법, "
        "4) 대표적인 주의사항/부작용을 쉽고 짧게 bullet 형식으로 설명해 주세요. "
        "의사가 아닌 AI 데모 서비스이므로, 마지막에 반드시 "
        "'정확한 복약 안내는 약사·의사와 상의해 주세요.'라는 문장을 포함해 주세요."
    )

    user_msg = f"약 이름: {pill_name}\n이 약에 대해 위 기준에 맞게 한국어로 설명해 주세요."

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()


# Gradio에서 쓸 분석 함수

def analyze_pill(image):
    if image is None:
        return "이미지가 업로드되지 않았습니다.", ""

    pill_name, prob = classify_pill(image)
    detail = explain_pill_with_gpt(pill_name)

    header_text = f"예측된 약 이름: {pill_name} (신뢰도: {prob:.1f}%)"
    return header_text, detail



# Gradio UI CSS 

custom_css = """

body, .gradio-container {
    background-color: #ffffff !important;
    font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", system-ui, sans-serif;
}

.gradio-container .gr-block,
.gradio-container .gr-panel,
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .styler,
.gradio-container .wrap,
.gradio-container .contain {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    min-height: 0 !important;
}

.pill-phone-card {
    max-width: 800px;
    margin: 20px auto;
    background: #ffffff;
    border-radius: 32px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.06);
    padding: 30px;
    border: 1px solid #f0f0f0;
}

.mint-point {
    color: #4FD1C5 !important; /* 민트색 */
}

.pill-landing-title, .pill-header-title {
    font-size: 32px;
    font-weight: 800;
    text-align: center;
    color: #38B2AC; /* 다크 민트 */
    margin-bottom: 8px;
}

.pill-landing-sub, .pill-header-sub {
    text-align: center;
    font-size: 14px;
    color: #718096;
    margin-bottom: 24px;
}

.pill-landing-highlight {
    color: #319795;
    font-weight: 700;
}

.pill-landing-box {
    margin-top: 10px;
    padding: 20px;
    border-radius: 20px;
    background: #F0FFF4; 
    border: 1px dashed #B2F5EA;
    font-size: 13px;
    color: #2D3748;
}


/* 랜딩 화면의 시작하기 버튼만 전체 폭 */
.pill-landing-start-btn {
    margin-top: 20px;
    width: 100%;
    background: linear-gradient(135deg, #4FD1C5, #38B2AC) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    border-radius: 16px !important;
    height: 50px;
    border: none !important;
}

/* 도구 화면의 메인 버튼 (폭은 flex로 맞춤) */
.pill-btn-main {
    background: linear-gradient(135deg, #4FD1C5, #38B2AC) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    border-radius: 16px !important;
    border: none !important;
}
.pill-guide-list span.num {
    display: inline-block;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #4FD1C5;
    color: #fff;
    text-align: center;
    font-size: 12px;
    line-height: 20px;
    margin-right: 8px;
}

.pill-image-wrapper .gradio-image {
    border-radius: 24px;
    overflow: hidden;
    border: 2px solid #E6FFFA;
}
.pill-result-box {
    margin-top: 16px;
    padding: 16px;
    border-radius: 20px;
    background: #ffffff;
    border: 1px solid #E6FFFA;
}

.pill-result-title {
    font-size: 14px;
    font-weight: 700;
    color: #2C7A7B;
    margin-bottom: 8px;
}

.pill-btn-sub {
    background: #E6FFFA !important;
    color: #2C7A7B !important;
    border-radius: 16px !important;
    border: none !important;
}

.pill-footer-note, .pill-landing-footer {
    margin-top: 24px;
    font-size: 12px;
    color: #A0AEC0;
    text-align: center;
}

.btn-main, .btn-secondary, .pill-btn-main, .pill-btn-sub {
    height: 54px !important; 
    min-height: 54px !important; 
    max-height: 54px !important;
    line-height: 54px !important; 
    padding: 0 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
"""


# 화면 전환용 함수

def go_tool():
    return gr.update(visible=False), gr.update(visible=True)


# Gradio Blocks

with gr.Blocks(css=custom_css, title="AI 복약 가이드") as demo:

    with gr.Column(elem_classes=["pill-phone-card"]) as main_card:

        # 랜딩 화면
        with gr.Column(visible=True) as landing_group:
            gr.Markdown("""
<div class="pill-landing-title">AI 복약 가이드</div>
<div class="pill-landing-sub">
사진 한 장으로 <span class="pill-landing-highlight">어떤 약인지, 어떻게 먹어야 하는지</span><br>
빠르게 확인할 수 있는 Azure 기반 데모 서비스입니다.
</div>

<div class="pill-landing-box">
  <div class="pill-landing-step">① 알약 앞·뒷면을 또렷하게 촬영해 주세요.</div>
  <div class="pill-landing-step">② 사진을 업로드하면 알약을 인식하고 이름을 예측합니다.</div>
  <div class="pill-landing-step">③ Azure OpenAI가 효능, 복용법, 주의사항을 쉽게 설명해 줍니다.</div>
</div>
""")

            start_btn = gr.Button("시작하기", elem_classes=["pill-landing-start-btn"])

            gr.Markdown("""
<div class="pill-landing-footer">
※ 본 서비스는 교육용 데모이며, 실제 복약 전에는 반드시 의료진·약사와 상담해 주세요.<br>
숙명여대 Azure Winter School Team 5
</div>
""")

        # 실제 도구 화면 
        with gr.Column(visible=False) as tool_group:

            gr.Markdown("""
<div class="pill-header-title">AI 복약 가이드</div>
<div class="pill-header-sub">
알약 사진을 업로드하면 어떤 약인지 분류하고,<br>
복용 방법과 주의사항을 안내해 드립니다.
</div>
""")

            gr.Markdown("""
<div class="pill-guide-title" style="font-weight:700; font-size:15px; margin-bottom:10px;">📸 알약 촬영 가이드</div>
<div class="pill-guide-list">
<div style="margin-bottom:5px;"><span class="num">1</span> 알약이 <b>화면 중앙</b>에 오도록 촬영</div>
<div style="margin-bottom:5px;"><span class="num">2</span> <b>밝은 조명</b> 아래에서 찍어 주세요</div>
<div style="margin-bottom:15px;"><span class="num">3</span> <b>깔끔한 배경</b>일수록 인식률이 높아집니다</div>
</div>
""")

            with gr.Column(elem_classes=["pill-image-wrapper"]):
                image_in = gr.Image(
                    type="pil",
                    label="",
                    height=280,
                    width=280,
                    show_label=False,
                )

            with gr.Row(elem_classes=["pill-btn-row"], equal_height=True):
                clear_btn = gr.Button("다시 선택", elem_classes=["pill-btn-sub"])
                submit_btn = gr.Button("결과 분석하기", elem_classes=["pill-btn-main"])

            with gr.Column(elem_classes=["pill-result-box"]):
                gr.Markdown('<div class="pill-result-title">🔍 인식된 약품 정보</div>')
                pill_header = gr.Textbox(
                    placeholder="이미지를 업로드한 뒤 [결과 분석하기] 버튼을 눌러 주세요.",
                    interactive=False,
                    lines=1,
                    show_label=False,
                )

            with gr.Column(elem_classes=["pill-result-box"]):
                gr.Markdown('<div class="pill-result-title">💊 상세 복약 가이드</div>')
                pill_detail = gr.Textbox(
                    placeholder="약의 효능, 복용 방법, 주의사항이 이곳에 표시됩니다.",
                    interactive=False,
                    lines=10,
                    show_label=False,
                )

            gr.Markdown("""
<div class="pill-footer-note">
※ 본 서비스는 교육용 데모이며, 실제 복약 전에는 반드시 의료진·약사와 상담해 주세요.<br>
 숙명여대 Azure Winter School Team 5
</div>
""")

    #버튼 동작 
    start_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=None,
        outputs=[landing_group, tool_group],
    )

    submit_btn.click(
        fn=analyze_pill,
        inputs=image_in,
        outputs=[pill_header, pill_detail],
    )

    clear_btn.click(
        fn=lambda: (None, "", ""),
        inputs=None,
        outputs=[image_in, pill_header, pill_detail],
    )

if __name__ == "__main__":
    demo.launch()
