import io
import os
import requests
import gradio as gr
from openai import AzureOpenAI
from dotenv import load_dotenv  

load_dotenv()

# Custom Vision 설정 (환경변수에서 가져오기)
PREDICTION_URL = os.getenv("PREDICTION_URL")
PREDICTION_KEY = os.getenv("PREDICTION_KEY")

# Azure OpenAI 설정 (환경변수에서 가져오기)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# 약 이미지 → Custom Vision 예측 
def classify_pill(image):
    """
    Gradio에서 받은 PIL 이미지(image)를 Custom Vision으로 보내서
    가장 확률 높은 약 이름(tag_name)과 확률을 반환
    """
    # PIL 이미지를 bytes로 변환
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    headers = {
        "Content-Type": "application/octet-stream",
        "Prediction-Key": PREDICTION_KEY,
    }

    res = requests.post(PREDICTION_URL, headers=headers, data=img_bytes)
    res.raise_for_status()
    data = res.json()

    predictions = data.get("predictions", [])
    if not predictions:
        return "알 수 없음", 0.0

    top = max(predictions, key=lambda x: x["probability"])
    pill_name = top["tagName"]
    prob = float(top["probability"])
    return pill_name, prob


# 약 이름 → Azure OpenAI 설명 
def explain_pill(pill_name: str, prob: float) -> str:
    """
    약 이름과 확률을 받아서, Azure OpenAI에게 설명을 생성하게 함
    """
    if pill_name == "알 수 없음":
        return "이미지에서 약을 잘 인식하지 못했어요. 사진을 조금 더 선명하게 다시 찍어 주세요."

    user_prompt = f"""
    사용자가 올린 사진을 통해 Custom Vision 모델이 약을 '{pill_name}' 이라고 {prob*100:.1f}% 확신으로 예측했습니다.

    1. 이 약이 어떤 약인지 (성분, 일반적인 효능) 쉽게 설명해 주세요.
    2. 보통 어떤 상황에서 복용하는지.
    3. 10대~20대가 많이 쓰는 표현 정도 난이도로, 너무 무섭지 않게 주의사항(과다복용, 같이 먹으면 안 되는 경우 등)도 정리해 주세요.
    4. 의사가 아니기 때문에 최종 복용 전에는 반드시 의사/약사와 상담하라고 마지막에 한 줄 정도 덧붙여 주세요.
    """

    completion = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 친절한 약 설명 챗봇이야. "
                    "약에 대한 정확한 정보와 함께, 너무 어렵지 않은 말로 설명해 줘."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=600,
    )

    return completion.choices[0].message.content.strip()


# Gradio에서 쓸 파이프라인 함수 
def pill_pipeline(image):
    pill_name, prob = classify_pill(image)
    result_text = f"예측된 약 이름: {pill_name}  (신뢰도: {prob*100:.1f}%)"
    explanation = explain_pill(pill_name, prob)
    return result_text, explanation


# Gradio UI 세련된 메디컬 테마 적용
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;600;700;800&display=swap');

/* 전체 배경: 깨끗한 라이트 그레이와 화이트 */
body {
    background-color: #F5F7FA;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
}

.gradio-container {
    max-width: 850px !important;
    margin: 40px auto !important;
}

/* 카드 디자인: 애플 스타일의 부드러운 그림자와 둥근 모서리 */
.pill-card {
    background: #ffffff !important;
    border-radius: 24px !important;
    border: 1px solid #E5E9F0 !important;
    padding: 40px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03) !important;
}

/* 타이틀: 신뢰감 있는 딥 네이비 블루 */
.pill-hero-title {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    color: #1A202C !important;
    letter-spacing: -0.04em !important;
    margin-bottom: 12px !important;
    text-align: center;
}

.pill-hero-sub {
    font-size: 1.1rem !important;
    color: #4A5568 !important;
    text-align: center;
    line-height: 1.6;
    margin-bottom: 30px !important;
}

.pill-hero-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 8px;
    background: #EDF2F7;
    color: #4A5568;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 20px;
}

/* 하이라이트 컬러: 차분한 메디컬 블루 */
.pill-hero-highlight {
    color: #3182CE;
    font-weight: 700;
}

/* 메인 버튼: 신뢰감 있는 블루 그라데이션 */
.pill-start-btn, .pill-btn-main {
    background: linear-gradient(135deg, #3182CE 0%, #2B6CB0 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    height: 54px !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(49, 130, 206, 0.2) !important;
}

.pill-start-btn:hover, .pill-btn-main:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(49, 130, 206, 0.3) !important;
}

/* 보조 버튼: 부드러운 그레이 */
.pill-btn-secondary {
    background: #EDF2F7 !important;
    border: none !important;
    color: #4A5568 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    height: 54px !important;
}

/* 입력/출력창 스타일 */
.pill-output textarea, .pill-image {
    border-radius: 16px !important;
    border: 1px solid #E2E8F0 !important;
    background: #F8FAFC !important;
    padding: 15px !important;
}

.pill-label {
    font-weight: 700 !important;
    color: #2D3748 !important;
    margin-bottom: 8px !important;
}

.pill-footer {
    text-align: center;
    color: #A0AEC0;
    font-size: 0.9rem;
    margin-top: 30px;
}
"""

# Gradio 테마 설정 (Clean & Professional)
theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray",
).set(
    block_title_text_weight="700",
    block_label_text_size="sm",
    button_primary_background_fill="*primary_600",
)

with gr.Blocks(css=custom_css, theme=theme, title="On-nuri AI 복약 가이드") as demo:
    with gr.Column(elem_classes="pill-app"):
        # ---------- 1. 랜딩 화면 ----------
        with gr.Column(elem_classes="pill-card", elem_id="landing") as landing_col:
            gr.HTML(
                """
                <div style="text-align:center;">
                  <div class="pill-hero-badge">Smart Health Care · AI Predictor</div>
                  <h1 class="pill-hero-title">On-nuri 스마트 복약안내</h1>
                  <p class="pill-hero-sub">
                    복잡한 약 정보, 사진 한 장으로 해결하세요.<br/>
                    AI가 분석한 <span class="pill-hero-highlight">약 성분 · 효능 · 주의사항</span> 가이드를 제공합니다.
                  </p>
                </div>
                """
            )

            with gr.Row():
                gr.Markdown(
                    """
                    ### 🔍 서비스 활용 안내
                    1. **인식:** 알약의 앞/뒷면이 잘 보이도록 촬영해 주세요.
                    2. **분석:** Custom Vision AI가 수천 개의 약 데이터를 대조합니다.
                    3. **가이드:** Azure OpenAI가 이해하기 쉬운 복약 지도를 생성합니다.
                    """,
                )

            start_btn = gr.Button("분석 시작하기", elem_classes="pill-start-btn")
            
            gr.HTML('<p class="pill-hero-foot" style="text-align:center; color:#A0AEC0; font-size:0.8rem; margin-top:20px;">'
                    '※ 본 서비스는 교육용 데모이며, 정확한 복용법은 의사·약사와 상담하십시오.</p>')

        # ---------- 2. 도구 화면 ----------
        with gr.Column(elem_classes="pill-card", visible=False) as tool_col:
            gr.Markdown("### 💊 약 사진을 업로드해 주세요")

            image_in = gr.Image(
                type="pil",
                label="알약 이미지 (앞/뒷면)",
                elem_classes="pill-image",
            )

            with gr.Row():
                clear_btn = gr.Button("초기화", elem_classes="pill-btn-secondary")
                submit_btn = gr.Button("결과 분석하기", elem_classes="pill-btn-main")

            with gr.Column():
                result_box = gr.Textbox(
                    label="인식된 약품 정보",
                    placeholder="분석 결과가 여기에 표시됩니다.",
                    interactive=False,
                )
                explain_box = gr.Textbox(
                    label="상세 복약 가이드",
                    placeholder="AI가 생성한 설명이 여기에 표시됩니다.",
                    lines=12,
                    interactive=False,
                )

            gr.HTML('<div class="pill-footer">© 숙명여대 Azure Winter School 5팀 · Powered by Azure Cognitive Services</div>')

    # --------- 동작 연결 ---------
    def show_tool():
        return gr.update(visible=False), gr.update(visible=True)

    start_btn.click(fn=show_tool, outputs=[landing_col, tool_col])
    submit_btn.click(fn=pill_pipeline, inputs=image_in, outputs=[result_box, explain_box])
    clear_btn.click(fn=lambda: (None, "", ""), outputs=[image_in, result_box, explain_box])

if __name__ == "__main__":
    demo.launch()
