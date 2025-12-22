import io
import os
import requests
import gradio as gr
from openai import AzureOpenAI
from dotenv import load_dotenv  # ✅ 추가

# .env 불러오기
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


# Gradio UI 
custom_css = """
body {
    background: radial-gradient(circle at top, #e7f6f5 0%, #f9fdfc 40%, #ffffff 100%);
    font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
}

/* 공통 래퍼 */
.pill-app {
    max-width: 960px;
    margin: 40px auto 32px auto;
}

/* 카드 공통 */
.pill-card {
    background: #ffffff;
    border-radius: 22px;
    padding: 28px 30px;
    box-shadow: 0 18px 45px rgba(15, 118, 110, 0.08);
}

/* 랜딩 타이틀 */
.pill-hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #0b9b8c;
    margin-bottom: 10px;
    letter-spacing: -0.03em;
    text-align: center;
}

.pill-hero-sub {
    font-size: 1.02rem;
    color: #4b5563;
    text-align: center;
    margin-bottom: 18px;
}

.pill-hero-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: #e0f4f1;
    color: #04756f;
    font-size: 0.8rem;
    margin-bottom: 16px;
}

.pill-hero-highlight {
    color: #ff7e41;
    font-weight: 700;
}

/* 랜딩 버튼 */
.pill-start-btn {
    background: #ff914d !important;
    border-color: #ff914d !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 999px !important;
    padding: 10px 28px !important;
    font-size: 1rem !important;
}

.pill-start-btn:hover {
    filter: brightness(1.05);
}

/* 작은 설명 텍스트 */
.pill-hero-foot {
    font-size: 0.85rem;
    color: #6b7280;
    text-align: center;
    margin-top: 10px;
}

/* 메인 도구 라벨 */
.pill-label {
    font-weight: 600;
    font-size: 0.92rem;
    color: #0f766e;
}

/* 이미지 업로드 박스 */
.pill-image .wrap {
    border-radius: 16px !important;
    border: 1.5px dashed #86d1c7 !important;
    background: #f4fbfa !important;
}

/* 결과 텍스트 박스 */
.pill-output textarea {
    background: #f9fafb !important;
    border-radius: 14px !important;
}

/* 버튼들 */
.pill-btn-main {
    background: #ff914d !important;
    border-color: #ff914d !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 999px !important;
}

.pill-btn-secondary {
    background: #e5f5f4 !important;
    border-color: #bbe4df !important;
    color: #0f766e !important;
    font-weight: 500 !important;
    border-radius: 999px !important;
}

/* 푸터 */
.pill-footer {
    font-size: 0.8rem;
    color: #6b7280;
    text-align: center;
    margin-top: 8px;
}
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="orange",
    neutral_hue="slate",
).set(
    button_large_radius="999px",
    block_radius="20px",
)

with gr.Blocks(css=custom_css, theme=theme, title="알약 인식 & 설명 챗봇") as demo:
    with gr.Column(elem_classes="pill-app"):
        # ---------- 1. 랜딩 화면 ----------
        with gr.Column(elem_classes="pill-card", elem_id="landing") as landing_col:
            gr.Markdown(
                """
                <div style="text-align:center;">
                  <div class="pill-hero-badge">MS Azure + Custom Vision + Azure OpenAI</div>
                  <div class="pill-hero-title">알약 인식 & 설명 챗봇</div>
                  <div class="pill-hero-sub">
                    알약 사진 한 장으로 <span class="pill-hero-highlight">약 이름 · 효능 · 복용 주의사항</span>까지 한 번에 확인해보세요.
                  </div>
                </div>
                """,
                elem_id="landing_text",
            )

            with gr.Row():
                gr.Markdown(
                    """
                    - 알약 사진을 업로드하면 **Custom Vision**이 어떤 약인지 분류하고  
                    - 분류 결과를 바탕으로 **Azure OpenAI**가 복용 방법과 주의사항을 자연어로 설명합니다.  
                    """,
                    elem_id="landing_desc",
                )

            start_btn = gr.Button("시작하기", elem_classes="pill-start-btn")
            gr.Markdown(
                """
                <div class="pill-hero-foot">
                  ※ 본 서비스는 교육용 데모입니다. 실제 복용 결정은 반드시 의료 전문가와 상의해 주세요.
                </div>
                """,
            )

        # ---------- 2. 실제 도구 화면 (처음엔 숨김, 세로 배치) ----------
        with gr.Column(elem_classes="pill-card", visible=False) as tool_col:
            gr.Markdown(
                "#### 알약 사진 업로드 후, 분석하기 버튼을 눌러 주세요.",
            )

            # 세로 배치
            image_in = gr.Image(
                type="pil",
                label="알약 사진 업로드",
                elem_classes="pill-image pill-label",
            )

            with gr.Row():
                clear_btn = gr.Button("초기화", elem_classes="pill-btn-secondary")
                submit_btn = gr.Button("분석하기", elem_classes="pill-btn-main")

            result_box = gr.Textbox(
                label="분류 결과",
                lines=2,
                interactive=False,
                elem_classes="pill-output pill-label",
            )

            explain_box = gr.Textbox(
                label="약 설명 (Azure OpenAI)",
                lines=14,
                interactive=False,
                elem_classes="pill-output pill-label",
            )

            gr.Markdown(
                """
                <div class="pill-footer">
                  MS Azure 기반 · 숙명여대 Azure Winter School 5팀
                </div>
                """
            )

    # --------- 동작 연결 ---------

    # 시작하기 → 랜딩 숨기고 도구 보이기
    def show_tool():
        return gr.update(visible=False), gr.update(visible=True)

    start_btn.click(
        fn=show_tool,
        inputs=None,
        outputs=[landing_col, tool_col],
    )

    # 분석하기 버튼 → 기존 파이프라인 실행
    submit_btn.click(
        fn=pill_pipeline,
        inputs=image_in,
        outputs=[result_box, explain_box],
    )

    # 초기화 버튼
    def reset_all():
        return None, "", ""

    clear_btn.click(
        fn=reset_all,
        inputs=None,
        outputs=[image_in, result_box, explain_box],
    )

if __name__ == "__main__":
    demo.launch()