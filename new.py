import io
import os
import requests
import difflib 
import gradio as gr
from openai import AzureOpenAI
from dotenv import load_dotenv

#  환경 변수 로드 (.env 사용)
load_dotenv()

# Custom Vision
PREDICTION_URL = os.getenv("PREDICTION_URL")
PREDICTION_KEY = os.getenv("PREDICTION_KEY")

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Azure Vision (OCR 용)
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")  # https://pill-vision-team5.cognitiveservices.azure.com/
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# 공통: 문자열 정규화 + 유사도 점수 함수
def _normalize(text: str) -> str:
    """알파벳/숫자만 남기고 대문자로 통일"""
    if not text:
        return ""
    return "".join(ch for ch in text.upper() if ch.isalnum())

def _similarity(a: str, b: str) -> float:
    """0~1 사이 유사도 (SequenceMatcher 사용)"""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# Azure Vision OCR로 알약 표면 글자 읽기

def ocr_pill_text(image) -> str:
    """
    알약 표면의 알파벳/숫자를 OCR로 읽어서 한 줄 문자열로 반환.
    실패하면 "" 반환.
    """
    if image is None:
        return ""

    if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
        # Vision 리소스 설정 안 돼 있으면 OCR 패스
        return ""

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    url = (
        AZURE_VISION_ENDPOINT.rstrip("/")
        + "/computervision/imageanalysis:analyze"
        + "?api-version=2023-10-01&features=read"
    )

    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": AZURE_VISION_KEY,
    }

    try:
        resp = requests.post(url, headers=headers, data=img_bytes, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("OCR 호출 에러:", e)
        return ""

    texts = []
    try:
        read_result = data.get("readResult") or {}
        blocks = read_result.get("blocks") or []
        for b in blocks:
            for line in b.get("lines", []):
                txt = line.get("text", "").strip()
                if txt:
                    texts.append(txt)
    except Exception as e:
        print("OCR 파싱 에러:", e)
        return ""

    joined = " ".join(texts)
    return joined[:120]  # 너무 길면 잘라줌
def pick_best_with_gpt(preds, ocr_text: str) -> str:
    if not preds or not ocr_text:
        return preds[0]["tagName"]

    candidates_txt = "\n".join(
        f"- {p['tagName']} (확률: {p['probability']*100:.1f}%)"
        for p in preds
    )

    system_msg = (
        "당신은 약품 라벨을 매칭해 주는 도우미입니다. "
        "OCR로 읽은 영문 글자와 Custom Vision이 예측한 후보 약 이름(대부분 한글)을 보고, "
        "가장 가능성이 높은 한글 약 이름 하나만 골라 주세요. "
        "반드시 후보 목록에 있는 이름만 그대로 출력하세요."
    )

    user_msg = (
        f"OCR 텍스트: {ocr_text}\n\n"
        f"후보 리스트:\n{candidates_txt}\n\n"
        "가장 가능성 높은 약 이름 하나만 출력하세요."
    )

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    chosen = resp.choices[0].message.content.strip()

    tag_names = {p["tagName"] for p in preds}
    if chosen not in tag_names:
        return preds[0]["tagName"]

    return chosen
# Custom Vision + OCR 같이 써서 최종 약 이름 선택

def classify_pill(image):
    """
    1) Custom Vision 전체 predictions 가져옴
    2) OCR로 알약 표면 글자 읽기
    3) OCR 성공 시: 알파벳 유사도가 가장 큰 tagName 선택
       - 유사도 너무 낮으면 그냥 원래 top1(tagName) 사용
    4) OCR 실패 시: 원래 top1만 사용
    """
    if image is None:
        return "이미지 없음", 0.0, ""

    # Custom Vision 호출
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
        return "분류 실패", 0.0, ""

    # 확률 기준 기본 1등 (fallback용)
    base = max(preds, key=lambda x: x["probability"])
    base_tag = base["tagName"]
    base_prob = base["probability"] * 100

    # OCR 
    ocr_text = ocr_pill_text(image)  # 이미 만들어둔 OCR 함수
    if not ocr_text:
        # OCR 실패 → Custom Vision 결과 그대로 사용
        return base_tag, base_prob, ""

    ocr_norm = _normalize(ocr_text)  # 알파벳/숫자만 남기고 대문자로

    # 알파벳 유사도 먼저 보는 로직 
    best_tag = base_tag
    best_prob = base_prob
    best_sim = -1.0

    for p in preds:
        tag = p["tagName"]
        prob = p["probability"] * 100
        tag_norm = _normalize(tag)

        sim = _similarity(ocr_norm, tag_norm)  # 0~1

        if sim > best_sim or (sim == best_sim and prob > best_prob):
            best_sim = sim
            best_tag = tag
            best_prob = prob

    SIM_THRESHOLD = 0.25  # 필요하면 조절
    if best_sim < SIM_THRESHOLD:
        return base_tag, base_prob, ocr_text

    return best_tag, best_prob, ocr_text


# Azure OpenAI로 약 설명 생성

def explain_pill_with_gpt(pill_name: str, ocr_text: str = "", prob: float = 0.0) -> str:
    if pill_name in ["이미지 없음", "분류 실패"]:
        return "이미지 인식이 제대로 되지 않아 약 정보를 생성할 수 없습니다. 다시 촬영해 주세요."

    system_msg = (
        "당신은 복약 안내를 도와주는 친절한 약사입니다. "
        "모델이 예측한 약 이름과 알약 표면의 글자(알파벳/숫자)를 참고해서 "
        "해당 약에 대한 정보를 한국어로 설명해 주세요. "
        "1) 어떤 약인지, 2) 일반적인 효능, 3) 기본 복용 방법, "
        "4) 대표적인 주의사항/부작용을 bullet 형식으로 정리해 주세요. "
        "AI 데모 서비스이므로 실제 제품명이나 성분이 100% 정확하지 않을 수 있습니다. "
        "답변 마지막에는 반드시 '정확한 복약 안내는 약사·의사와 상의해 주세요.' 문장을 포함하세요."
    )

    user_msg = (
        f"모델이 예측한 약 이름: {pill_name}\n"
        f"모델 신뢰도: {prob:.1f}%\n"
        f"OCR로 읽힌 알약 표면 글자: '{ocr_text}'\n\n"
        "위 정보를 바탕으로, 가장 가능성이 높은 약품을 기준으로 설명해 주세요. "
        "약 이름이 애매하거나 여러 후보가 있을 수 있으면, "
        "첫 번째 bullet에서 포장지/설명서를 반드시 확인하라고 언급해 주세요."
    )

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# Gradio에서 호출할 최종 함수

def analyze_pill(image):
    if image is None:
        return "이미지가 업로드되지 않았습니다.", ""

    pill_name, prob, ocr_text = classify_pill(image)
    detail = explain_pill_with_gpt(pill_name, ocr_text, prob)

    if ocr_text:
        header_text = (
            f"예측된 약 이름: {pill_name} (신뢰도: {prob:.1f}%) | "
            f"알약 표면 글자: {ocr_text}"
        )
    else:
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