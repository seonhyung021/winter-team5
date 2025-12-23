import io
import os
import requests
import gradio as gr
from openai import AzureOpenAI
from dotenv import load_dotenv

# 환경 변수 로드 (.env 사용 시)
load_dotenv()

# --- 설정 정보 (본인의 정보로 확인) ---
# 주의: Object Detection이므로 URL 중간에 /detect/가 있어야 하며, 끝은 /image여야 합니다.
PREDICTION_URL = "https://pillclassfication-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/aafa7eeb-a9f7-43ef-8d15-c6af7792f641/detect/iterations/Iteration1/image"
PREDICTION_KEY = "DqxBxChOwYy0zlye2PVJroXvj9ZtM40TCya1LgN1ZOigbVsXTiKhJQQJ99BLACL93NaXJ3w3AAAIACOG5H9l"

AZURE_OPENAI_ENDPOINT = "https://pill-vision-team5.cognitiveservices.azure.com/"
AZURE_OPENAI_API_KEY = "1zMeGpeavZ7XDghNmt5m9RS6jo1yDOnt8aSfWiFwU2aMmr9Er9d7JQQJ99BLACL93NaXJ3w3AAAEACOGUd7Q"
DEPLOYMENT_NAME = "pill-vision-team5"

# Azure OpenAI 클라이언트 생성
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# --- 1. Custom Vision Object Detection 함수 ---
def classify_pill(image):
    if image is None:
        return "이미지 없음", 0.0

    # RGBA -> RGB 변환 (에러 방지용)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    headers = {
        "Content-Type": "application/octet-stream",
        "Prediction-Key": PREDICTION_KEY,
    }

    try:
        # API 호출
        resp = requests.post(PREDICTION_URL, headers=headers, data=img_bytes)
        
        if resp.status_code != 200:
            print(f"API 에러 발생: {resp.text}")
            return f"오류(Code:{resp.status_code})", 0.0
            
        data = resp.json()
        preds = data.get("predictions", [])
        
        if not preds:
            return "알약을 찾을 수 없음", 0.0

        # Object Detection 결과 중 확률(probability)이 가장 높은 것 선택
        best = max(preds, key=lambda x: x["probability"])
        tag_name = best.get("tagName", "알 수 없는 약")
        prob = best["probability"] * 100
        
        # 신뢰도가 너무 낮으면 인식 실패로 처리 (임계값 30%)
        if prob < 30:
            return "인식 결과 불분명", prob

        return tag_name, prob

    except Exception as e:
        print(f"네트워크 오류: {e}")
        return "연결 실패", 0.0

# --- 2. Azure OpenAI 설명 생성 함수 ---
def explain_pill_with_gpt(pill_name: str) -> str:
    if pill_name in ["이미지 없음", "알약을 찾을 수 없음", "인식 결과 불분명", "연결 실패"] or "오류" in pill_name:
        return "알약 인식이 제대로 되지 않아 정보를 생성할 수 없습니다. 다시 촬영해 주세요."

    system_msg = (
        "당신은 복약 안내를 도와주는 친절한 약사입니다. "
        "사용자가 복용하려는 약의 이름을 알려주면, "
        "1) 어떤 약인지, 2) 일반적인 효능, 3) 기본 복용 방법, "
        "4) 대표적인 주의사항/부작용을 쉽고 짧게 bullet 형식으로 설명해 주세요. "
        "의사가 아닌 AI 데모 서비스이므로, 마지막에 반드시 "
        "'정확한 복약 안내는 약사·의사와 상의해 주세요.'라는 문장을 포함해 주세요."
    )

    user_msg = f"약 이름: {pill_name}\n이 약에 대해 위 기준에 맞게 한국어로 설명해 주세요."

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT 설명 생성 중 오류가 발생했습니다: {e}"

# --- 3. Gradio 분석 메인 함수 ---
def analyze_pill(image):
    if image is None:
        return "이미지가 업로드되지 않았습니다.", ""

    pill_name, prob = classify_pill(image)
    detail = explain_pill_with_gpt(pill_name)

    header_text = f"예측된 약 이름: {pill_name} (신뢰도: {prob:.1f}%)"
    return header_text, detail

# --- 4. Gradio UI (CSS 및 화면 구성) ---
custom_css = """
body, .gradio-container { background-color: #ffffff !important; font-family: sans-serif; }
.pill-phone-card { max-width: 800px; margin: 20px auto; background: #ffffff; border-radius: 32px; box-shadow: 0 10px 40px rgba(0,0,0,0.06); padding: 30px; border: 1px solid #f0f0f0; }
.pill-landing-title, .pill-header-title { font-size: 32px; font-weight: 800; text-align: center; color: #38B2AC; margin-bottom: 8px; }
.pill-landing-sub, .pill-header-sub { text-align: center; font-size: 14px; color: #718096; margin-bottom: 24px; }
.pill-landing-box { margin-top: 10px; padding: 20px; border-radius: 20px; background: #F0FFF4; border: 1px dashed #B2F5EA; font-size: 13px; color: #2D3748; }
.pill-landing-start-btn { margin-top: 20px; width: 100%; background: linear-gradient(135deg, #4FD1C5, #38B2AC) !important; color: #ffffff !important; font-weight: 800 !important; border-radius: 16px !important; height: 50px; border: none !important; cursor: pointer; }
.pill-btn-main { background: linear-gradient(135deg, #4FD1C5, #38B2AC) !important; color: #ffffff !important; font-weight: 800 !important; border-radius: 16px !important; border: none !important; height: 54px !important; cursor: pointer; }
.pill-btn-sub { background: #E6FFFA !important; color: #2C7A7B !important; border-radius: 16px !important; border: none !important; height: 54px !important; cursor: pointer; }
.pill-result-box { margin-top: 16px; padding: 16px; border-radius: 20px; background: #ffffff; border: 1px solid #E6FFFA; }
.pill-result-title { font-size: 14px; font-weight: 700; color: #2C7A7B; margin-bottom: 8px; }
"""

with gr.Blocks(css=custom_css, title="AI 복약 가이드") as demo:
    with gr.Column(elem_classes=["pill-phone-card"]):
        # 1화면: 랜딩
        with gr.Column(visible=True) as landing_group:
            gr.Markdown("""
            <div class="pill-landing-title">AI 복약 가이드</div>
            <div class="pill-landing-sub">사진 한 장으로 약 정보를 빠르게 확인하세요.</div>
            <div class="pill-landing-box">
              <div class="pill-landing-step">① 알약을 또렷하게 촬영해 주세요.</div>
              <div class="pill-landing-step">② 사진을 업로드하고 분석하기를 누르세요.</div>
              <div class="pill-landing-step">③ 상세한 복약 가이드를 확인하세요.</div>
            </div>
            """)
            start_btn = gr.Button("시작하기", elem_classes=["pill-landing-start-btn"])

        # 2화면: 도구
        with gr.Column(visible=False) as tool_group:
            gr.Markdown('<div class="pill-header-title">AI 복약 가이드</div>')
            image_in = gr.Image(type="pil", label="알약 사진 업로드", height=280)
            
            with gr.Row():
                clear_btn = gr.Button("다시 선택", elem_classes=["pill-btn-sub"])
                submit_btn = gr.Button("결과 분석하기", elem_classes=["pill-btn-main"])

            with gr.Column(elem_classes=["pill-result-box"]):
                gr.Markdown('<div class="pill-result-title">🔍 인식된 약품 정보</div>')
                pill_header = gr.Textbox(label="", interactive=False, placeholder="결과가 여기에 표시됩니다.")

            with gr.Column(elem_classes=["pill-result-box"]):
                gr.Markdown('<div class="pill-result-title">💊 상세 복약 가이드</div>')
                pill_detail = gr.Textbox(label="", interactive=False, lines=10, placeholder="설명이 여기에 표시됩니다.")

    # 버튼 이벤트
    start_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [landing_group, tool_group])
    submit_btn.click(analyze_pill, image_in, [pill_header, pill_detail])
    clear_btn.click(lambda: (None, "", ""), None, [image_in, pill_header, pill_detail])

if __name__ == "__main__":
    demo.launch()