import io
import os
import requests
import gradio as gr
from openai import AzureOpenAI

# Custom Vision 설정 
PREDICTION_URL = "https://customvisionpredictionstudent07-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/c60aaa22-46b0-40ff-9078-79f00e82e4d0/classify/iterations/Iteration1/image"   # ?iterationId=... 까지 포함된 전체 URL
PREDICTION_KEY = "3ZazAn012xcZ0Wz7oP5TvoqhvKnyscOxDaPAsdBybER6iW6uovJeJQQJ99BLACLArgHXJ3w3AAAIACOGa5Yl"

#  Azure OpenAI 설정
AZURE_OPENAI_ENDPOINT = "https://team5-openai.openai.azure.com/"
AZURE_OPENAI_API_KEY = "3eu1gnnzE0D5CLzz2AwO7VGZKWv32cPU7CTy9jwdTPdqdFWW8297JQQJ99BLACNns7RXJ3w3AAABACOGKUsp"

DEPLOYMENT_NAME = "gpt-4o-mini"

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
demo = gr.Interface(
    fn=pill_pipeline,
    inputs=gr.Image(type="pil", label="알약 사진 업로드"),
    outputs=[
        gr.Textbox(label="분류 결과"),
        gr.Textbox(label="약 설명 (Azure OpenAI)", lines=12),
    ],
    title="알약 인식 & 설명 챗봇",
    description="알약 사진을 올리면 어떤 약인지 분류하고, AI가 복용 설명과 주의사항을 알려줍니다.",
)

if __name__ == "__main__":
    demo.launch()
