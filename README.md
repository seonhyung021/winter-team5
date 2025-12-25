# winter-team5
💊 무물약 (무엇이든 물어보세요 약사님) — Azure 기반 약품 식별 & 복약 안내 서비스 💊
<img width="800" src="/files/img.png">  
 사진 한 장으로 약품을 인식하고, Azure OpenAI를 통해 복약 안내를 제공하는 AI 서비스
 Microsoft Azure Custom Vision + Azure OpenAI + OCR 기반 프로젝트
 숙명여자대학교 Azure Winter School Team 5 프로젝트

⸻

 데모 영상

<video src="/files/demo1.mp4" controls width="800"></video>


⸻

 주요 화면
랜딩화면
<br>
<img width="600" src="/files/landing.png"> 

약 인식


<img width="600" src="/files/teest1.jpg"> 
<br>
<img width="600" src="/files/test1.jpg"> 

 프로젝트 목표
	•	 약 사진을 업로드하면
	•	 Custom Vision이 약 종류를 분류
	•	 Azure OpenAI가 약 설명 + 복용법 + 주의사항을 자연어로 생성
	•	 사용자가 이해하기 쉬운 복약 가이드 제공
	•	 교육용 데모 수준이지만 실제 서비스 UX와 최대한 유사하게 구현

⸻

 주요 기능

1️⃣ 약 이미지 분류 (Azure Custom Vision)

<img width="600" src="/files/learning.PNG"> 

	•	OCR (Azure Ai Service)로 이중 분류해 정확도 높임
	•	학습된 이미지 기반 Tag 분류
	•	신뢰도(%) 제공

⸻

2️⃣ GPT 기반 약 설명 생성 (Azure OpenAI)
<img width="600" src="/files/test2.png"> 

	•	예측된 약 이름 기반
	•	다음 정보 제공:
	•	약 설명
	•	효능
	•	기본 복용 방법
	•	주의사항 / 부작용
	•	한국어 자연어 서술
<img width="600" src="/files/question.png"> 
	•	추가 질문 
	<br>
⸻

3️⃣ 사용자 친화적 UI (Gradio)
	•	모바일 앱 느낌의 카드형 UI
	•	민트톤 의료 UX 색상 시스템
	•	랜딩 화면 / 기능 화면 분리

⸻

 개인정보 & 보안 안내
※ 업로드 가능한 이미지 용량은 최대 6MB입니다.
※ 업로드된 사진은 약 분류 및 안내 제공을 위한 분석에만 일시적으로 사용됩니다.
※ 개인 정보는 저장되지 않으며, 분석 후 즉시 폐기됩니다.
※ 본 프로젝트는 교육용 데모 서비스입니다.

 시스템 구조
[사용자]
   ↓ 이미지 업로드
[Gradio UI]
   ↓
[Azure Custom Vision, Azure AI Service]
   → 약 분류 / 확률 계산
   ↓
[Azure OpenAI]
   → 약 설명 + 복약 가이드 생성
   ↓
[사용자에게 결과 제공]

 사용 기술
Frontend - Gradio UI Custom CSS
Backend - Python
AI Vision - Azure Custom Vision
NLP - Azure OpenAI GPT
OCR (확장 가능) - Azure AI Vision
Infra - Microsoft Azure
협업 - GitHub

 실행 방법

1️⃣ 필수 라이브러리
pip install gradio
pip install requests
pip install python-dotenv
pip install openai

2️⃣ .env 파일 생성
PREDICTION_URL=
PREDICTION_KEY=

AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
DEPLOYMENT_NAME=

AZURE_VISION_ENDPOINT=
AZURE_VISION_KEY=

3️⃣ 실행
python newmain.py

 확장 가능성
	•	OCR 기반 알약 표면 문자 인식 정확도 향상
	•	실제 약 데이터베이스 연동 (식약처 API 등)
	•	약물 병용 위험 분석 추가
	•	모바일 앱으로 확장 가능

⸻

 한계
	•	교육용 수준 정확도
	•	학습 데이터 범위 밖 약품 인식 불가
	•	의료 자문 대체 불가

 Acknowledgements
	•	Microsoft Azure
	•	숙명여자대학교 Azure Winter School Program
