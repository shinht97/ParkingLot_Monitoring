# 📌 프로젝트명: Car Detection and Alert System  

---

📄 프로젝트 설명:  
이 파이썬 스크립트는 YOLOv5 ONNX 모델과 OpenCV, MSS를 사용하여
지정된 감지 영역 내 차량이 탐지되면 사운드를 재생하고,
Flask 서버를 통해 실시간 스트리밍도 제공합니다.

---

✅ 주요 기능:  
- YOLOv5 ONNX를 통한 차량 탐지 (class_id = 2)
- mss를 사용한 화면 캡처 기반 객체 탐지
- 차량이 감지 영역에 진입 시 사운드 5초간 재생
- 웹 페이지를 통한 실시간 스트리밍 제공 (http://localhost:5000/video_feed)
- 동일 차량 지속 감지 시 중복 알람 방지

---

📦 필요 라이브러리:  
```
pip install -r requirements.txt 로 설치하거나 개별 설치:
- numpy
- opencv-python
- onnxruntime
- mss
- flask
- pygame
```

---

🎵 알람 사운드:  
- 프로젝트 루트에 sound.wav 파일이 있어야 함  

---

🖼️ 사용 모델:  
- yolov5s.onnx (루트 경로에 위치해야 함)

---

📐 감지 영역:  
- DETECTION_AREA 변수로 지정된 4개의 꼭짓점으로 정의된 사각형

---

▶ 실행 방법:
```
python main.py
```

---  

🔒 웹 스트리밍은 별도 웹 브라우저에서 확인 가능:  
```
http://localhost:5000/video_feed
```

---

📁 exe로 패키징 시 참고:  
PyInstaller 사용 시 사운드 및 ONNX 파일은 resource_path 함수로 경로 설정됨

