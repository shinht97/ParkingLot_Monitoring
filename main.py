import cv2
import numpy as np
import onnxruntime as ort
import mss
import sys
import os
import threading
import time
from flask import Flask, Response
import pygame

# ------------------ 설정 ------------------
DEBUG = True
ONNX_MODEL_PATH = "./yolov5s.onnx"
SOUND_FILE = "./sound.wav"
CAR_CLASS_ID = 2
CONF_THRESHOLD = 0.4
DETECTION_AREA = [[785, 310], [1135, 200], [1625, 370], [1320, 615]]
SOUND_DURATION = 5  # 초
# -----------------------------------------

# 초기화
pygame.mixer.init()
sound = pygame.mixer.Sound(SOUND_FILE)

# 웹 스트리밍
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

@app.route("/video_feed")
def video_feed():
    def generate():
        global output_frame
        while True:
            with lock:
                if output_frame is None:
                    continue
                _, buffer = cv2.imencode('.jpg', output_frame)
                frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 모델 로드
session = ort.InferenceSession(resource_path(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape[2:]

def preprocess(_img):
    resized = cv2.resize(_img, input_shape)
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    return img

def postprocess(prediction, original_shape):
    boxes = []
    for pred in prediction[0][0]:
        conf = pred[4]
        if conf < CONF_THRESHOLD:
            continue
        cls_id = np.argmax(pred[5:])
        if cls_id != CAR_CLASS_ID:
            continue
        x, y, w, h = pred[0:4]
        x1 = int((x - w/2) * original_shape[1] / input_shape[0])
        y1 = int((y - h/2) * original_shape[0] / input_shape[1])
        x2 = int((x + w/2) * original_shape[1] / input_shape[0])
        y2 = int((y + h/2) * original_shape[0] / input_shape[1])
        boxes.append((x1, y1, x2, y2))
    return boxes

def is_left(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0

def is_point_in_area(point, area):
    a, b, c, d = area
    return is_left(point, a, b) and is_left(point, b, c) and is_left(point, c, d) and is_left(point, d, a)

def play_sound():
    sound.play(loops=2)
    time.sleep(SOUND_DURATION)

def main():
    global output_frame

    # 웹 스트리밍 서버 시작
    threading.Thread(target=run_flask, daemon=True).start()

    with mss.mss() as sct:
        
        triggered = False
        last_detection_time = 0

        if DEBUG:
            cv2.namedWindow("Car Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Car Detection", 800, 480)
            monitor = sct.monitors[2]
        else:
            monitor = sct.monitors[1]
        try:
            while True:
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                input_tensor = preprocess(frame)
                outputs = session.run(None, {input_name: input_tensor})
                boxes = postprocess(outputs, frame.shape)

                car_in_area = False

                for box in boxes:
                    x1, y1, x2, y2 = box
                    x_quarter = int(x2 - (x2 - x1) / 4)
                    y_center = int((y1 + y2) / 2)
                    point = (x_quarter, y_center)

                    if is_point_in_area(point, DETECTION_AREA):
                        car_in_area = True

                        if not triggered:
                            print("🚗 차량 감지됨 - 소리 재생 중...")
                            threading.Thread(target=play_sound, daemon=True).start()
                            triggered = True
                            last_detection_time = time.time()

                        break  # 하나만 처리

                # 차량이 더 이상 영역에 없다면 트리거 해제
                if not car_in_area:
                    if triggered:
                        if time.time() - last_detection_time > SOUND_DURATION:
                            print("🚗 차량 사라짐 - 트리거 리셋")
                            triggered = False

                # 시각화
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    x_quarter = int(x2 - (x2 - x1) / 4)
                    y_center = int((y1 + y2) / 2)
                    cv2.circle(frame, (x_quarter, y_center), 5, (0, 255, 0), -1)

                cv2.polylines(frame, [np.array(DETECTION_AREA, np.int32)], True, (255, 255, 255), 2)

                with lock:
                    output_frame = frame.copy()

                if DEBUG:
                    cv2.imshow("Car Detection", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break

        except KeyboardInterrupt:
            print("🛑 사용자 종료 요청됨")
        finally:
            if DEBUG:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
