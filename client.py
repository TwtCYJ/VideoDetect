import cv2
import base64
import socketio
import json

# 创建SocketIO客户端
sio = socketio.Client()

@sio.event
def connect():
    print("Connected to server")

@sio.event
def disconnect():
    print("Disconnected from server")

@sio.on('processing_result')
def on_message(data):
    # 打印接收到的识别结果
    print("Received:", data)
    global detection_results
    detection_results = json.loads(data)

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    img_data = base64.b64encode(buffer).decode('utf-8')
    return img_data

def draw_boxes(frame, detections):
    for detection in detections:
        x, y, w, h = detection['bbox']
        label = detection.get('label', 'Object')
        confidence = detection.get('confidence', 0.0)

        # 绘制识别框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    # 连接到SocketIO服务器
    sio.connect('http://localhost:5000')

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    global detection_results
    detection_results = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            encoded_frame = encode_frame(frame)
            sio.emit('video_frame', encoded_frame)

            # 在帧中绘制识别框
            draw_boxes(frame, detection_results)

            # 显示视频帧
            cv2.imshow('Client Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sio.disconnect()

if __name__ == "__main__":
    main()
