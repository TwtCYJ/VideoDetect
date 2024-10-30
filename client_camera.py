import cv2
import base64
import socketio
import json
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建SocketIO客户端
sio = socketio.Client()


@sio.event
def connect():
    logging.info("Connected to server")


@sio.event
def disconnect():
    logging.info("Disconnected from server")


@sio.on('frame_result')
def on_message(data):
    global detection_results
    detection_results = data  # 更新全局变量
    print('Received result:')
    for result in data:
        print(
            f"Frame {result['frame']}: Detected {result['color']} light with area {result['area']} "
            f"at position ({result['position']['x']}, {result['position']['y']}) with status {result['status']}")

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    img_data = base64.b64encode(buffer).decode('utf-8')
    return img_data


def draw_boxes(frame, detections):
    for detection in detections:
        # x, y, w, h = detection['position']
        x = detection['position']['x']
        y = detection['position']['y']
        w = detection['position']['w']
        h = detection['position']['h']
        label = detection.get('area', 'Object')
        confidence = detection.get('status', 0.0)

        # 绘制识别框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
        cv2.putText(frame, f'{label}: {confidence}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main():
    # 连接到SocketIO服务器
    sio.connect('http://localhost:5000')

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    global detection_results
    detection_results = []

    frame_count = 0  # 初始化帧计数

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video or cannot read the frame")
                break

            encoded_frame = encode_frame(frame)
            # 构建数据包，包含编码后的图像和帧计数
            data = {
                'image': encoded_frame,
                'frame_count': frame_count
            }
            sio.emit('video_frame', json.dumps(data))  # 发送JSON格式的数据
            logging.info(f"Sent frame {frame_count}")

            # 在帧中绘制识别框
            draw_boxes(frame, detection_results)

            # 显示视频帧
            cv2.imshow('Client Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Video playback interrupted by user")
                break

            frame_count += 1  # 增加帧计数

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sio.disconnect()


if __name__ == "__main__":
    main()
