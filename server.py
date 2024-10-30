from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import logging
import json

app = Flask(__name__)
socketio = SocketIO(app)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def detect_light_color(frame, frame_count, threshold=25, min_area=500):
    try:
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges for green and red in HSV
        green_lower = np.array([35, 100, 100])
        green_upper = np.array([85, 255, 255])

        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])

        # Create masks for green and red
        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Detect contours for green and red lights
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        results = []

        # Check for green light
        for contour in contours_green:
            area = cv2.contourArea(contour)
            if area > min_area:  # Adjust the area threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                results.append({"color": "green", "bbox": [x, y, w, h], "status": "Normal"})
                logging.info(f"Frame {frame_count}: Detected green light with area {area} at position ({x}, {y})")

        # Check for red light
        for contour in contours_red:
            area = cv2.contourArea(contour)
            if area > 100:  # Adjust the area threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                results.append({"color": "red", "bbox": [x, y, w, h], "status": "Fault"})
                logging.info(f"Frame {frame_count}: Detected red light with area {area} at position ({x}, {y})")

        return frame, results
    except Exception as e:
        logging.error(f"Error in detect_light_color: {e}")
        return frame, []


@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # 解析JSON数据
        if isinstance(data, str):
            data = json.loads(data)

        # Decode image data
        img_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Decoded frame is None")

        # 获取帧计数
        frame_count = data.get('frame_count', 0)

        processed_frame, results = detect_light_color(frame, frame_count)

        # 格式化结果为JSON对象
        result_json = []
        for result in results:
            color = result["color"]
            bbox = result["bbox"]
            area = (bbox[2] * bbox[3])  # Assuming area is width * height
            result_json.append({
                "frame": frame_count,
                "color": color,
                "area": area,
                "status": result["status"],
                "position": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
            })

            # Send JSON result back to the client
        emit('frame_result', result_json)
    except json.JSONDecodeError:
        logging.error("Error decoding JSON data")
        emit('frame_result', {"error": "Invalid JSON data"})
    except Exception as e:
        logging.error(f"Error in handle_video_frame: {e}")
        emit('frame_result', {"error": str(e)})


if __name__ == '__main__':
    # 在生产环境中禁用调试模式
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
