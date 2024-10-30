import cv2
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def detect_light_changes(video_path, threshold=25, min_area=500, fault_color=(0, 140, 255), normal_color=(0, 255, 0)):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error("无法打开视频文件")
        return

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        logging.error("无法读取视频帧")
        return

    # 转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    while True:
        # 读取下一帧
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算帧差分
        diff = cv2.absdiff(prev_gray, gray)

        # 阈值化
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # 轮廓检测
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤噪声并绘制轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:  # 过滤小面积
                continue

            # 根据某些条件判断设备状态
            x, y, w, h = cv2.boundingRect(contour)

            # 假设这里用面积大小简单判断状态（仅为示例）
            if area > 1000:  # 这个条件可以根据实际需求调整
                color = fault_color  # 橘色表示故障
                status = "Fault"
            else:
                color = normal_color  # 绿色表示正常运行
                status = "Normal"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            logging.info(f"Frame {frame_count}: Detected area {area} at position ({x}, {y}) with status {status}")

        # 显示结果
        cv2.imshow('Frame', frame)

        # 更新前一帧
        prev_gray = gray

        # 按下 'q' 键退出
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 使用视频文件进行测试
video_file_path = 'video.mp4'
detect_light_changes(video_file_path)
