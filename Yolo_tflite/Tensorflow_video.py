import cv2
import numpy as np
import tensorflow as tf
import cvzone

# 載入 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="./model/yolov10n_float32.tflite")
interpreter.allocate_tensors()

# 取得輸入與輸出資訊
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 載入類別清單
with open("./classification/coco.txt", "r") as f:
    class_list = f.read().splitlines()

# 視訊輸入
cap = cv2.VideoCapture("t.mp4")
cv2.namedWindow('RGB')

# 抓影像尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 調整畫面大小 (可選，或直接用原始大小)
    frame = cv2.resize(frame, (1020, 600))

    # 前處理：轉為模型輸入格式
    input_shape = input_details[0]['shape']  # [1, 640, 640, 3]
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # 執行推論
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # 取得推論結果
    detections = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (num_detections, 6)

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det

        if conf < 0.5:
            continue

        # 注意！這裡座標通常是「相對於輸入圖大小」，所以要根據 frame 來縮放
        x1 = int(x1 * frame.shape[1])
        y1 = int(y1 * frame.shape[0])
        x2 = int(x2 * frame.shape[1])
        y2 = int(y2 * frame.shape[0])

        # 防止超出畫面邊界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if int(class_id) >= len(class_list):
            continue  # 避免 class_id 超出範圍

        label = class_list[int(class_id)]
        #cvzone.putTextRect(frame, f'{label} {conf:.2f}', (x1, y1), 1, 1)

        # 如果想畫框也可以打開
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

