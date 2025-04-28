import cv2
import numpy as np
import tensorflow as tf
import cvzone

# 載入 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="./model/yolov11n_float32.tflite")
interpreter.allocate_tensors()

# 取得輸入與輸出資訊
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 載入類別清單
with open("./classification/coco.txt", "r") as f:
    class_list = f.read().splitlines()

image = cv2.imread("./image/bus.jpg")

# 前處理：轉為模型輸入格式
input_shape = input_details[0]['shape']  # [1, 640, 640, 3]
img = cv2.resize(image, (input_shape[2], input_shape[1]))
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

    # 注意！這裡座標通常是「相對於輸入圖大小」，所以要根據 image 來縮放
    x1 = int(x1 * image.shape[1])
    y1 = int(y1 * image.shape[0])
    x2 = int(x2 * image.shape[1])
    y2 = int(y2 * image.shape[0])

    # 防止超出畫面邊界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    if int(class_id) >= len(class_list):
        continue  # 避免 class_id 超出範圍

    label = class_list[int(class_id)]
    #cvzone.putTextRect(image, f'{label} {conf:.2f}', (x1, y1), 1, 1)

    # 如果想畫框也可以打開
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
cv2.imwrite("output.jpg", image)

