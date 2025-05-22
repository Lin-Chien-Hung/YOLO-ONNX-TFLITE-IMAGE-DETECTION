import cv2
import numpy as np
import tensorflow as tf
import cvzone


# 影像前處理 (原始影像, 模型(寬), 模型(高))
def preprocess(org_image, dst_w, dst_h):

    image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
    
    # 調整輸入影像 至 符合模型大小
    image = cv2.resize(org_image, (dst_w, dst_h))
    
    image = image.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(image, axis=0)

    # 回傳處理完畢的影像
    return input_tensor


if __name__ == '__main__':
    
    # 載入 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path="./model/yolov11n_float32.tflite")
    interpreter.allocate_tensors()
    
    # 取得輸入與輸出資訊
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 載入類別清單
    with open("./classification/coco.txt", "r") as f:
        class_list = f.read().splitlines()

    # 讀取圖片
    image = cv2.imread("./image/bus.jpg")
    
    # 讀取模型的維度[1, 640, 640, 3](影像辨識張數, 高, 寬, 通道數) | 高、寬，將根據模型訓練時，所提供的影像而定，在此為 640 * 640
    input_shape = input_details[0]['shape']
    
    # 將輸入 影像或影片 調整至與 模型 相同大小 (前處理)
    input_tensor = preprocess(image, input_shape[2], input_shape[1])
    
    # 將 reshape 過後的影像 輸入至模型當中進行推論(辨識)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # 取得推論結果
    detections = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (num_detections, 6)
    
    for det in detections:
        # 辨識物件 左上座標點(x1,y1), 右下座標點(x2,y2), 信心分數, 辨識類別
        x1, y1, x2, y2, conf, class_id = det
        # 過濾信心分數低於0.5以下的分數
        if conf < 0.5:
            continue
        # 辨識出的物件名稱
        label = class_list[int(class_id)]
        # 注意！這裡座標通常是「相對於輸入圖大小」，所以要根據 image 來縮放
        x1 = int(x1 * image.shape[1])
        y1 = int(y1 * image.shape[0])
        x2 = int(x2 * image.shape[1])
        y2 = int(y2 * image.shape[0])
    
        # 畫框
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 添加物件名稱
        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    # 存檔  
    cv2.imwrite("output.jpg", image)

