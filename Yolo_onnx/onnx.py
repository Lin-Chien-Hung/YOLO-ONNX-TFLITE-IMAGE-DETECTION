import cv2
import numpy as np
import onnxruntime as ort

# 影像前處理 (原始影像, 原始影像(寬), 原始影像(高), 模型(寬), 模型(高))
def preprocess(bgr_image, src_w, src_h, dst_w, dst_h):
    
    # BGR 轉 RGB
    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 調整輸入影像 至 符合模型大小
    image = cv2.resize(image, (dst_w, dst_h))

    # 正規化 + 調整維度
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    input_tensor = np.expand_dims(image, axis=0)

    # 回傳處理完畢的影像
    return input_tensor

if __name__ == '__main__':
    
    # 使用 ort 加速 ONNX 模型，並載入
    Onnx_model = ort.InferenceSession("./model/yolov11n.onnx", providers=["CUDAExecutionProvider"])

    # 讀取訓練時，所提供的物件類別
    with open('./classification/coco.txt') as f:
        class_list = f.read().strip().split('\n')
    
    # 讀取圖片
    image = cv2.imread("./image/bus.jpg")
    
    # 讀取原始圖片 寬 & 高，後續物件畫框時，還原用
    image_height, image_width, _ = image.shape
    
    # 讀取模型的維度[1, 3, 640, 640](影像辨識張數, 通道數, 高, 寬) | 高、寬，將根據模型訓練時，所提供的影像而定，在此為 640 * 640
    _, _, model_height, model_width = Onnx_model.get_inputs()[0].shape
    
    # 將輸入 影像或影片 調整至與 模型 相同大小 (前處理)
    input_tensor = preprocess(image, image_width, image_height, model_width, model_height)

    # 將 reshape 過後的輸入檔輸入至模型當中進行推論(辨識)
    outputs = Onnx_model.run(None, {Onnx_model.get_inputs()[0].name: input_tensor})
    output = np.squeeze(outputs[0])

    # 抽取 推論 過後的輸出
    for det in output:
        # 辨識物件 左上座標點(x1,y1), 右下座標點(x2,y2), 信心分數, 辨識類別
        x1, y1, x2, y2, conf, class_id = det
        # 過濾信心分數低於0.5以下的分數
        if conf < 0.5:
            continue
        # 辨識出的物件名稱
        label = class_list[int(class_id)]
        # 根據 640 × 640 的影像辨識結果，將其辨識位置還原至原始圖片上
        x1 = int(x1 * (image_width / model_width))
        y1 = int(y1 * (image_height / model_height))
        x2 = int(x2 * (image_width / model_width))
        y2 = int(y2 * (image_height / model_height))

        # 畫框
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 添加物件名稱
        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 存檔
    cv2.imwrite("output.jpg", image)
    #cv2.imshow("Detected Objects", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
