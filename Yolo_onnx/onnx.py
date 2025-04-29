import cv2
import numpy as np
import onnxruntime as ort

# 影像前處理
def preprocess(bgr_image, src_w, src_h, dst_w, dst_h):
    # BGR 轉 RGB
    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 調整大小，加黑邊
    image = cv2.resize(image, (dst_w, dst_h))

    # 正規化 + 調整維度
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    input_tensor = np.expand_dims(image, axis=0)

    return input_tensor

if __name__ == '__main__':
    
    # 載入 ONNX 模型
    session = ort.InferenceSession("./model/yolov11n.onnx", providers=["CUDAExecutionProvider"])
    # 載入 圖片
    image = cv2.imread("./image/bus.jpg")
    # 把圖片的 寬 高 抓出來
    image_height, image_width, _ = image.shape
    # 640, 640    
    _, _, model_height, model_width = session.get_inputs()[0].shape
    # 圖片, 圖片 寬 高, 模型 寬 高
    input_tensor = preprocess(image, image_width, image_height, model_width, model_height)

    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    output = np.squeeze(outputs[0])

    # 類別
    with open('./classification/coco.txt') as f:
        class_list = f.read().strip().split('\n')

    for det in output:
        x1, y1, x2, y2, conf, class_id = det
        
        if conf < 0.5:
            continue
            
        label = class_list[int(class_id)]
        x1 = int(x1 * (image_width / model_width))
        y1 = int(y1 * (image_height / model_height))
        x2 = int(x2 * (image_width / model_width))
        y2 = int(y2 * (image_height / model_height))

        # 畫框
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite("output.jpg", image)
    #cv2.imshow("Detected Objects", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
