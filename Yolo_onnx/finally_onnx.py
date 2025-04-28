import cv2
import numpy as np
import onnxruntime as ort

# 影像前處理
def preprocess(bgr_image, src_w, src_h, dst_w, dst_h):
    # BGR 轉 RGB
    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 計算縮放比例與黑邊
    ratio = min(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * ratio / 2) * 2)
    new_h = int(round(src_h * ratio / 2) * 2)
    x_offset = (dst_w - new_w) // 2
    y_offset = (dst_h - new_h) // 2

    # 調整大小，加黑邊
    image = cv2.resize(image, (new_w, new_h))
    image = cv2.copyMakeBorder(
        image, y_offset, y_offset, x_offset, x_offset, 
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # 正規化 + 調整維度
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    input_tensor = np.expand_dims(image, axis=0)

    return input_tensor, ratio, x_offset, y_offset

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
    input_tensor, ratio, x_offset, y_offset = preprocess(image, image_width, image_height, model_width, model_height)

    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    output = np.squeeze(outputs[0])

    print(output.shape)

    # 類別
    with open('./classification/coco.txt') as f:
        class_list = f.read().strip().split('\n')

    for i in range(output.shape[0]):
        confidence = output[i][4]
        if confidence > 0.1:
            label = int(output[i][5])
            xmin = int((output[i][0] - x_offset) / ratio)
            ymin = int((output[i][1] - y_offset) / ratio)
            xmax = int((output[i][2] - x_offset) / ratio)
            ymax = int((output[i][3] - y_offset) / ratio)

            # 畫框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            # 標上類別與置信度
            text = f"{class_list[label]} {confidence:.2f}"
            cv2.putText(image, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite("output.jpg", image)
    #cv2.imshow("Detected Objects", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
