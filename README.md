# ⦿ Object-Detect-With-YOLO-ONNX-And-TFLITE-In-V10、v11

* Title : Object-Detect-With-YOLO-ONNX-And-TFLITE-In-V10、v11
* Author : 林建宏 (Lin, Chien-Hung)

## ⦿ 題目 (topic)

本研究採用影像辨識技術 YOLO V10 及 V11，並將原始 PyTorch（.pt）模型格式轉換為 ONNX（.onnx）與 TFLite（.tflite），以利於模型部署與應用。

## ⦿ (程式)資歷夾中具以下兩種檔案 ：
- **Yolo_onnx  : 將輸入模型 Yolo Pytorch(.pt) 轉換至 Onnx(.onnx) 並做影像辨識。
- **Yolo_tflite: 將輸入模型 Yolo Pytorch(.pt) 轉換至 TFLite(.tflite) 並做影像辨識。

## ⦿ 操作流程(Operation process)：
1. 先將模型輸入至各個資料夾當中轉換型態的程式，轉換至對應的資料格式。
2. 在使用 Yolo v10 時 nms 請設置為 Fales # (300, 6)
3. 在使用 Yolo v11 時 nms 請設置為 True  # (8400,84) ==> (300,6)

## ⦿ 創建、撰寫流程、服務功能(Create and write processes and service functions)

2. VM 環境建立 : 用於程式撰寫、模型訓練的地方.
3. Docker 環境建立 : 用於建立各種不同執行環境的地方.
4. Jupyter 套件安裝 : 更加便於程式撰寫及觀看的地方.
5. 防火牆設定 : 提供更多的埠(port)的地方.
6. 程式撰寫 : html程式撰寫、串接 Gemini API、 Prompt參數設定、美觀編排.
7. 服務啟動
