# ⦿ Object-Detect-With-YOLO-ONNX-And-TFLITE-In-V10、v11

* Title : Object-Detect-With-YOLO-ONNX-And-TFLITE-In-V10、v11
* Author : 林建宏 (Lin, Chien-Hung)
* 
![image](./gemini.png)

## ⦿ 題目 (topic)
開發一個網頁應用程式，允許使用者輸入文字內容，並即時顯示對該內容的情感分析結果（正面、負面或中性...等，名目您可以自己決定）。
服務使用Google Cloud Platforｍ雲端部建，不限制使用VM、容器、或無伺服器架構達成，但必須串接Gemini API的回訊功能。

此項實作包含程式介面、服務環境搭建，以及LLM模型服務串接。目的是檢驗您能否適任全端作業模式。
完成後請將服務上線並提供網站網址，並將原始碼放上Github，回信時請您簡單說明使用了哪幾項服務功能。

## ⦿ 環境 (Requirements)
### 雲端伺服器(Cloud server)
* Google Cloud Platform (GCP) CPU:N2D 空間:100G 防火牆:建立防火牆規則，開啟port並設定TCP/UDP。
### 虛擬環境(Virtual Machine, VM)
* Ubuntu 24.01 LTS x86/64
* Docker 2.5.0-cuda12.4-cudnn9-devel
* Jupyter-lab
* html
* Python 3.11.10

## ⦿ (程式)資歷夾中具以下兩種檔案 ：
- **emotion_detection.html : 網頁程式介面，負責接收文字訊息，並將接收到的資料傳回至後端(Emotion_detect.py)進行處理。
- **Emotion_detect.py : 將接收到的前端資料發送給 Gemini Api 做情緒上的理解，並將處理結果回傳至前端(emotion_detection.html)進行輸出。
- **gemini.png : Gemini 網上的圖片。(來源 : https://blog.google/technology/ai/gemini-api-developers-cloud/)

## ⦿ 操作流程(Operation process)：
1. 開啟terminal，進入具有.py的資料夾，輸入以下指令：python3 Emotion_detect.py。
2. 開啟terminal，進入具有HTML的資料夾，輸入以下指令：python3 -m http.server 6006 --bind 0.0.0.0。
3. 當使用者輸入文字並按下按鈕過後，系統將會回傳情緒預測回HTML。

## ⦿ 創建、撰寫流程、服務功能(Create and write processes and service functions)
1. GCP 環境建立 : 用於 VM 環境建立.
2. VM 環境建立 : 用於程式撰寫、模型訓練的地方.
3. Docker 環境建立 : 用於建立各種不同執行環境的地方.
4. Jupyter 套件安裝 : 更加便於程式撰寫及觀看的地方.
5. 防火牆設定 : 提供更多的埠(port)的地方.
6. 程式撰寫 : html程式撰寫、串接 Gemini API、 Prompt參數設定、美觀編排.
7. 服務啟動
