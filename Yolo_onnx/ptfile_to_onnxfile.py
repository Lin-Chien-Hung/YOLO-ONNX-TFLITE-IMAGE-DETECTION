from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("./model/yolov11n.pt")

# yolov10 nms=False, yolov11 nms=True
# Export the model to ONNX format
model.export(format="onnx",nms=True)  

# Load the exported ONNX model
#onnx_model = YOLO("yolov10n.onnx")

# Run inference
#results = onnx_model("https://ultralytics.com/images/bus.jpg")

