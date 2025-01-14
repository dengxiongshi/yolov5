from ultralytics import YOLO

weights = r'../weights/yolov8s.pt'

model = YOLO(model=weights)
model.export(format="onnx", imgsz=(384, 640), simplify=True, opset=11, device=0)