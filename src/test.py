from super_gradients.training import models
from super_gradients.common.object_names import Models


model = models.get(Models.YOLO_NAS_S, num_classes=80, checkpoint_path='coco')
model.export("yolo_nas.onnx", preprocessing=True)
