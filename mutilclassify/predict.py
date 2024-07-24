# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 classification inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python classify/predict.py --weights yolov5s-cls.pt --source 0                               # webcam
                                                                   img.jpg                         # image
                                                                   vid.mp4                         # video
                                                                   screen                          # screenshot
                                                                   path/                           # directory
                                                                   list.txt                        # list of images
                                                                   list.streams                    # list of streams
                                                                   'path/*.jpg'                    # glob
                                                                   'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                                   'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python classify/predict.py --weights yolov5s-cls.pt                 # PyTorch
                                           yolov5s-cls.torchscript        # TorchScript
                                           yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                           yolov5s-cls_openvino_model     # OpenVINO
                                           yolov5s-cls.engine             # TensorRT
                                           yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                           yolov5s-cls_saved_model        # TensorFlow SavedModel
                                           yolov5s-cls.pb                 # TensorFlow GraphDef
                                           yolov5s-cls.tflite             # TensorFlow Lite
                                           yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                           yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import platform
import pathlib

import numpy as np

plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    print_args,
    strip_optimizer,
)
from utils.torch_utils import select_device, smart_inference_mode


def detect_day_or_night(img):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 计算全局平均亮度
    global_mean_brightness = np.mean(hsv[:, :, 2])

    # 局部亮度分析：检测图像中的高亮区域
    bright_areas = cv2.inRange(hsv, (0, 0, 200), (255, 255, 255))
    bright_ratio = np.sum(bright_areas) / (bright_areas.shape[0] * bright_areas.shape[1])

    # 边缘检测
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])

    # 检测天空区域
    sky_mask = cv2.inRange(hsv, (90, 0, 0), (140, 255, 255))
    sky_ratio = np.sum(sky_mask) / (sky_mask.shape[0] * sky_mask.shape[1])

    # 阈值
    global_brightness_threshold = 50
    bright_area_threshold = 0.01
    edge_density_threshold = 0.02
    sky_threshold = 0.1

    if (global_mean_brightness > global_brightness_threshold or bright_ratio > bright_area_threshold) and (
            edge_density > edge_density_threshold or sky_ratio > sky_threshold):
        return True
    else:
        return False
    
    
def draw_txt(image, text):
    img = image.copy()
    (h, w) = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # 字体大小，可以根据需要调整
    color = (255, 0, 0)  # 文本颜色，这里是白色
    thickness = 2  # 线条粗细

    # 获取文本边界框的尺寸
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 计算文本中心位置（水平居中，上部1/10位置）
    text_x = (w - text_width) // 2
    text_y = h // 10  # 或者使用其他固定值，如 text_y = 50
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return img


@smart_inference_mode()
def run(
        weights=ROOT / "yolov5s-cls.pt",  # model.pt path(s)
        source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "data/coco128.yaml",  # dataset.yaml path
        imgsz=(224, 224),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        nosave=False,  # do not save images/videos
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/predict-mlcls",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.Tensor(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            results = model(im)

        # Post-process
        with dt[2]:
            # Convert outputs to binary predictions (0 or 1)
            pred = F.sigmoid(results)

        # Process predictions
        for i, prob in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt

            s += "%gx%g " % im.shape[2:]  # print string
            annotator = Annotator(im0, example=str(names), pil=True)

            # Print results
            # top5i = prob.argsort(0, descending=True)[:5].tolist()  # top 5 indices
            # s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "
            # conf = (prob > conf_thres).float()
            # result = [i for i, x in enumerate(conf.tolist()) if x != 0]
            # s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in result)}, "
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in range(len(prob)))}, "

            thres = 0.8
            thunder_thres = 0.95
            # Write results
            if detect_day_or_night(im0):
                day_night_txt = 'day'
                s += 'day,   '
                prob[1] = prob[1] if prob[1] >= thres else 0  # 云概率提升
                if prob[5] >= conf_thres and prob[3] >= thres:
                    prob[2] = 0  # 同时存在闪电和下雨，则无雾
                elif prob[5] >= conf_thres and prob[2] < thres:
                    prob[2] = 0
                if prob[2] < thres:
                    prob[2] = 0
                if prob[3] < thres:
                    prob[3] = 0
                elif prob[1] >= conf_thres and prob[2] >= conf_thres and prob[3] >= thres:
                    prob[3] = 0
            else:
                day_night_txt = 'night'
                s += 'night, '
                prob[2], prob[3] = 0, 0  # 没雾, 没雨
                if prob[1] < conf_thres:
                    prob[5] = 0
                elif prob[5] < thunder_thres:   # 闪电概率提升
                    prob[5] = 0

            conf = (prob > conf_thres).float()
            result = [i for i, x in enumerate(conf.tolist()) if x != 0]
            text = "\n".join(f"{prob[j]:.2f} {names[j]}" for j in result)
            
            if save_img or view_img:  # Add bbox to image
                annotator.text([32, 32], text, txt_color=(255, 255, 255))
            if save_txt:  # Write to file
                with open(f"{txt_path}.txt", "a") as f:
                    f.write(text + "\n")

            # Stream results
            im0 = annotator.result()
            im0 = draw_txt(im0, day_night_txt)
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parses command line arguments for YOLOv5 inference settings including model, source, device, and image size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train-mlcls/weather/weights/best.pt",
                        help="model path(s)")
    parser.add_argument("--source", type=str, default=r"\\10.10.10.8\determined\dengxiongshi\yolov5\datasets\6.mp4", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[224], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/predict-mlcls", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with options for ONNX DNN and video frame-rate stride adjustments."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
