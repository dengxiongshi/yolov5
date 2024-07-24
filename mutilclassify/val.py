# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset.

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python classify/val.py --weights yolov5s-cls.pt                 # PyTorch
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
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from typing import List, Dict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.callbacks import Callbacks
from utils.metrics import ConfusionMatrix, compute_metrics

from models.common import DetectMultiBackend
from utils.dataloaders import create_classification_dataloader, create_multilabel_classification_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_img_size,
    check_requirements,
    colorstr,
    increment_path,
    print_args, check_dataset,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        data,
        weights=ROOT / "yolov5s-cls.pt",  # model.pt path(s)
        batch_size=128,  # batch size
        imgsz=224,  # inference size (pixels)
        conf_thres=0.5,  # confidence threshold
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        task="val",  # train, val, test
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        verbose=False,  # verbose output
        project=ROOT / "runs/val-mlcls",  # save to project/name
        name="exp",  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(""),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    model.eval()
    cuda = device.type != "cpu"
    nc = int(data["nc"])  # number of classes

    # Dataloader
    if not training:
        if pt:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_multilabel_classification_dataloader(
            data[task],
            imgsz,
            batch_size,
            nc,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))

    s = ("%22s" + "%11s" * 4) % ("Class", "Images", "val_loss", "mAP", "F1")
    pred, targets, dt = [], [], (Profile(device=device), Profile(device=device), Profile(device=device))
    accurary = 0
    loss = torch.zeros(1, device=device)
    stats = []
    callbacks.run("on_val_start")
    n = len(dataloader)  # number of batches

    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, labels, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                labels = labels.to(device)
            nb, _, height, width = im.shape  # batch size, channels, height, width

        with dt[1]:
            y = model(im)

        with dt[2]:
            pred.append(torch.sigmoid(y))
            # pred.append(y)
            targets.append(labels)
            if compute_loss:
                loss += compute_loss(y, labels)

        seen += 1

        # predicted_classes: Dict[str, List] = {class_: [] for class_ in names}
        # for class_, pred in y.items():
        #     predicted_classes[class_].append(round(pred.tolist()[0][0]))

        # stats.append((y, labels))

        # out_labels = (y >= conf_thres).type(torch.IntTensor)
        # out_labels = out_labels.to(device)
        # pred.append(out_labels)
        # predictions = out_labels == labels.view(out_labels.shape)
        # accurary += torch.mean(predictions.type(torch.FloatTensor))

    loss /= n  # average loss
    # accurary /= n  # average accurary

    pred, targets = torch.cat(pred), torch.cat(targets)
    # correct = (targets[:, None] == pred).float()
    metrics = compute_metrics(pred.cpu().numpy().copy(), targets.cpu().numpy().copy(), conf_thres, verbose=False)
    meanAP, ACC, ebF1, OF1, CF1, HA = metrics['mAP'], metrics['ACC'], metrics['ebF1'], metrics['OF1'], metrics['CF1'], metrics['HA']
    # correct = (targets[:, None] == pred).float()
    # acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    # top1, top5 = acc.mean(0).tolist()

    # if verbose or training:  # all classes
    # LOGGER.info(f"{'Class':>24}{'Images':>12}{'meanAP':>12}{'ACC':>12}")
    pf = "%22s" + "%11i" * 1 + "%11.3g" * 3  # print format
    LOGGER.info(pf % ('all', targets.shape[0], loss, meanAP, ebF1))
    # for i, c in model.names.items():
    #     acc_i = acc[targets == i]
    #     top1i, top5i = acc_i.mean(0).tolist()
    #     LOGGER.info(f"{c:>24}{acc_i.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")
    if not training:
        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}" % t)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    return (meanAP, ACC, ebF1, OF1, CF1, HA), loss


def parse_opt():
    """Parses and returns command line arguments for YOLOv5 model evaluation and inference settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-cls.pt", help="model.pt path(s)")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--task", default="test", help="train, val, test")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--verbose", nargs="?", const=True, default=True, help="verbose output")
    parser.add_argument("--project", default=ROOT / "runs/val-mlcls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes the YOLOv5 model prediction workflow, handling argument parsing and requirement checks."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
