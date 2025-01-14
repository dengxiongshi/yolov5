import os
import sys

import torch
import yaml

import quantize
import argparse
from pathlib import Path
from utils.general import print_args, check_file, check_dataset, check_yaml, colorstr, check_img_size, init_seeds
from utils.torch_utils import select_device, torch_distributed_zero_first, LOGGER

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))


def run_SensitiveAnalysis(opt, device='cpu'):
    """敏感层分析, 打印影响较大的前10个层"""
    # Hyperparameters
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints
    # prepare model
    print("Preparing Model ...")
    model = quantize.prepare_model(opt.weights, device)
    quantize.replace_to_quantization_model(model)

    # Image size
    opt.gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    opt.imgsz = check_img_size(opt.imgsz, opt.gs, floor=opt.gs * 2)  # verify imgsz is gs-multiple
    # prepare dataset
    print("Preparing Dataset ...")
    init_seeds(opt.seed + 1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']

    train_dataloader = quantize.prepare_dataset(train_path, opt, split="train")
    val_dataloader = quantize.prepare_dataset(val_path, opt, split="val")

    # calibration model
    print("Calibrating ...")
    quantize.calibrate_model(model, train_dataloader, device, 200)
    # sensitive analysis
    print("Sensitive Analysis ...")
    ignored_layer = quantize.sensitive_analysis(model, val_dataloader, opt, opt.sensitive_summary)

    return ignored_layer


def run_PTQ(opt, device='cpu'):
    """除敏感层(ignore_layers)之外的层进行ptq量化"""
    # Hyperparameters
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    # LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints
    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(opt.weights, device)
    quantize.replace_to_quantization_model(model, opt.ignore_layers)
    # Image size
    opt.gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    opt.imgsz = check_img_size(opt.imgsz, opt.gs, floor=opt.gs * 2)  # verify imgsz is gs-multiple
    # prepare dataset
    print("Prepare Dataset ....")
    init_seeds(opt.seed + 1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    train_dataloader = quantize.prepare_dataset(train_path, opt, split="train")
    val_dataloader = quantize.prepare_dataset(val_path, opt, split="val")
    # calibration model
    print("Calibrating ...")
    quantize.calibrate_model(model, train_dataloader, device, 200)

    summary = quantize.SummaryTool(opt.ptq_summary)

    if opt.eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = quantize.evaluate_coco(opt.data, model, val_dataloader, conf_thres=opt.conf_thres,
                                        iou_thres=opt.iou_thres)
            summary.append(["Origin", ap])
    if opt.eval_ptq:
        print("Evaluate PTQ...")
        ap = quantize.evaluate_coco(opt.data, model, val_dataloader, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)
        summary.append(["PTQ", ap])

    if opt.save_ptq:
        print("Export PTQ...")
        quantize.export_ptq(model, opt)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, default=r"D:\python_work\yolov5-7.0\weights\yolov5s.pt",
                        help='initial weights path')
    # parser.add_argument('--cocodir', type=str,  default="../datasets/coco", help="coco directory")
    parser.add_argument('--data', type=str, default=r"D:\python_work\yolov5-7.0\datasets\coco128\coco.yaml",
                        help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch_size', type=int, default=4, help="batch size for data loader")
    parser.add_argument("--imgsz", "--img", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")

    # export
    parser.add_argument("--img-size", nargs="+", type=int, default=[384, 640], help="image (h, w)")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--simplify", type=bool, default=True, help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=13, help="ONNX: opset version")

    parser.add_argument('--sensitive', type=bool, default=True, help="use sensitive analysis or not befor ptq")
    parser.add_argument("--sensitive_summary", type=str, default="sensitive-summary.json", help="summary save file")
    parser.add_argument("--ignore_layers", type=str, default="model\.105\.m\.(.*)", help="regx")

    parser.add_argument("--save_ptq", type=bool, default=True, help="file")
    parser.add_argument("--ptq", type=str, default="ptq_yolov5s.onnx", help="file")

    parser.add_argument("--conf_thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="iou threshold")

    parser.add_argument("--eval_origin", type=bool, default=True, help="do eval for origin model")
    parser.add_argument("--eval_ptq", type=bool, default=True, help="do eval for ptq model")

    parser.add_argument("--ptq_summary", type=str, default="ptq_summary.json", help="summary save file")

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":

    opt = parse_opt()
    print_args(vars(opt))
    # args = argparse.Namespace(**vars(opt))
    # args = parse_opt()
    opt.data, opt.hyp, opt.weights = check_file(opt.data), check_yaml(opt.hyp), str(opt.weights)

    args = argparse.Namespace(**vars(opt))
    # device
    device = select_device(opt.device, batch_size=opt.batch_size)
    # is_cuda = (args.device != "cpu") and torch.cuda.is_available()
    # device = torch.device("cuda:0" if is_cuda else "cpu")
    # 敏感层分析
    if opt.sensitive:
        print("Sensitive Analysis ...")
        ignored_layer = run_SensitiveAnalysis(opt, device)
        args.ignore_layers = list(map(lambda x: x.replace(".", r"\.") + r"\.(.*)", ignored_layer))  # 转换为正则表达式

    # PTQ并导出模型
    print("Running PTQ ...")
    run_PTQ(args, device)
    print("PTQ Quantization done.")