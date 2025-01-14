import torch
import os, argparse

import yaml
from torch import nn

import quantize
from torch.cuda import amp
import torch.optim as optim
from pytorch_quantization import nn as quant_nn
import rules
from typing import Callable
from copy import deepcopy
import tqdm

from models.common import Conv
from models.yolo import Model
from utils.downloads import attempt_download
from utils.general import init_seeds, check_dataset, RANK, check_img_size, ROOT, print_args, FILE, check_file, \
    check_yaml
from utils.torch_utils import select_device, torch_distributed_zero_first, LOCAL_RANK


def run_finetune(args, model, train_loader, val_loader, supervision_policy: Callable = None, fp16=True):
    summary = quantize.SummaryTool("finetune.json")
    # 训练的准备工作
    origin_model = deepcopy(model).eval()
    quantize.disable_quantization(origin_model).apply()

    model.train()
    model.requires_grad_(True)

    scaler = amp.GradScaler(enabled=fp16)  # fp16

    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 优化器

    quant_lossfn = torch.nn.MSELoss()  # 损失函数

    device = next(model.parameters()).device

    lrschedule = {
        0: 1e-6,
        3: 1e-5,
        8: 1e-6
    }

    # hook 函数
    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)

        return forward_hook

    # model & origin_model ==> supervision pairs
    supervision_module_pairs = []
    for (mname, ml), (oriname, ori) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer):
            continue

        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue

        supervision_module_pairs.append([ml, ori])

    # 循环epoch
    best_ap = 0.
    for epoch in range(args.num_epoch):

        # 动态学习率
        if epoch in lrschedule:
            learning_rate = lrschedule[epoch]
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        model_outputs = []
        origin_outputs = []
        remove_handle = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs)))
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

            # 训练
        model.train()
        pbar = tqdm.tqdm(train_loader, desc="QAT", total=args.iters)
        for idx_batch, datas in enumerate(pbar):
            if idx_batch >= args.iters:
                break

            imgs = datas[0].to(device).float() / 255.0

            with amp.autocast(enabled=fp16):
                model(imgs)

                # origin model inference
                with torch.no_grad():
                    origin_model(imgs)

                # 计算量化损失
                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    quant_loss += quant_lossfn(mo, fo)

                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                quant_loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            # print(f"QAT Finetuning {epoch + 1} / {args.num_epoch}, Loss: {quant_loss.detach().item():.5f}, LR: {learning_rate:g}")
            pbar.set_description(
                f"QAT Finetuning {epoch + 1} / {args.num_epoch}, Loss: {quant_loss.detach().item():.5f}, LR: {learning_rate:g}")

        # 移除handle
        for rm in remove_handle:
            rm.remove()

        # 模型验证
        ap = quantize.evaluate_coco(model, val_loader)
        summary.append([f"QAT{epoch}", ap])

        if ap > best_ap:
            print(f"Save qat model to {args.qat} @ {ap:.5f}")
            best_ap = ap
            rules.run_export(model, opt)
            # quantize.export_ptq(model, "qat_yolov7.onnx", device)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, default=r"D:\python_work\yolov5\weights\yolov5s.pt", help='initial weights path')
    parser.add_argument('--data', type=str, default=r"D:\python_work\yolov5-7.0\datasets\coco128\coco.yaml", help='dataset.yaml path')
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

    parser.add_argument('--num_epoch', type=int, default=10, help=' max epoch for finetune')
    parser.add_argument("--iters", type=int, default=200, help="iters per epoch")
    parser.add_argument('--lr', type=float, default=1e-5, help=' learning rate for QAT finetune')

    parser.add_argument("--ignore_layers", type=str, default="model\.24\.m\.(.*)", help="regx")

    parser.add_argument("--save_ptq", type=bool, default=True, help="file")
    parser.add_argument("--ptq", type=str, default="ptq_yolov5.onnx", help="file")

    parser.add_argument("--save_qat", type=bool, default=True, help="file")
    parser.add_argument("--qat", type=str, default="qat_yolov5.onnx", help="file")

    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="NMS IoU threshold")

    parser.add_argument("--eval_origin", type=bool, default=True, help="do eval for origin model")
    parser.add_argument("--eval_ptq", type=bool, default=True, help="do eval for ptq model")
    parser.add_argument("--eval_qat", type=bool, default=True, help="do eval for qat model")

    parser.add_argument("--eval_summary", type=str, default="eval_summary.json",
                        help="all evaluate data are saved in the summary save file")

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":

    opt = parse_opt()
    print_args(vars(opt))
    opt.data, opt.hyp, opt.weights = check_file(opt.data), check_yaml(opt.hyp), str(opt.weights)

    # device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints
    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(opt.weights, device)

    quantize.replace_bottleneck_forward(model)
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

    # 在标定前将scale的工作给做掉
    # rules.apply_custom_rules_to_quantizer(model, device)
    rules.apply_custom_rules_to_quantizer(model, opt)

    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)

    summary = quantize.SummaryTool(opt.eval_summary)

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
        # quantize.export_ptq(model, args.ptq, device)
        rules.run_export(model, opt)


    # 判断传入的模块是否需要在QAT训练期间计算损失
    def supervision_policy():
        supervision_list = []
        for item in model.model:
            supervision_list.append(id(item))

        supervision_stride = 1
        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2)

        def impl(name, module):
            if id(module) not in supervision_list:
                return False

            idx = supervision_list.index(id(module))
            if idx in keep_idx:
                print(f"Supervision: {name} will compute loss with origin model during QAT training...")
            else:
                print(f"Supervision: {name} not compute loss during QAT training...")

            return idx in keep_idx  # True/False

        return impl


    print("Begining Finetune ....")

    run_finetune(opt, model, train_dataloader, val_dataloader, supervision_policy=supervision_policy())

    print("QAT Finished ....")