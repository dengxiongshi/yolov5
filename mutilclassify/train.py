# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset.

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 2022 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3

Datasets:           --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof, imagenet, or 'path/to/data'
YOLOv5-cls models:  --model yolov5n-cls.pt, yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt, yolov5x-cls.pt
Torchvision models: --model resnet50, efficientnet_b0, etc. See https://pytorch.org/vision/stable/models.html
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from mutilclassify import val as validate
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel, MultiLabelClassificationModel
from utils.dataloaders import create_classification_dataloader, create_multilabel_classification_dataloader
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    WorkingDirectory,
    check_git_info,
    check_git_status,
    check_requirements,
    colorstr,
    download,
    increment_path,
    init_seeds,
    print_args,
    yaml_save, check_dataset, check_img_size, labels_to_class_weights, check_file, check_yaml, strip_optimizer,
)
from utils.loggers import GenericLogger
from utils.plots import imshow_cls
from utils.torch_utils import (
    ModelEMA,
    de_parallel,
    model_info,
    reshape_classifier_output,
    select_device,
    smart_DDP,
    smart_optimizer,
    smartCrossEntropyLoss,
    torch_distributed_zero_first, EarlyStopping, smartBCEWithLogitsLoss, smartBCELoss, MultiLabelBCELoss,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()


# def criterion(loss_fn, preds, gts, device):
#     losses = torch.zeros((len(preds.keys())), device=device)
#     for i, key in enumerate(preds):
#         losses[i] = loss_fn(preds[key], torch.unsqueeze(gts[key], 1).float().to(device))
#     return torch.mean(losses)


def train(hyp, opt, device, callbacks):
    """Trains a YOLOv5 model, managing datasets, model optimization, logging, and saving checkpoints."""
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, data, batch_size, epochs, workers, weights, noval, imgsz, pretrained = (
        opt.save_dir,
        Path(opt.data),
        opt.batch_size,
        opt.epochs,
        opt.workers,
        opt.weights,
        opt.noval,
        opt.imgsz,
        str(opt.pretrained).lower() == "true",
    )
    # callbacks.run("on_pretrain_routine_start")

    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / "last.pt", wdir / "best.pt"

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    # LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    # yaml_save(save_dir / "hyp.yaml", hyp)
    yaml_save(save_dir / "opt.yaml", vars(opt))

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    data_dict = None
    # if RANK in {-1, 0}:
    #     include_loggers = list(LOGGERS)
    #     if getattr(opt, "ndjson_console", False):
    #         include_loggers.append("ndjson_console")
    #     if getattr(opt, "ndjson_file", False):
    #         include_loggers.append("ndjson_file")
    #
    #     loggers = Loggers(
    #         save_dir=save_dir,
    #         weights=weights,
    #         opt=opt,
    #         hyp=hyp,
    #         logger=LOGGER,
    #         include=tuple(include_loggers),
    #     )
    #
    #     # Register actions
    #     for k in methods(loggers):
    #         callbacks.register_action(k, callback=getattr(loggers, k))
    #
    #     # Process custom dataset artifact link
    #     data_dict = loggers.remote_dataset

    # Config
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = int(data_dict["nc"])  # number of classes
    names = data_dict["names"]  # class names

    assert len(names) == nc, f"Label class {len(names)} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Model
    # check_suffix(weights, ".pt")  # check weights
    # pretrained = weights.endswith(".pt")
    # if pretrained:
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        if opt.cfg is not None:
            model = MultiLabelClassificationModel(cfg=opt.cfg, nc=nc)
        else:
            if Path(opt.weights).is_file() or opt.weights.endswith(".pt"):
                model = attempt_load(opt.weights, device="cpu", fuse=False)
            elif opt.weights in torchvision.models.__dict__:  # TorchVision models i.e. resnet50, efficientnet_b0
                model = torchvision.models.__dict__[opt.model](weights="IMAGENET1K_V1" if pretrained else None)
            else:
                m = hub.list("ultralytics/yolov5")  # + hub.list('pytorch/vision')  # models
                raise ModuleNotFoundError(f"--model {opt.model} not found. Available models are: \n" + "\n".join(m))
            if isinstance(model, DetectionModel):
                LOGGER.warning("WARNING ⚠️ pass YOLOv5 classifier model with '-cls' suffix, i.e. '--model yolov5s-cls.pt'")
                model = MultiLabelClassificationModel(model=model, nc=nc, cutoff=opt.cutoff or 10)  # convert to classification model
            reshape_classifier_output(model, nc)  # update class count
    for m in model.modules():
        if not pretrained and hasattr(m, "reset_parameters"):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # set dropout
    for p in model.parameters():
        p.requires_grad = True  # for training
    model = model.to(device)

    # amp = check_amp(model)  # check AMP

    # Image size
    # gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        # loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    # optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)

    # Scheduler
    # if opt.cos_lr:
    #     lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    # else:
    # lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    lrf = 0.01  # final lr (fraction of lr0)
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Trainloader
    train_loader, dataset = create_multilabel_classification_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        nc,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rank=LOCAL_RANK,
        workers=workers,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_multilabel_classification_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            nc,
            cache=None if noval else opt.cache,
            rank=-1,
            workers=workers * 2,
            prefix=colorstr("val: "),
        )[0]

        # callbacks.run("on_pretrain_routine_end", labels, names)  #modify

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    # nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    # hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    # hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    # model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Info
    # if RANK in {-1, 0}:
    #     images, labels, _, _ = next(iter(train_loader))
    #     file = imshow_cls(images[:25], labels[:25], names=model.names, f=save_dir / "train_images.jpg")
    #     logger.log_images(file, name="Train Examples")
    #     logger.log_graph(model, imgsz)  # log model

    # Train
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    best_fitness, start_epoch = 0.0, 0
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = smartBCEWithLogitsLoss(label_smoothing=opt.label_smoothing)  # loss function
    callbacks.run("on_train_start")
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} test\n'
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting {opt.weights} training on {data} dataset with {nc} classes for {epochs} epochs...\n\n'
    )
    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        mloss = torch.zeros(1, device=device)  # mean losses
        losses = []
        model.train()
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 5) % ("Epoch", "GPU_mem", "Instances", "train_loss", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)

        optimizer.zero_grad()  # 将上一次迭代的梯度值置零
        for i, (imgs, targets, paths, _) in pbar:  # progress bar
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True)

            # Warmup
            # if ni <= nw:
            #     xi = [0, nw]  # x interp
            #     # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            #     accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
            #     for j, x in enumerate(optimizer.param_groups):
            #         # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            #         x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
            #         if "momentum" in x:
            #             x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Forward
            with amp.autocast(enabled=cuda):  # stability issues when enabled
                pred = model(imgs)  # forward
                loss = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

                # 计算阈值为0.5时的预测结果向量
                # preds = torch.round(pred)
                # x_test = compute_loss(x, targets.to(device))
            # loss_items = torch.tensor(losses).mean().item()
            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            # if ni - last_opt_step >= accumulate:
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
                # last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 3)
                    % (f"{epoch}/{epochs - 1}", mem, targets.shape[0], tloss, imgs.shape[-1])
                )
                # callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        # lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # Log metrics
        if RANK in {-1, 0}:
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, vloss = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=ema.ema,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        plots=False,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # test accuracy, loss
            # Update best mAP
            fitness = np.nanmean(np.array(results))
            stop = stopper(epoch=epoch, fitness=fitness)
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            metrics = {
                "train/loss": tloss,
                "val/loss": vloss,
                "metrics/mAP": results[0],
                "metrics/accuracy": results[1],
                "metrics/F1": results[2],
                "lr/0": optimizer.param_groups[0]["lr"],
            }  # learning rate
            logger.log_metrics(metrics, epoch)

            # Save model
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(),
                    "ema": None,  # deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fitness)

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    # Train complete
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        dataloader=val_loader,
                        save_dir=save_dir,
                        verbose=True,
                        plots=True,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots

        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """Parses command line arguments for YOLOv5 training including model path, dataset, epochs, and more, returning
    parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/yolov5s-cls.pt", help="initial weights path")
    parser.add_argument("--data", type=str, default=r"datasets/coco128/coco.yaml", help="dataset.yaml path")
    parser.add_argument('--cfg', type=str, default='models/classify/yolov5s-cls.yaml', help='build model by yaml')
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=600, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument("--project", default="runs/train-mlcls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", nargs="?", const=True, default=True, help="start from i.e. --pretrained False")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="SGD", help="optimizer")
    parser.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--decay", type=float, default=5e-5, help="weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--cutoff", type=int, default=None, help="Model layer cutoff index for Classify() head")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout (fraction)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--patience", type=int, default=0, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """Executes YOLOv5 training with given options, handling device setup and DDP mode; includes pre-training checks."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    opt.data, opt.hyp, opt.weights = (
        check_file(opt.data),
        check_yaml(opt.hyp),
        str(opt.weights),
    )  # checks
    assert len(opt.weights), "either --weights must be specified"

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, "AutoBatch is coming soon for classification, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt.hyp, opt, device, callbacks)


def run(**kwargs):
    """
    Executes YOLOv5 model training or inference with specified parameters, returning updated options.

    Example: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
