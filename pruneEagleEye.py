import argparse
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import yaml
from yaml.events import NodeEvent
# import ruamel.yaml
# from ruamel import yaml

from models.yolo_prune import *
from utils.general import set_logging, check_git_info, check_file, intersect_dicts
from utils.prune_utils import *
from utils.adaptive_bn import *
from utils.torch_utils import de_parallel

GIT_INFO = check_git_info()


# obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])


def rand_prune_and_eval(model, ignore_idx, opt):
    # np.random.seed(123)
    # origin_nparameters = obtain_num_parameters(model)
    origin_flops = model.flops
    ignore_conv_idx = [i.replace('bn', 'conv') for i in ignore_idx]
    max_remain_ratio = 1.0
    candidates = 0
    max_mAP = 0
    maskbndict = {}
    maskconvdict = {}
    with open(opt.cfg) as f:
        oriyaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

    # with open(opt.cfg, 'r', encoding='utf-8') as f:
    #     data = f.read()
    #     oriyaml = yaml.load(data, Loader=ruamel.yaml.RoundTripLoader)  # model dict

    ABE = AdaptiveBNEval(model, opt, device, hyp)

    while True:
        pruned_yaml = deepcopy(oriyaml)

        # obtain mask
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name in ignore_conv_idx:
                    mask = torch.ones(module.weight.data.size()[0]).to(device)  # [N, C, H, W]
                else:
                    rand_remain_ratio = (max_remain_ratio - opt.min_remain_ratio) * (
                        np.random.rand(1)) + opt.min_remain_ratio
                    # rand_remain_ratio = 1.0
                    mask = obtain_filtermask_l1(module, rand_remain_ratio).to(device)
                # name: model.0.conv
                # module: Conv2d(3, 16, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
                maskbndict[(name[:-4] + 'bn')] = mask
                maskconvdict[name] = mask

        pruned_yaml = update_yaml(pruned_yaml, model, ignore_conv_idx, maskconvdict, opt)

        compact_model = Model(pruned_yaml, pruning=False).to(device)
        current_flops = compact_model.flops
        if (current_flops / origin_flops > opt.remain_ratio + opt.delta) or (
                current_flops / origin_flops < opt.remain_ratio - opt.delta):
            del compact_model
            del pruned_yaml
            continue
        weights_inheritance(model, compact_model, from_to_map, maskbndict)
        mAP = ABE(compact_model)
        print('mAP@0.5 of candidate sub-network is {:f}'.format(mAP))

        if mAP > max_mAP:
            max_mAP = mAP
            with open(opt.path, "w", encoding='utf-8') as f:
                yaml.safe_dump(pruned_yaml, f, encoding='utf-8', allow_unicode=True, default_flow_style=True,
                               sort_keys=False)
                # yaml.dump(pruned_yaml, f, Dumper=ruamel.yaml.RoundTripDumper)
            # with open(opt.path[:-5]+'_.yaml', "w", encoding='utf-8') as fd:
            #     yaml.safe_dump(pruned_yaml,fd,encoding='utf-8', allow_unicode=True, sort_keys=False)
            ckpt = {'epoch': -1,
                    'best_fitness': [max_mAP],
                    'model': deepcopy(de_parallel(compact_model)).half(),
                    'ema': None,
                    'updates': None,
                    'optimizer': None,
                    'opt': None,
                    'git': None,  # {remote, branch, commit} if a git repo
                    'date': None}
            torch.save(ckpt, opt.weights[:-3] + '-EagleEyepruned.pt')

        candidates = candidates + 1
        del compact_model
        del pruned_yaml
        if candidates > opt.max_iter:
            break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="runs/train/prune/DWContrans2d_FRM/weights/best.pt",
                        help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/pruneModels/DWContrans2d_FRM.yaml', help='model.yaml')
    parser.add_argument('--data', type=str, default='datasets/coco128/coco.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--path', type=str, default='models/pruneModels/DWContrans2d_FRM_pruned.yaml',
                        help='the path to save pruned yaml')
    parser.add_argument('--min_remain_ratio', type=float, default=0.2)
    parser.add_argument('--max_iter', type=int, default=700, help='maximum number of arch search')
    parser.add_argument('--remain_ratio', type=float, default=0.5, help='the whole parameters/FLOPs remain ratio')
    parser.add_argument('--delta', type=float, default=0.02, help='scale of arch search')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt()

    print_args(vars(opt))

    opt.data, opt.cfg, opt.hyp, opt.weights = check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(
        opt.weights)

    set_logging()
    device = select_device(opt.device, batch_size=opt.batch_size)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    data_dict = None
    data_dict = data_dict or check_dataset(opt.data)  # check if None

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes

    ckpt = torch.load(opt.weights, map_location=device)
    model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load  # load strictly

    # Parse Module
    CBL_idx, ignore_idx, from_to_map = parse_module_defs(model.yaml)
    rand_prune_and_eval(model, ignore_idx, opt)


