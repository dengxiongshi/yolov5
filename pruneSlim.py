import yaml
from models.yolo_prune import *

from utils.general import set_logging, check_file, intersect_dicts
from utils.torch_utils import select_device, de_parallel
from utils.prune_utils import *
from utils.adaptive_bn import *


def prune_and_eval(model, ignore_idx, opt):
    bn_weights = gather_bn_weights(model, ignore_idx)

    sorted_bn, _ = torch.sort(bn_weights)
    thresh_index = int(len(bn_weights) * opt.global_percent)  # bn channel nums to leave
    thresh = sorted_bn[thresh_index].cuda()

    print(f'bn |gamma| will be more than {thresh:.4f}.')

    # get conv and bn mask
    maskbndict = {}
    maskconvdict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if name in ignore_idx:
                mask = torch.ones(module.weight.data.shape)
            else:
                mask = obtain_filtermask_bn(module, thresh)

            maskbndict[name] = mask
            maskconvdict[name[:-2] + 'conv'] = mask

    with open(opt.cfg) as f:
        oriyaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
    ignore_conv_idx = [i.replace('bn', 'conv') for i in ignore_idx]
    pruned_yaml = update_yaml(oriyaml, model, ignore_conv_idx, maskconvdict, opt)

    compact_model = Model(pruned_yaml, pruning=False).to(device)

    weights_inheritance(model, compact_model, from_to_map, maskbndict)
    ABE = AdaptiveBNEval(model, opt, device, hyp)
    mAP = ABE(compact_model)

    with open(opt.path, "w", encoding='utf-8') as f:
        yaml.safe_dump(pruned_yaml, f, encoding='utf-8', allow_unicode=True, default_flow_style=True, sort_keys=False)
        # yaml.dump(pruned_yaml, f, Dumper=ruamel.yaml.RoundTripDumper)
    # with open(opt.path[:-5]+'_.yaml', "w", encoding='utf-8') as fd:
    #     yaml.safe_dump(pruned_yaml,fd,encoding='utf-8', allow_unicode=True, sort_keys=False)
    ckpt = {'epoch': -1,
            'best_fitness': [mAP],
            'model': deepcopy(de_parallel(compact_model)).half(),
            'ema': None,
            'updates': None,
            'optimizer': None,
            'opt': None,
            'git': None,  # {remote, branch, commit} if a git repo
            'date': None}
    torch.save(ckpt, opt.weights[:-3] + '-Slimpruned.pt')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="runs/train/slim/yolov5s_512/weights/best.pt", help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/pruneModels/yolov5s_512.yaml', help='model.yaml')
    parser.add_argument('--data', type=str, default='datasets/coco128/coco.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--path', type=str, default='models/pruneModels/yolov5s_512_pruned.yaml',
                        help='the path to save pruned yaml')

    parser.add_argument('--global_percent', type=float, default=0.6, help='global channel prune percent')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt()

    print_args(vars(opt))

    opt.data, opt.cfg, opt.hyp, opt.weights = check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights)
    set_logging()
    device = select_device(opt.device)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Create model
    model = Model(opt.cfg).to(device)
    ckpt = torch.load(opt.weights, map_location=device)
    exclude = []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=True)  # load strictly

    # Parse Module
    CBL_idx, ignore_idx, from_to_map = parse_module_defs(model.yaml)
    prune_and_eval(model, ignore_idx, opt)