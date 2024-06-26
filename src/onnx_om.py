import torch.nn as nn
import onnxruntime as ort
import numpy as np
import cv2
import time
import torch
import torchvision
from torchvision.ops import box_iou



def onnx_inference(model_path_, input_data_):
    if len(input_data_.shape) == 3:
        C, H, W = input_data_.shape
        input_data_ = input_data_.reshape(1, C, H, W)
    sess = ort.InferenceSession(model_path_, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    for idx in range(len(sess.get_inputs())):
        print((sess.get_inputs()[idx].name, sess.get_inputs()[idx].shape))
    for idx in range(len(sess.get_outputs())):
        print((sess.get_outputs()[idx].name, sess.get_outputs()[idx].shape))
    output_data_ = sess.run([out_name], {input_name: input_data_})[0]
    return output_data_


def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


if __name__=='__main__':
    onnx_file = r"D:\python_work\yolov5\weights\FireSmoke_20240118.onnx"

    img_file = r"C:\Users\dengxs\Desktop\userdata\snap\2024_01_24\2024_01_24_17_40_55_0001.jpg"

    input_pic = cv2.imread(img_file)
    input_pic = cv2.resize(input_pic, (640, 384))
    input_pic2 = input_pic.copy()

    cv2.namedWindow('input', cv2.WINDOW_NORMAL)
    cv2.imshow('input', input_pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #################################
    input_data = (input_pic / 255.0).astype(np.float32).transpose((2, 0, 1))[::-1]
    onnx_output = onnx_inference(onnx_file, input_data).reshape(1, 15120, 7).astype(np.float32)
    nms_out = non_max_suppression(torch.from_numpy(onnx_output), 0.25, 0.45)

    boxs = []
    for res in nms_out:
        for item in res:
            boxs.append([int(item[0]), int(item[1]), int(item[2]), int(item[3]), 2, int(item[5])])
            print(item[0], item[1], item[2], item[3], item[4], item[5])

    for box in boxs:
        if box[5] == 0:
            cv2.rectangle(input_pic, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 1)
        elif box[5] == 1:
            cv2.rectangle(input_pic, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 1)
        else:
            cv2.rectangle(input_pic, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)

    cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
    cv2.imshow('pic', input_pic)
    cv2.waitKey()
    cv2.destroyAllWindows()

    ##############################################################################
    bin_file = r"C:\Users\dengxs\Desktop\userdata\snap\2024_01_24\2024_01_24_17_40_55_0000_RGGB_1520x2688_U16_RAW_d001.bin"
    om_output = np.fromfile(bin_file, dtype=np.float32).reshape(1, 15120, 7).astype(np.float32)

    nms_out = non_max_suppression(torch.from_numpy(om_output), 0.25, 0.45)

    boxs = []
    for res in nms_out:
        for item in res:
            boxs.append([int(item[0]), int(item[1]), int(item[2]), int(item[3]), 2, int(item[5])])
            print(item[0], item[1], item[2], item[3], item[4], item[5])

    for box in boxs:
        if box[5] == 0:
            cv2.rectangle(input_pic2, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 1)
        elif box[5] == 1:
            cv2.rectangle(input_pic2, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 1)
        else:
            cv2.rectangle(input_pic2, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)

    cv2.imshow('pic', input_pic2)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print('finish')
