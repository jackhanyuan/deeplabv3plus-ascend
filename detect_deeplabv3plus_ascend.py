#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# by [jackhanyuan](https://github.com/jackhanyuan) 07/03/2022

import argparse
import copy
import glob
import os
import re
import sys
import time
from pathlib import Path

import cv2
import acl
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from acl_net import check_ret, Net


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
IMG_EXT = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def preprocess_input(image):
    image /= 255.0
    return image


def detect_image(image, num_classes, input_shape=(512, 512), fp16=False):
    image = cvtColor(image)
    image_h, image_w = image.size
    org_img = copy.deepcopy(image)

    dtype = np.float16 if fp16 else np.float32

    img_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
    img_data = np.expand_dims(np.transpose(preprocess_input(np.array(img_data, dtype)), (2, 0, 1)), 0)
    if fp16:
        img_data = img_data.astype("float16")
    img_bytes = np.frombuffer(img_data.tobytes(), dtype)

    result = net.run([img_bytes])[0]

    pred = np.frombuffer(bytearray(result), dtype)
    pred = pred.reshape(num_classes, input_shape[0], input_shape[1])
    pred = torch.from_numpy(pred)
    pred = F.softmax(pred.float().permute(1, 2, 0), dim=-1).numpy()
    pred = pred[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh), \
           int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
    pred = cv2.resize(pred, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
    pred_img = pred.argmax(axis=-1)

    return org_img, pred_img


def draw_image(org_img, pred, num_classes, blend=True):

    colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
             (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
             (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
             (128, 64, 12)]

    seg_img = np.zeros((np.shape(pred)[0], np.shape(pred)[1], 3))
    for c in range(num_classes):
        seg_img[:, :, 0] += ((pred[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pred[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pred[:, :] == c) * (colors[c][2])).astype('uint8')
    image = Image.fromarray(np.uint8(seg_img))

    if blend:
        image = Image.blend(org_img, image, 0.7)

    return image


def load_label(label_name):
    label_lookup_path = label_name
    with open(label_lookup_path, 'r') as f:
        label_contents = f.readlines()

    labels = np.array(list(map(lambda x: x.strip(), label_contents)))
    return labels
    

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'ascend/deeplab_mobilenetv2.om',
                        help='str: weights path.')
    parser.add_argument('--labels', nargs='+', type=str, default=ROOT / 'ascend/deeplabv3plus.label')
    parser.add_argument('--imgsz', nargs='+', type=int, default=(512, 512),
                        help='int tuple: the model inference size (w, h).')
    parser.add_argument('--images-dir', type=str, default=ROOT / 'img')
    parser.add_argument('--output-dir', type=str, default=ROOT / 'img_out')
    parser.add_argument('--device', type=int, default=0, help='int: npu device id, i.e. 0 or 1.')
    parser.add_argument('--save-img', action='store_true', default=True,
                        help='bool: whether to save image, default=True.')
    parser.add_argument('--blend', action='store_true', default=True,
                        help='bool: whether to mix the original image and the predicted image.')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    t0 = time.perf_counter()
    print("ACL Init:")
    ret = acl.init()
    check_ret("acl.init", ret)
    device_id = opt.device

    # 1.Load model
    print("Loading model %s." % opt.weights)
    model_path = str(opt.weights)
    net = Net(device_id, model_path)

    input_size = opt.imgsz
    output_dir = increment_path(Path(opt.output_dir) / 'exp', exist_ok=False)  # increment path
    output_dir.mkdir(parents=True, exist_ok=True)  # make dir
    
    # 2.Load label
    label_path = opt.labels
    labels = load_label(label_path)
    num_classes = len(labels)
    

    # 3.Start Detect
    print()
    print("Start Detect:")

    images_dir = opt.images_dir
    images = sorted(os.listdir(images_dir))

    count = 0
    total_count = len(images)
    for image_name in images:
        if image_name.lower().endswith(IMG_EXT):
            t1 = time.perf_counter()
            count += 1

            image_path = os.path.join(images_dir, image_name)
            image = Image.open(image_path)

            # detect image
            org_img, pred_img = detect_image(image, num_classes=num_classes, input_shape=input_size, fp16=False)

            # count area for every labels
            s = ""
            for i in range(len(labels)):
                count_area = int(np.sum(pred_img == i))
                if count_area > 0:
                    s += f"{count_area} pixel{'s' * (count_area > 1)} {labels[i]}, "  # add to string
                
            # draw imgage
            output_img = draw_image(org_img, pred_img, num_classes=num_classes, blend=opt.blend)

            # save image
            if opt.save_img:
               output_path = os.path.join(output_dir, image_name)
               output_img.save(output_path)

            t2 = time.perf_counter()
            t = t2 - t1
            print('image {}/{} {}: {}Done. ({:.3f}s)'.format(count, total_count, image_path, s, t))

    t3 = time.perf_counter()
    t = t3 - t0
    print('This detection cost {:.3f}s.'.format(t))
    print("Results saved to {}.".format(output_dir))
    print()