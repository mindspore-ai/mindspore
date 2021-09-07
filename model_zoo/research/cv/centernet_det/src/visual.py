# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Data operations, will be used in train.py
"""

import os
import json
import random
import cv2
import numpy as np
import pycocotools.coco as COCO
from .model_utils.config import eval_config as eval_cfg
from .image import get_affine_transform, affine_transform


coco_class_name2id = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,
                      'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
                      'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
                      'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21,
                      'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
                      'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34,
                      'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39,
                      'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43,
                      'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50,
                      'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56,
                      'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62,
                      'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70,
                      'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77,
                      'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82,
                      'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88,
                      'hair drier': 89, 'toothbrush': 90}


def coco_box_to_bbox(box):
    """convert height/width to position coordinates"""
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
    return bbox


def resize_image(image, anns, width, height):
    """resize image to specified scale"""
    h, w = image.shape[0], image.shape[1]
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0
    trans_output = get_affine_transform(c, s, 0, [width, height])
    out_img = cv2.warpAffine(image, trans_output, (width, height), flags=cv2.INTER_LINEAR)

    num_objects = len(anns)
    resize_anno = []
    for i in range(num_objects):
        ann = anns[i]
        bbox = coco_box_to_bbox(ann['bbox'])
        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[0::2] = np.clip(bbox[0::2], 0, width - 1)
        bbox[1::2] = np.clip(bbox[1::2], 0, height - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if (h > 0 and w > 0):
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            bbox = [ct[0] - w / 2, ct[1] - h / 2, w, h, 1]
            ann["bbox"] = bbox
            gt = ann
            resize_anno.append(gt)
    return out_img, resize_anno


def merge_pred(ann_path, mode="val", name="merged_annotations"):
    """merge annotation info of each image together"""
    files = os.listdir(ann_path)
    data_files = []
    for file_name in files:
        if "json" in file_name:
            data_files.append(os.path.join(ann_path, file_name))
    pred = {"images": [], "annotations": []}
    for file in data_files:
        anno = json.load(open(file, 'r'))
        if "images" in anno:
            for img in anno["images"]:
                pred["images"].append(img)
        if "annotations" in anno:
            for ann in anno["annotations"]:
                pred["annotations"].append(ann)
    json.dump(pred, open('{}/{}_{}.json'.format(ann_path, name, mode), 'w'))


def visual(ann_path, image_path, save_path, ratio=1, mode="val", name="merged_annotations"):
    """visulize all images based on dataset and annotations info"""
    merge_pred(ann_path, mode, name)
    ann_path = os.path.join(ann_path, name + '_' + mode + '.json')
    visual_allimages(ann_path, image_path, save_path, ratio)


def visual_allimages(anno_file, image_path, save_path, ratio=1):
    """visualize all images and annotations info"""
    coco = COCO.COCO(anno_file)
    image_ids = coco.getImgIds()
    images = []
    anns = {}
    for img_id in image_ids:
        idxs = coco.getAnnIds(imgIds=[img_id])
        if idxs:
            images.append(img_id)
            anns[img_id] = idxs

    for img_id in images:
        file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(image_path, file_name)
        annos = coco.loadAnns(anns[img_id])
        img = cv2.imread(img_path)
        return visual_image(img, annos, save_path, ratio)


def visual_image(img, annos, save_path, ratio=None, height=None, width=None, name=None, score_threshold=0.01):
    """visualize image and annotations info"""
    h, w = img.shape[0], img.shape[1]
    if height is not None and width is not None and (height != h or width != w):
        img, annos = resize_image(img, annos, width, height)
    elif ratio not in (None, 1):
        img, annos = resize_image(img, annos, w * ratio, h * ratio)

    color_list = np.array(eval_cfg.color_list).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
    colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 3)

    h, w = img.shape[0], img.shape[1]
    num_objects = len(annos)
    name_list = []
    id_list = []
    for class_name, class_id in coco_class_name2id.items():
        name_list.append(class_name)
        id_list.append(class_id)

    for i in range(num_objects):
        ann = annos[i]
        bbox = coco_box_to_bbox(ann['bbox'])
        cat_id = ann['category_id']
        if cat_id in id_list:
            get_id = id_list.index(cat_id)
            name = name_list[get_id]
            c = colors[get_id].tolist()
        if "score" in ann:
            score = ann["score"]
            if score < score_threshold:
                continue
            txt = '{}{:.2f}'.format(name, ann["score"])
            cat_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (bbox[0], int(bbox[1] - cat_size[1] - 5)),
                          (int(bbox[0] + cat_size[0]), int(bbox[1] - 2)), c, -1)
            cv2.putText(img, txt, (bbox[0], int(bbox[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        ct = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        cv2.circle(img, ct, 2, c, thickness=-1, lineType=cv2.FILLED)
        bbox = np.array(bbox, dtype=np.int32).tolist()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)

    if annos and "image_id" in annos[0]:
        img_id = annos[0]["image_id"]
    else:
        img_id = random.randint(0, 9999999)
    image_name = "cv_image_" + str(img_id) + ".png"
    cv2.imwrite("{}/{}".format(save_path, image_name), img)
