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
"""post process for 310 inference"""
import os
import argparse
import numpy as np
from PIL import Image

from src.config import config
from src.eval_utils import metrics

batch_size = 1
parser = argparse.ArgumentParser(description="ssd acc calculation")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--img_path", type=str, required=True, help="image file path.")
parser.add_argument("--drop", action="store_true", help="drop iscrowd images or not.")
args = parser.parse_args()

def get_imgSize(file_name):
    img = Image.open(file_name)
    return img.size

def get_result(result_path, img_id_file_path):
    anno_json = os.path.join(config.coco_root, config.instances_set.format(config.val_data_type))

    if args.drop:
        from pycocotools.coco import COCO
        train_cls = config.classes
        train_cls_dict = {}
        for i, cls in enumerate(train_cls):
            train_cls_dict[cls] = i
        coco = COCO(anno_json)
        classs_dict = {}
        cat_ids = coco.loadCats(coco.getCatIds())
        for cat in cat_ids:
            classs_dict[cat["id"]] = cat["name"]

    files = os.listdir(img_id_file_path)
    pred_data = []

    for file in files:
        img_ids_name = file.split('.')[0]
        img_id = int(np.squeeze(img_ids_name))
        if args.drop:
            anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = coco.loadAnns(anno_ids)
            annos = []
            iscrowd = False
            for label in anno:
                bbox = label["bbox"]
                class_name = classs_dict[label["category_id"]]
                iscrowd = iscrowd or label["iscrowd"]
                if class_name in train_cls:
                    x_min, x_max = bbox[0], bbox[0] + bbox[2]
                    y_min, y_max = bbox[1], bbox[1] + bbox[3]
                    annos.append(list(map(round, [y_min, x_min, y_max, x_max])) + [train_cls_dict[class_name]])
            if iscrowd or (not annos):
                continue

        img_size = get_imgSize(os.path.join(img_id_file_path, file))
        image_shape = np.array([img_size[1], img_size[0]])
        result_path_0 = os.path.join(result_path, img_ids_name + "_0.bin")
        result_path_1 = os.path.join(result_path, img_ids_name + "_1.bin")
        boxes = np.fromfile(result_path_0, dtype=np.float32).reshape(config.num_ssd_boxes, 4)
        box_scores = np.fromfile(result_path_1, dtype=np.float32).reshape(config.num_ssd_boxes, config.num_classes)

        pred_data.append({
            "boxes": boxes,
            "box_scores": box_scores,
            "img_id": img_id,
            "image_shape": image_shape
        })
    mAP = metrics(pred_data, anno_json)
    print(f" mAP:{mAP}")

if __name__ == '__main__':
    get_result(args.result_path, args.img_path)
