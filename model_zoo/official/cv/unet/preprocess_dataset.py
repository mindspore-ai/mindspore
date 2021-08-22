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
Preprocess dataset.
Images within one folder is an image, the image file named `"image.png"`, the mask file named `"mask.png"`.
"""
import os
import cv2
import numpy as np
from src.model_utils.config import config

def annToMask(ann, height, width):
    """Convert annotation to RLE and then to binary mask."""
    from pycocotools import mask as maskHelper
    segm = ann['segmentation']
    if isinstance(segm, list):
        rles = maskHelper.frPyObjects(segm, height, width)
        rle = maskHelper.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = maskHelper.frPyObjects(segm, height, width)
    else:
        rle = ann['segmentation']
    m = maskHelper.decode(rle)
    return m

def preprocess_cell_nuclei_dataset(param_dict):
    """
    Preprocess for Cell Nuclei dataset.
    merge all instances to a mask, and save the mask at data_dir/img_id/mask.png.
    """
    print("========== start preprocess Cell Nuclei dataset ==========")
    data_dir = param_dict["data_dir"]
    img_ids = sorted(next(os.walk(data_dir))[1])
    for img_id in img_ids:
        path = os.path.join(data_dir, img_id)
        if (not os.path.exists(os.path.join(path, "image.png"))) or \
                (not os.path.exists(os.path.join(path, "mask.png"))):
            img = cv2.imread(os.path.join(path, "images", img_id + ".png"))
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
                img = np.concatenate([img, img, img], axis=-1)
            mask = []
            for mask_file in next(os.walk(os.path.join(path, "masks")))[2]:
                mask_ = cv2.imread(os.path.join(path, "masks", mask_file), cv2.IMREAD_GRAYSCALE)
                mask.append(mask_)
            mask = np.max(mask, axis=0)
            cv2.imwrite(os.path.join(path, "image.png"), img)
            cv2.imwrite(os.path.join(path, "mask.png"), mask)

def preprocess_coco_dataset(param_dict):
    """
    Preprocess for coco dataset.
    Save image and mask at save_dir/img_name/image.png save_dir/img_name/mask.png
    """
    print("========== start preprocess coco dataset ==========")
    from pycocotools.coco import COCO
    anno_json = param_dict["anno_json"]
    coco_cls = param_dict["coco_classes"]
    coco_dir = param_dict["coco_dir"]
    save_dir = param_dict["save_dir"]
    coco_cls_dict = {}
    for i, cls in enumerate(coco_cls):
        coco_cls_dict[cls] = i
    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]
    image_ids = coco.getImgIds()
    images_num = len(image_ids)
    for ind, img_id in enumerate(image_ids):
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        img_name, _ = os.path.splitext(file_name)
        image_path = os.path.join(coco_dir, file_name)
        if not os.path.isfile(image_path):
            print("{}/{}: {} is in annotations but not exist".format(ind + 1, images_num, image_path))
            continue
        if not os.path.exists(os.path.join(save_dir, img_name)):
            os.makedirs(os.path.join(save_dir, img_name))
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        h = coco.imgs[img_id]["height"]
        w = coco.imgs[img_id]["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for instance in anno:
            m = annToMask(instance, h, w)
            c = coco_cls_dict[classs_dict[instance["category_id"]]]
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        img = cv2.imread(image_path)
        cv2.imwrite(os.path.join(save_dir, img_name, "image.png"), img)
        cv2.imwrite(os.path.join(save_dir, img_name, "mask.png"), mask)

def preprocess_dataset(data_dir):
    """Select preprocess function."""
    if config.dataset.lower() == "cell_nuclei":
        preprocess_cell_nuclei_dataset({"data_dir": data_dir})
    elif config.dataset.lower() == "coco":
        if config.split == 1.0:
            train_data_path = os.path.join(data_dir, "train")
            val_data_path = os.path.join(data_dir, "val")
            train_param_dict = {"anno_json": config.anno_json, "coco_classes": config.coco_classes,
                                "coco_dir": config.coco_dir, "save_dir": train_data_path}
            preprocess_coco_dataset(train_param_dict)
            val_param_dict = {"anno_json": config.val_anno_json, "coco_classes": config.coco_classes,
                              "coco_dir": config.val_coco_dir, "save_dir": val_data_path}
            preprocess_coco_dataset(val_param_dict)
        else:
            param_dict = {"anno_json": config.anno_json, "coco_classes": config.coco_classes,
                          "coco_dir": config.coco_dir, "save_dir": data_dir}
            preprocess_coco_dataset(param_dict)
    else:
        raise ValueError("Not support dataset mode {}".format(config.dataset))
    print("========== end preprocess dataset ==========")

if __name__ == '__main__':
    preprocess_dataset(config.data_path)
