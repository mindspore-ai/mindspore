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
from __future__ import division
import os
import numpy as np
from PIL import Image
from mindspore.mindrecord import FileWriter
from src.config import config

def create_coco_label():
    """Create image label."""
    image_files = []
    image_anno_dict = {}
    coco_root = config.coco_root
    data_type = config.coco_train_data_type
    from src.coco_text import COCO_Text
    anno_json = config.cocotext_json
    ct = COCO_Text(anno_json)
    image_ids = ct.getImgIds(imgIds=ct.train,
                             catIds=[('legibility', 'legible')])
    for img_id in image_ids:
        image_info = ct.loadImgs(img_id)[0]
        file_name = image_info['file_name'][15:]
        anno_ids = ct.getAnnIds(imgIds=img_id)
        anno = ct.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        im = Image.open(image_path)
        width, _ = im.size
        for label in anno:
            bbox = label["bbox"]
            bbox_width = int(bbox[2])
            if 60 * bbox_width < width:
                continue
            x1, x2 = int(bbox[0]), int(bbox[0] + bbox[2])
            y1, y2 = int(bbox[1]), int(bbox[1] + bbox[3])
            annos.append([x1, y1, x2, y2] + [1])
        if annos:
            image_anno_dict[image_path] = np.array(annos)
            image_files.append(image_path)
    return image_files, image_anno_dict

def create_anno_dataset_label(train_img_dirs, train_txt_dirs):
    image_files = []
    image_anno_dict = {}
    # read
    img_basenames = []
    for file in os.listdir(train_img_dirs):
        # Filter git file.
        if 'gif' not in file:
            img_basenames.append(os.path.basename(file))
    img_names = []
    for item in img_basenames:
        temp1, _ = os.path.splitext(item)
        img_names.append((temp1, item))
    for img, img_basename in img_names:
        image_path = train_img_dirs + '/' + img_basename
        annos = []
        if len(img) == 6 and '_' not in img_basename:
            gt = open(train_txt_dirs + '/' + img + '.txt').read().splitlines()
            if img.isdigit() and int(img) > 1200:
                continue
            for img_each_label in gt:
                spt = img_each_label.replace(',', '').split(' ')
                if ' ' not in img_each_label:
                    spt = img_each_label.split(',')
                annos.append([spt[0], spt[1], str(int(spt[0]) + int(spt[2])), str(int(spt[1]) +  int(spt[3]))] + [1])
            if annos:
                image_anno_dict[image_path] = np.array(annos)
                image_files.append(image_path)
    return image_files, image_anno_dict

def create_icdar_svt_label(train_img_dir, train_txt_dir, prefix):
    image_files = []
    image_anno_dict = {}
    img_basenames = []
    for file_name in os.listdir(train_img_dir):
        if 'gif' not in file_name:
            img_basenames.append(os.path.basename(file_name))
    img_names = []
    for item in img_basenames:
        temp1, _ = os.path.splitext(item)
        img_names.append((temp1, item))
    for img, img_basename in img_names:
        image_path = train_img_dir + '/' + img_basename
        annos = []
        file_name = prefix + img + ".txt"
        file_path = os.path.join(train_txt_dir, file_name)
        gt = open(file_path, 'r', encoding='UTF-8-sig').read().splitlines()
        if not gt:
            continue
        for img_each_label in gt:
            spt = img_each_label.replace(',', '').split(' ')
            if ' ' not in img_each_label:
                spt = img_each_label.split(',')
            annos.append([spt[0], spt[1], spt[2], spt[3]] + [1])
        if annos:
            image_anno_dict[image_path] = np.array(annos)
            image_files.append(image_path)
    return image_files, image_anno_dict

def create_train_dataset(dataset_type):
    image_files = []
    image_anno_dict = {}
    if dataset_type == "pretraining":
        # pretrianing: coco, flick, icdar2013 train, icdar2015, svt
        coco_image_files, coco_anno_dict = create_coco_label()
        flick_image_files, flick_anno_dict = create_anno_dataset_label(config.flick_train_path[0],
                                                                       config.flick_train_path[1])
        icdar13_image_files, icdar13_anno_dict = create_icdar_svt_label(config.icdar13_train_path[0],
                                                                        config.icdar13_train_path[1], "gt_img_")
        icdar15_image_files, icdar15_anno_dict = create_icdar_svt_label(config.icdar15_train_path[0],
                                                                        config.icdar15_train_path[1], "gt_")
        svt_image_files, svt_anno_dict = create_icdar_svt_label(config.svt_train_path[0], config.svt_train_path[1], "")
        image_files = coco_image_files + flick_image_files + icdar13_image_files + icdar15_image_files + svt_image_files
        image_anno_dict = {**coco_anno_dict, **flick_anno_dict, \
            **icdar13_anno_dict, **icdar15_anno_dict, **svt_anno_dict}
        data_to_mindrecord_byte_image(image_files, image_anno_dict, config.pretrain_dataset_path, \
            prefix="ctpn_pretrain.mindrecord", file_num=8)
    elif dataset_type == "finetune":
        # finetune: icdar2011, icdar2013 train, flick
        flick_image_files, flick_anno_dict = create_anno_dataset_label(config.flick_train_path[0],
                                                                       config.flick_train_path[1])
        icdar11_image_files, icdar11_anno_dict = create_icdar_svt_label(config.icdar11_train_path[0],
                                                                        config.icdar11_train_path[1], "gt_")
        icdar13_image_files, icdar13_anno_dict = create_icdar_svt_label(config.icdar13_train_path[0],
                                                                        config.icdar13_train_path[1], "gt_img_")
        image_files = flick_image_files + icdar11_image_files + icdar13_image_files
        image_anno_dict = {**flick_anno_dict, **icdar11_anno_dict, **icdar13_anno_dict}
        data_to_mindrecord_byte_image(image_files, image_anno_dict, config.finetune_dataset_path, \
            prefix="ctpn_finetune.mindrecord", file_num=8)
    elif dataset_type == "test":
        # test: icdar2013 test
        icdar_test_image_files, icdar_test_anno_dict = create_icdar_svt_label(config.icdar13_test_path[0],\
            config.icdar13_test_path[1], "")
        image_files = sorted(icdar_test_image_files)
        image_anno_dict = icdar_test_anno_dict
        data_to_mindrecord_byte_image(image_files, image_anno_dict, config.test_dataset_path, \
            prefix="ctpn_test.mindrecord", file_num=1)
    else:
        print("dataset_type should be pretraining, finetune, test")

def data_to_mindrecord_byte_image(image_files, image_anno_dict, dst_dir, prefix="cptn_mlt.mindrecord", file_num=1):
    """Create MindRecord file."""
    mindrecord_path = os.path.join(dst_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)

    ctpn_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
    }
    writer.add_schema(ctpn_json, "ctpn_json")
    for image_name in image_files:
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        print("img name is {}, anno is {}".format(image_name, annos))
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()

if __name__ == "__main__":
    create_train_dataset("pretraining")
    create_train_dataset("finetune")
    create_train_dataset("test")
