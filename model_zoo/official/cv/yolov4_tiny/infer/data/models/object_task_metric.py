#Copyright 2021 Huawei Technologies Co., Ltd
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_image_id(label_file):
    """
    :param: label file path, default is coco2017_minival.txt
    :return: image id
    """
    image_list = []
    with open(label_file, 'r')as f_read:
        ban_list = f_read.read().split('\n')[:-1]
        for item in ban_list:
            image_path = item.split(' ')[1]
            image_name = image_path.split('/')[-1]
            image_id = image_name.split('.')[0].split('_')[-1]
            image_list.append(int(image_id))
    return image_list


def get_category_id(class_id):
    """
    :param: class id which corresponding coco.names
    :return: category id is used in instances_val2017.json
    """
    if 0 <= class_id <= 10:
        class_id = class_id + 1
    elif 11 <= class_id <= 23:
        class_id = class_id + 2
    elif 24 <= class_id <= 25:
        class_id = class_id + 3
    elif 26 <= class_id <= 39:
        class_id = class_id + 5
    elif 40 <= class_id <= 59:
        class_id = class_id + 6
    elif class_id == 60:
        class_id = class_id + 7
    elif class_id == 61:
        class_id = class_id + 9
    elif 62 <= class_id <= 72:
        class_id = class_id + 10
    elif 73 <= class_id <= 79:
        class_id = class_id + 11
    return class_id

def get_img_set(anno_json_path):
    """Get image path and annotation from COCO."""
    need_img_ids = []
    coco = COCO(anno_json_path)
    image_ids = coco.getImgIds()
    print("first dataset is {}".format(len(image_ids)))
    for img_id in image_ids:
        iscrowd = False
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        for label in anno:
            iscrowd = iscrowd or label["iscrowd"]

        if iscrowd:
            continue
        need_img_ids.append(img_id)

    return need_img_ids

def get_dict_from_file(file_path, id_list):
    """
    :param: file_path contain all infer result
    :param: id_list contain all images id which is corresponding instances_val2017.json
    :return: dict_list contain infer result of every images
    """
    print(len(id_list))
    ls = []
    image_dict = {}
    count = -1
    with open(file_path, 'r')as fs:
        ban_list = fs.read().split('\n')
        for item in ban_list:
            if item == '':
                continue
            if item[0] != '#':
                count = count + 1
                continue
            image_list = item.split(',')
            image_dict['image_id'] = id_list[count]
            image_dict['category_id'] = get_category_id(int(image_list[-1].strip().split(' ')[-1]))
            bbox_list = [float(i) for i in image_list[1].strip().split(' ')[1:]]
            bbox_list[2] = bbox_list[2] - bbox_list[0]
            bbox_list[3] = bbox_list[3] - bbox_list[1]
            image_dict['bbox'] = bbox_list
            image_dict['score'] = float(image_list[2].strip().split(' ')[-1])
            ls.append(image_dict.copy())
    return ls


def get_img_id(file_name):
    """
    get image id list from result data
    """
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset


if __name__ == "__main__":
    ban_path = './coco2017_minival.txt'
    input_file = './result/result.txt'
    if not os.path.exists(ban_path):
        print('The infer text file does not exist.')
    if not os.path.exists(input_file):
        print('The result text file does not exist.')

    image_id_list = get_image_id(ban_path)
    result_dict = get_dict_from_file(input_file, image_id_list)
    json_file_name = './result.json'
    with open(json_file_name, 'w') as f:
        json.dump(result_dict, f)

    # set iouType to 'segm', 'bbox' or 'keypoints'
    ann_type = ('segm', 'bbox', 'keypoints')
    # specify type here
    ann_type = ann_type[1]
    coco_gt_file = './instances_val2017.json'
    coco_gt = COCO(coco_gt_file)
    coco_dt_file = './result.json'

    coco_dt = coco_gt.loadRes(coco_dt_file)
    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.imgIds = get_img_set(coco_gt_file)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
