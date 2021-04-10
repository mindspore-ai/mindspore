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

"""Evaluation for CTPN"""
import os
import argparse
import numpy as np

from src.text_connector.detector import detect

parser = argparse.ArgumentParser(description="CTPN evaluation")
parser.add_argument("--dataset_path", type=str, default="", help="Dataset path.")
parser.add_argument("--result_path", type=str, default="", help="Image path.")
parser.add_argument("--label_path", type=str, default="", help="label path.")
args_opt = parser.parse_args()

def get_pred(img_file, result_path):
    file_name = img_file.split('.')[0]
    proposal_file = os.path.join(result_path, file_name + "_0.bin")
    mask_file = os.path.join(result_path, file_name + "_1.bin")
    proposal = np.fromfile(proposal_file, dtype=np.float16).reshape(1000, 5)
    proposal_mask = np.fromfile(mask_file, dtype=np.int8).reshape(1000)

    return proposal, proposal_mask

def get_img_metas(imgSize):
    org_width, org_height = imgSize
    h_scale = 576 / org_height
    w_scale = 960 / org_width

    return np.array([576, 960, h_scale, w_scale])

def get_gt_box(img_file, label_path):
    label_file = os.path.join(label_path, img_file.replace("jpg", "txt"))
    file = open(label_file)
    lines = file.readlines()
    gt_boxs = []
    for line in lines:
        label_info = line.split(",")
        print(label_info)
        gt_boxs.append([int(label_info[0]), int(label_info[1]), int(label_info[2]), int(label_info[3])])

    return gt_boxs
def ctpn_infer_test(dataset_path='', result_path='', label_path=''):
    output_dir = "./output/"
    output_img_dir = "./output_img/"
    img_files = os.listdir(dataset_path)

    for file in img_files:
        print("processing image: ", file)
        from PIL import Image, ImageDraw
        img = Image.open(dataset_path + '/' + file)
        proposal, proposal_mask = get_pred(file, result_path)

        img_size = img.size
        img_metas = get_img_metas(img_size)
        all_box_tmp = proposal
        all_mask_tmp = np.expand_dims(proposal_mask, axis=1)

        using_boxes_mask = all_box_tmp * all_mask_tmp
        textsegs = using_boxes_mask[:, 0:4].astype(np.float32)
        scores = using_boxes_mask[:, 4].astype(np.float32)
        shape = img_metas[:2].astype(np.int32)

        bboxes = detect(textsegs, scores[:, np.newaxis], shape)

        draw = ImageDraw.Draw(img)
        image_h = img_metas[2]
        image_w = img_metas[3]
        gt_boxs = get_gt_box(file, label_path)
        for gt_box in gt_boxs:
            gt_x1 = gt_box[0]
            gt_y1 = gt_box[1]
            gt_x2 = gt_box[2]
            gt_y2 = gt_box[3]
            draw.line([(gt_x1, gt_y1), (gt_x1, gt_y2), (gt_x2, gt_y2), (gt_x2, gt_y1), (gt_x1, gt_y1)],\
                fill='green', width=2)
        file_name = "res_" + file.replace("jpg", "txt")
        output_file = os.path.join(output_dir, file_name)
        f = open(output_file, 'w')
        for bbox in bboxes:
            x1 = bbox[0] / image_w
            y1 = bbox[1] / image_h
            x2 = bbox[2] / image_w
            y2 = bbox[3] / image_h
            draw.line([(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)], fill='red', width=2)
            str_tmp = str(int(x1)) + "," + str(int(y1)) + "," + str(int(x2)) + "," + str(int(y2))
            f.write(str_tmp)
            f.write("\n")
        f.close()
        img.save(output_img_dir + file)

if __name__ == '__main__':
    ctpn_infer_test(args_opt.dataset_path, args_opt.result_path, args_opt.label_path)
