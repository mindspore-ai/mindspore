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
"""Evaluation utils for CTPN"""
import os
import subprocess
import numpy as np
from src.model_utils.config import config
from src.text_connector.detector import detect


def exec_shell_cmd(cmd):
    sub = subprocess.Popen(args="{}".format(cmd), shell=True, stdin=subprocess.PIPE, \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout_data, _ = sub.communicate()
    if sub.returncode != 0:
        raise ValueError("{} is not a executable command, please check.".format(cmd))
    return stdout_data.strip()

def get_eval_result():
    create_eval_bbox = 'rm -rf submit*.zip;cd ./submit/;zip -r ../submit.zip *.txt;cd ../;bash eval_res.sh'
    os.system(create_eval_bbox)
    get_eval_output = "grep hmean log | awk '{print $NF}' | awk -F} '{print $1}' |tail -n 1"
    hmean = exec_shell_cmd(get_eval_output)
    return float(hmean)


def eval_for_ctpn(network, dataset, eval_image_path):
    network.set_train(False)
    eval_iter = 0
    img_basenames = []
    local_path = os.getcwd()
    if config.enable_modelarts:
        local_path = os.path.join(config.modelarts_home, config.object_name)
    output_dir = os.path.join(local_path, "submit")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in os.listdir(eval_image_path):
        img_basenames.append(os.path.basename(file))
    img_basenames = sorted(img_basenames)
    for data in dataset.create_dict_iterator():
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']
        # run net
        output = network(img_data, gt_bboxes, gt_labels, gt_num)
        gt_bboxes = gt_bboxes.asnumpy()
        gt_labels = gt_labels.asnumpy()
        gt_num = gt_num.asnumpy().astype(bool)
        proposal = output[0]
        proposal_mask = output[1]
        for j in range(config.test_batch_size):
            img = img_basenames[config.test_batch_size * eval_iter + j]
            all_box_tmp = proposal[j].asnumpy()
            all_mask_tmp = np.expand_dims(proposal_mask[j].asnumpy(), axis=1)
            using_boxes_mask = all_box_tmp * all_mask_tmp
            textsegs = using_boxes_mask[:, 0:4].astype(np.float32)
            scores = using_boxes_mask[:, 4].astype(np.float32)
            shape = img_metas.asnumpy()[0][:2].astype(np.int32)
            bboxes = detect(textsegs, scores[:, np.newaxis], shape)
            from PIL import Image, ImageDraw
            im = Image.open(eval_image_path + '/' + img)
            draw = ImageDraw.Draw(im)
            image_h = img_metas.asnumpy()[j][2]
            image_w = img_metas.asnumpy()[j][3]
            gt_boxs = gt_bboxes[j][gt_num[j], :]
            for gt_box in gt_boxs:
                gt_x1 = gt_box[0] / image_w
                gt_y1 = gt_box[1] / image_h
                gt_x2 = gt_box[2] / image_w
                gt_y2 = gt_box[3] / image_h
                draw.line([(gt_x1, gt_y1), (gt_x1, gt_y2), (gt_x2, gt_y2), (gt_x2, gt_y1), (gt_x1, gt_y1)],\
                    fill='green', width=2)
            file_name = "res_" + img.replace("jpg", "txt")
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
            im.save(img)
        eval_iter = eval_iter + 1
