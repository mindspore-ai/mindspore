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
import time
import numpy as np
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.ctpn import CTPN
from src.config import config
from src.dataset import create_ctpn_dataset
from src.text_connector.detector import detect
set_seed(1)

parser = argparse.ArgumentParser(description="CTPN evaluation")
parser.add_argument("--dataset_path", type=str, default="", help="Dataset path.")
parser.add_argument("--image_path", type=str, default="", help="Image path.")
parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint file path.")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
args_opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)

def ctpn_infer_test(dataset_path='', ckpt_path='', img_dir=''):
    """ctpn infer."""
    print("ckpt path is {}".format(ckpt_path))
    ds = create_ctpn_dataset(dataset_path, batch_size=config.test_batch_size, repeat_num=1, is_training=False)
    config.batch_size = config.test_batch_size
    total = ds.get_dataset_size()
    print("*************total dataset size is {}".format(total))
    net = CTPN(config, is_training=False)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    eval_iter = 0

    print("\n========================================\n")
    print("Processing, please wait a moment.")
    img_basenames = []
    output_dir = os.path.join(os.getcwd(), "submit")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in os.listdir(img_dir):
        img_basenames.append(os.path.basename(file))
    img_basenames = sorted(img_basenames)
    for data in ds.create_dict_iterator():
        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']

        start = time.time()
        # run net
        output = net(img_data, gt_bboxes, gt_labels, gt_num)
        gt_bboxes = gt_bboxes.asnumpy()
        gt_labels = gt_labels.asnumpy()
        gt_num = gt_num.asnumpy().astype(bool)
        end = time.time()
        proposal = output[0]
        proposal_mask = output[1]
        print("start to draw pic")
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
            im = Image.open(img_dir + '/' + img)
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
        percent = round(eval_iter / total * 100, 2)
        eval_iter = eval_iter + 1
        print("Iter {} cost time {}".format(eval_iter, end - start))
        print('    %s [%d/%d]' % (str(percent) + '%', eval_iter, total), end='\r')

if __name__ == '__main__':
    ctpn_infer_test(args_opt.dataset_path, args_opt.checkpoint_path, img_dir=args_opt.image_path)
