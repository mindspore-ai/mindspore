# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Evaluation for Deeptext"""
import argparse
import os
import time

import numpy as np
from src.Deeptext.deeptext_vgg16 import Deeptext_VGG16
from src.config import config
from src.dataset import data_to_mindrecord_byte_image, create_deeptext_dataset
from src.utils import metrics

from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

set_seed(1)

parser = argparse.ArgumentParser(description="Deeptext evaluation")
parser.add_argument("--checkpoint_path", type=str, default='test', help="Checkpoint file path.")
parser.add_argument("--imgs_path", type=str, required=True,
                    help="Test images files paths, multiple paths can be separated by ','.")
parser.add_argument("--annos_path", type=str, required=True,
                    help="Annotations files paths of test images, multiple paths can be separated by ','.")
parser.add_argument("--device_id", type=int, default=7, help="Device id, default is 7.")
parser.add_argument("--mindrecord_prefix", type=str, default='Deeptext-TEST', help="Prefix of mindrecord.")
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)


def deeptext_eval_test(dataset_path='', ckpt_path=''):
    """Deeptext evaluation."""
    ds = create_deeptext_dataset(dataset_path, batch_size=config.test_batch_size,
                                 repeat_num=1, is_training=False)

    total = ds.get_dataset_size()
    net = Deeptext_VGG16(config)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    eval_iter = 0

    print("\n========================================\n")
    print("Processing, please wait a moment.")
    max_num = 32

    pred_data = []
    for data in ds.create_dict_iterator():
        eval_iter = eval_iter + 1

        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']

        start = time.time()
        # run net
        output = net(img_data, img_metas, gt_bboxes, gt_labels, gt_num)
        gt_bboxes = gt_bboxes.asnumpy()

        gt_bboxes = gt_bboxes[gt_num.asnumpy().astype(bool), :]
        print(gt_bboxes)
        gt_labels = gt_labels.asnumpy()
        gt_labels = gt_labels[gt_num.asnumpy().astype(bool)]
        print(gt_labels)
        end = time.time()
        print("Iter {} cost time {}".format(eval_iter, end - start))

        # output
        all_bbox = output[0]
        all_label = output[1] + 1
        all_mask = output[2]

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
            all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
            all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])

            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]

            pred_data.append({"boxes": all_bboxes_tmp_mask,
                              "labels": all_labels_tmp_mask,
                              "gt_bboxes": gt_bboxes,
                              "gt_labels": gt_labels})

            percent = round(eval_iter / total * 100, 2)

            print('    %s [%d/%d]' % (str(percent) + '%', eval_iter, total), end='\r')

    precisions, recalls = metrics(pred_data)
    print("\n========================================\n")
    for i in range(config.num_classes - 1):
        j = i + 1
        f1 = (2 *  precisions[j] * recalls[j]) / (precisions[j] + recalls[j] + 1e-6)
        print("class {} precision is {:.2f}%, recall is {:.2f}%,"
              "F1 is {:.2f}%".format(j, precisions[j] * 100, recalls[j] * 100, f1 * 100))
        if config.use_ambigous_sample:
            break


if __name__ == '__main__':
    prefix = args_opt.mindrecord_prefix
    config.test_images = args_opt.imgs_path
    config.test_txts = args_opt.annos_path
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    print("CHECKING MINDRECORD FILES ...")
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create Mindrecord. It may take some time.")
        data_to_mindrecord_byte_image(False, prefix, file_num=1)
        print("Create Mindrecord Done, at {}".format(mindrecord_dir))

    print("CHECKING MINDRECORD FILES DONE!")
    print("Start Eval!")
    deeptext_eval_test(mindrecord_file, args_opt.checkpoint_path)
