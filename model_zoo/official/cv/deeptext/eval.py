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
import os
import time

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.Deeptext.deeptext_vgg16 import Deeptext_VGG16
from src.dataset import data_to_mindrecord_byte_image, create_deeptext_dataset
from src.utils import metrics

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

set_seed(1)

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())


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

    print("\n========================================\n", flush=True)
    print("Processing, please wait a moment.", flush=True)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

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
        print(gt_bboxes, flush=True)
        gt_labels = gt_labels.asnumpy()
        gt_labels = gt_labels[gt_num.asnumpy().astype(bool)]
        print(gt_labels, flush=True)
        end = time.time()
        print("Iter {} cost time {}".format(eval_iter, end - start), flush=True)

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

            print('    %s [%d/%d]' % (str(percent) + '%', eval_iter, total), end='\r', flush=True)

    precisions, recalls = metrics(pred_data)
    print("\n========================================\n", flush=True)
    for i in range(config.num_classes - 1):
        j = i + 1
        f1 = (2 * precisions[j] * recalls[j]) / (precisions[j] + recalls[j] + 1e-6)
        print("class {} precision is {:.2f}%, recall is {:.2f}%,"
              "F1 is {:.2f}%".format(j, precisions[j] * 100, recalls[j] * 100, f1 * 100), flush=True)
        if config.use_ambigous_sample:
            break


def modelarts_pre_process():
    '''modelarts pre process function.'''

    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...", flush=True)
                print("unzip file num: {}".format(data_num), flush=True)
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)), flush=True)
                print("Extract Done.", flush=True)
            else:
                print("This is not zip.", flush=True)
        else:
            print("Zip has been extracted.", flush=True)

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1, flush=True)
            print("Unzip file save dir: ", save_dir_1, flush=True)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===", flush=True)
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}."
              .format(get_device_id(), zip_file_1, save_dir_1), flush=True)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    prefix = config.eval_mindrecord_prefix
    config.test_images = config.imgs_path
    config.test_txts = config.annos_path
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    print("CHECKING MINDRECORD FILES ...", flush=True)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        print("Create Mindrecord. It may take some time.", flush=True)
        data_to_mindrecord_byte_image(False, prefix, file_num=1)
        print("Create Mindrecord Done, at {}".format(mindrecord_dir), flush=True)

    print("CHECKING MINDRECORD FILES DONE!", flush=True)
    print("Start Eval!", flush=True)
    deeptext_eval_test(mindrecord_file, config.checkpoint_path)


if __name__ == '__main__':
    run_eval()
