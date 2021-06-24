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
# ===========================================================================
"""postprocess."""
import os
import numpy as np
from src.model_utils.config import config


def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count


def get_acc():
    '''calculate accuracy.'''
    img_tot = 0
    top1_correct = 0
    top5_correct = 0
    config.best_acc = 0
    config.index = 0

    gt_classes_list = np.load(config.label_path)
    file_num = len(os.listdir(config.post_result_path))

    for i in range(file_num):
        f_name = "dscnn+_bs" + str(config.per_batch_size) + "_" + str(i) + "_0.bin"
        output = np.fromfile(os.path.join(config.post_result_path, f_name), np.float32)
        output = output.reshape(config.per_batch_size, 12)

        top1_output = np.argmax(output, (-1))
        top5_output = np.argsort(output)[:, -5:]
        gt_classes = gt_classes_list[i]
        t1_correct = np.equal(top1_output, gt_classes).sum()
        top1_correct += t1_correct
        top5_correct += get_top5_acc(top5_output, gt_classes)
        img_tot += output.shape[0]

    results = [[top1_correct], [top5_correct], [img_tot]]

    results = np.array(results)

    top1_correct = results[0, 0]
    top5_correct = results[1, 0]
    img_tot = results[2, 0]
    acc1 = 100.0 * top1_correct / img_tot
    acc5 = 100.0 * top5_correct / img_tot
    if acc1 > config.best_acc:
        config.best_acc = acc1
        config.best_index = config.index
    print('Eval: top1_cor:{}, top5_cor:{}, tot:{}, acc@1={:.2f}%, acc@5={:.2f}%' \
          .format(top1_correct, top5_correct, img_tot, acc1, acc5))


if __name__ == "__main__":
    get_acc()
