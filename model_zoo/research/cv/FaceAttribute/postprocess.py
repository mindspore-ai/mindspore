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
"""postprocess for 310 inference"""
import os
import argparse
import numpy as np

batch_size = 1
parser = argparse.ArgumentParser(description="face attribute acc postprocess")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--label_path", type=str, default="./data/label", required=True, help="image label file path.")
args = parser.parse_args()

def calcul_acc(lab, preds):
    return sum(1 for x, y in zip(lab, preds) if x == y) / len(lab)

def get_result(result_path, img_label_path):
    """get accuracy result"""
    files = os.listdir(img_label_path)
    preds_age = []
    preds_gen = []
    preds_mask = []
    labels_age = []
    labels_gen = []
    labels_mask = []
    for file in files:
        label = np.fromfile(os.path.join(img_label_path, file), dtype=np.int32)
        labels_age.append(int(label[0]))
        labels_gen.append(int(label[1]))
        labels_mask.append(int(label[2]))
        file_name = file.split('.')[0]
        age_result_path = os.path.join(result_path, file_name+'_0.bin')
        gen_result_path = os.path.join(result_path, file_name+'_1.bin')
        mask_result_path = os.path.join(result_path, file_name+'_2.bin')
        output_age = np.fromfile(age_result_path, dtype=np.float32)
        output_gen = np.fromfile(gen_result_path, dtype=np.float32)
        output_mask = np.fromfile(mask_result_path, dtype=np.float32)
        preds_age.append(np.argmax(output_age, axis=0))
        preds_gen.append(np.argmax(output_gen, axis=0))
        preds_mask.append(np.argmax(output_mask, axis=0))
    age_acc = calcul_acc(labels_age, preds_age)
    gen_acc = calcul_acc(labels_gen, preds_gen)
    mask_acc = calcul_acc(labels_mask, preds_mask)
    print("age accuracy: {}".format(age_acc))
    print("gen accuracy: {}".format(gen_acc))
    print("mask accuracy: {}".format(mask_acc))
if __name__ == '__main__':
    get_result(args.result_path, args.label_path)
