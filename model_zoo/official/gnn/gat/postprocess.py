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
"""postprocess"""
import argparse
import os

import numpy as np


def MaskedAccuracy(label, mask, logits, num_class):
    """Calculate accuracy with mask"""
    logits = logits.reshape(-1, num_class)
    labels = label.reshape(-1, num_class)
    mask = mask.reshape(-1,)

    labels = labels.astype(np.float32)

    correct_prediction = (np.argmax(logits, axis=1) == np.argmax(labels, axis=1))
    accuracy_all = correct_prediction.astype(np.float32)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    accuracy_all *= mask
    return np.mean(accuracy_all)


def get_acc():
    """get infer accuracy."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cora', choices=['cora', 'citeseer'], help='dataset name')
    parser.add_argument('--result_path', type=str, default='./result_Files', help='result Files')
    parser.add_argument('--label_path', type=str, default='', help='y_test npy Files')
    parser.add_argument('--mask_path', type=str, default='', help='test_mask npy Files')
    args = parser.parse_args()

    label = np.load(args.label_path)
    mask = np.load(args.mask_path)
    logits = np.fromfile(os.path.join(args.result_path, 'feature_0.bin'), np.float16)
    if args.dataset_name == 'citeseer':
        logits = logits.reshape(1, 3312, 6)
    else:
        logits = logits.reshape(1, 2708, 7)

    num_class = label.shape[2]
    acc = MaskedAccuracy(label, mask, logits, num_class)
    print("test acc={}".format(acc))


if __name__ == "__main__":
    get_acc()
