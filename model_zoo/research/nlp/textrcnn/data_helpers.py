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
"""dataset helpers api"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='textrcnn')
parser.add_argument('--task', type=str, help='the data preprocess task, including dataset_split.')
parser.add_argument('--data_dir', type=str, help='the source dataset directory.', default='./data_src')
parser.add_argument('--out_dir', type=str, help='the target dataset directory.', default='./data')

args = parser.parse_args()


def dataset_split(label):
    """dataset_split api"""
    # label can be 'pos' or 'neg'
    pos_samples = []
    pos_file = os.path.join(args.data_dir, "rt-polaritydata", "rt-polarity." + label)
    pfhand = open(pos_file, encoding='utf-8')
    pos_samples += pfhand.readlines()
    pfhand.close()
    np.random.seed(0)
    perm = np.random.permutation(len(pos_samples))
    perm_train = perm[0:int(len(pos_samples) * 0.9)]
    perm_test = perm[int(len(pos_samples) * 0.9):]
    pos_samples_train = []
    pos_samples_test = []
    for pt in perm_train:
        pos_samples_train.append(pos_samples[pt])
    for pt in perm_test:
        pos_samples_test.append(pos_samples[pt])
    f = open(os.path.join(args.out_dir, 'train', label), "w")
    f.write(''.join(pos_samples_train))
    f.close()

    f = open(os.path.join(args.out_dir, 'test', label), "w")
    f.write(''.join(pos_samples_test))
    f.close()


if __name__ == '__main__':
    if args.task == "dataset_split":
        dataset_split('pos')
        dataset_split('neg')
