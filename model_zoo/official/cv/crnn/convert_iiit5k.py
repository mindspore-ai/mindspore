# Copyright 2020-21 Huawei Technologies Co., Ltd
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

import argparse
from scipy import io

###############################################
# load testdata
# testdata.mat structure
# test[:][0] : image name
# test[:][1] : label
# test[:][2] : 50 lexicon
# test[:][3] : 1000 lexicon
##############################################

def init_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-m', '--mat_file', type=str, default='testdata.mat',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--output_dir', type=str, default='./processed',
                        help='Directory where ord map dictionaries for the dataset were stored')

    parser.add_argument('-a', '--output_annotation', type=str, default='./annotation.txt',
                        help='Directory where ord map dictionaries for the dataset were stored')

    return parser.parse_args()


def mat_to_list(mat_file):
    ann_ori = io.loadmat(mat_file)
    testdata = ann_ori['testdata'][0]

    ann_output = []
    for elem in testdata:
        img_name = elem[0]
        label = elem[1]
        ann = img_name[0] + ',' + label[0]
        ann_output.append(ann)
    return ann_output


def convert():
    args = init_args()

    ann_list = mat_to_list(args.mat_file)

    ann_file = args.output_annotation
    with open(ann_file, 'w') as f:
        for line in ann_list:
            txt = line + '\n'
            f.write(txt)


if __name__ == "__main__":
    convert()
