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
"""
#################SimCLR postprocess########################
"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='SimCLR Postprocess')
parser.add_argument('--label_dir', type=str, default='', help='label data directory.')
parser.add_argument('--result_dir', type=str, default="./result_Files",
                    help='infer result dir.')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument('--class_num', type=int, default=10, help='dataset classification number, default is 10.')
args, _ = parser.parse_known_args()


if __name__ == '__main__':

    rst_path = args.result_dir
    labels = np.load(args.label_dir)
    top1 = 0
    total_data = len(os.listdir(rst_path))

    for i in range(total_data):
        file_name = os.path.join(rst_path, "cifar10_data_bs" + str(args.batch_size) + '_' + str(i) + '_0.bin')
        output = np.fromfile(file_name, dtype=np.float32).reshape(args.batch_size, args.class_num)
        for j in range(args.batch_size):
            predict = np.argmax(output[j], axis=0)
            y = labels[i][j]
            if predict == y:
                top1 += 1

    print("result of Accuracy is: ", top1 / (total_data * args.batch_size))
