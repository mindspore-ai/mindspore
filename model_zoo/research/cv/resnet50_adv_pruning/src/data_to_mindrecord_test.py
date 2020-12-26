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
"""
data transform to mindrecord format
"""
import os
import argparse
import numpy as np
from mindspore.mindrecord import FileWriter

parser = argparse.ArgumentParser(description='Export parameter')
parser.add_argument('--image_root_dir', type=str,
                    default='/home/dataset/Oxford_IIIT_Pet/images/', help='Dataset path')
parser.add_argument('--annotation_dir', type=str,
                    default='/home/dataset/Oxford_IIIT_Pet/annotations', help='Annotation path')
parser.add_argument('--mindrecord_file_name', type=str,
                    default='/home/dataset/Oxford_IIIT_Pet/test.mindrecord', help='Mindrecord path')
parser.add_argument('--mindrecord_num', type=int, default=1, help='Mindrecord num')
args_opt = parser.parse_args()

if __name__ == '__main__':
    txt_path = os.path.join(args_opt.annotation_dir, "test.txt")
    writer = FileWriter(args_opt.mindrecord_file_name, args_opt.mindrecord_num)
    test_json = {
        "image": {"type": "bytes"},
        "label_list": {"type": "int32"},
    }
    writer.add_schema(test_json, "test_json")

    image_path_list = []
    label_list = []
    with open(txt_path, "r") as rf:
        for line in rf:
            str_list = line.strip().split(' ')
            img_name = str_list[0]
            image_path_list.append(os.path.join(args_opt.image_root_dir, img_name + ".jpg"))
            label = int(str_list[1]) - 1
            label_list.append(label)

    for index, _ in enumerate(image_path_list):
        path = image_path_list[index]
        target = label_list[index]
        img = open(path, 'rb').read()
        if target is None:
            target = np.zeros(1)
        row = {"image": img, "label_list": target}
        writer.write_raw_data([row])

    print('total test images: ', len(image_path_list))
    writer.commit()
