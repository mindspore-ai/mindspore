# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""create_imagenet2012_label"""
import os
import json
import argparse

parser = argparse.ArgumentParser(description="resnet imagenet2012 label")
parser.add_argument("--img_path", type=str, required=True, help="imagenet2012 file path.")
args = parser.parse_args()


def create_label(file_path):
    print("[WARNING] Create imagenet label. Currently only use for Imagenet2012!")
    dirs = os.listdir(file_path)
    file_list = []
    for file in dirs:
        file_list.append(file)
    file_list = sorted(file_list)

    total = 0
    img_label = {}
    for i, file_dir in enumerate(file_list):
        files = os.listdir(os.path.join(file_path, file_dir))
        for f in files:
            img_label[f] = i
        total += len(files)

    with open("imagenet_label.json", "w+") as label:
        json.dump(img_label, label)

    print("[INFO] Completed! Total {} data.".format(total))


if __name__ == '__main__':
    create_label(args.img_path)
