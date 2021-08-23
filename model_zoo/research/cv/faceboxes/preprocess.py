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
FaceBoxes data preprocess.
"""
import os


def write2list(data_dir, go, output_path, is_train):
    """write image info to image list txt file"""
    if os.path.exists(output_path):
        os.remove(output_path)
    for items in go:
        items[1].sort()
        for file_dir in items[1]:
            go_deeper = os.walk(os.path.join(data_dir, "images", file_dir))
            for items_deeper in go_deeper:
                items_deeper[2].sort()
                for file in items_deeper[2]:
                    with open(output_path, 'a') as fw:
                        if is_train:
                            fw.write(file_dir + '/' + file + file.split('.')[0] + '\n')
                        else:
                            fw.write(file_dir + '/' + file + '\n')


if __name__ == '__main__':
    train_dir = os.path.join(os.getcwd(), 'data/widerface/train/')
    train_out = os.path.join(train_dir, 'train_img_list.txt')
    train_go = os.walk(os.path.join(train_dir, 'images'))
    val_dir = os.path.join(os.getcwd(), 'data/widerface/val/')
    val_out = os.path.join(val_dir, 'val_img_list.txt')
    val_go = os.walk(os.path.join(val_dir, 'images'))
    write2list(train_dir, train_go, train_out, True)
    write2list(val_dir, val_go, val_out, False)
