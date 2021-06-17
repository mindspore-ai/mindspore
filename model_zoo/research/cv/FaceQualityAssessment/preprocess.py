# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""preprocess dataset folder."""
import os
import shutil
import argparse

def seperate_image_label(data_path, result_path):
    '''seperate txt and jpg files as preprocess'''
    dirs = os.listdir(data_path)
    img_path = os.path.join(result_path, "image")
    label_path = os.path.join(result_path, "label")
    os.makedirs(img_path)
    os.makedirs(label_path)
    for file in dirs:
        if file != "Code":
            file_suffix = file.split('.')[1]
            if file_suffix == "jpg":
                file_path = os.path.join(data_path, file)
                save_path = os.path.join(img_path, file)
                shutil.copy(file_path, save_path)
            elif file_suffix == "txt":
                file_path = os.path.join(data_path, file)
                save_path = os.path.join(label_path, file)
                shutil.copy(file_path, save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Face Quality Assessment preprocess')
    parser.add_argument('--data_path', type=str, default='', help='data_path, e.g. ./face_quality_dataset/AWLW2000')
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='path to store preprocess')
    arg = parser.parse_args()
    seperate_image_label(data_path=arg.data_path, result_path=arg.result_path)
