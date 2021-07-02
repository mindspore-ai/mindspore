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
"""pre process for 310 inference"""
import os
import shutil
import cv2
import numpy as np
from src.model_utils.config import config
from dependency.centernet.src.lib.detectors.base_detector import CenterFaceDetector


def preprocess(dataset_path, preprocess_path):
    event_list = os.listdir(dataset_path)
    input_path = os.path.join(preprocess_path, "input")
    meta_path = os.path.join(preprocess_path, "meta/meta")
    if not os.path.exists(input_path):
        os.makedirs(os.path.join(preprocess_path, "input"))
    if not os.path.exists(meta_path):
        os.makedirs(os.path.join(preprocess_path, "meta/meta"))

    detector = CenterFaceDetector(config, None)
    name_list = []
    meta_list = []
    i = 0
    for _, event in enumerate(event_list):
        file_list_item = os.listdir(os.path.join(dataset_path, event))
        im_dir = event
        for _, file in enumerate(file_list_item):
            im_name = file.split('.')[0]
            zip_name = '%s/%s' % (im_dir, file)
            img_path = os.path.join(dataset_path, zip_name)
            image = cv2.imread(img_path)
            for scale in config.test_scales:
                _, meta = detector.pre_process(image, scale)
            img_file_path = os.path.join(input_path, file)
            shutil.copyfile(img_path, img_file_path)
            meta_file_path = os.path.join(preprocess_path + "/meta/meta", im_name + ".txt")
            with open(meta_file_path, 'w+') as f:
                f.write(str(meta))
            name_list.append(im_name)
            meta_list.append(meta)
            i += 1
            print(f"preprocess: no.[{i}], img_name:{im_name}")
    np.save(os.path.join(preprocess_path + "/meta", "name_list.npy"), np.array(name_list))
    np.save(os.path.join(preprocess_path + "/meta", "meta_list.npy"), np.array(meta_list))


if __name__ == '__main__':
    preprocess(config.dataset_path, config.preprocess_path)
