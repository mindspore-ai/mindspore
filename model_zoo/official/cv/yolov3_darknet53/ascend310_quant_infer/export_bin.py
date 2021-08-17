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
"""generate data and label needed for AIR model inference"""
import os
import sys
import numpy as np


def generate_data():
    """
    Generate data and label needed for AIR model inference at Ascend310 platform.
    """
    config.batch_size = 1
    data_path = os.path.join(config.data_dir, "val2014")
    ds, data_size = create_yolo_dataset(data_path, config.annFile, is_training=False, batch_size=config.batch_size,
                                        max_epoch=1, device_num=1, rank=0, shuffle=False, config=config)
    print('testing shape : {}'.format(config.test_img_shape))
    print('total {} images to eval'.format(data_size))

    save_folder = "./data"
    image_folder = os.path.join(save_folder, "image_bin")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    list_image_shape = []
    list_image_id = []

    for i, data in enumerate(ds.create_dict_iterator()):
        image = data["image"].asnumpy()
        image_shape = data["image_shape"]
        image_id = data["img_id"]
        file_name = "YoloV3-DarkNet_coco_bs_" + str(config.batch_size) + "_" + str(i) + ".bin"
        file_path = image_folder + "/" + file_name
        image.tofile(file_path)
        list_image_shape.append(image_shape.asnumpy())
        list_image_id.append(image_id.asnumpy())
    shapes = np.array(list_image_shape)
    ids = np.array(list_image_id)
    np.save(save_folder + "/image_shape.npy", shapes)
    np.save(save_folder + "/image_id.npy", ids)


if __name__ == '__main__':
    sys.path.append("..")
    from model_utils.config import config
    from src.yolo_dataset import create_yolo_dataset

    generate_data()
