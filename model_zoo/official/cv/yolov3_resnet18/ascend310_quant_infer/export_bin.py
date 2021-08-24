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
from mindspore import context


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def generate_data(dataset_path):
    """
    Generate data and label needed for AIR model inference at Ascend310 platform.
    """
    ds = create_yolo_dataset(dataset_path, is_training=False)
    cur_dir = os.getcwd() + "/data"
    img_folder = cur_dir + "/00_image"
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    shape_folder = cur_dir + "/01_image_shape"
    if not os.path.exists(shape_folder):
        os.makedirs(shape_folder)
    total = ds.get_dataset_size()
    ann_list = []
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    prefix = "Yolov3-resnet18_coco_bs_1_"
    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        image_np = data['image']
        image_shape = data['image_shape']
        annotation = data['annotation']
        file_name = prefix + str(i) + ".bin"
        image_path = os.path.join(img_folder, file_name)
        image_np.tofile(image_path)
        shape_path = os.path.join(shape_folder, file_name)
        image_shape.tofile(shape_path)

        ann_list.append(annotation)
    ann_file = np.array(ann_list)
    np.save(os.path.join(cur_dir, "annotation_list.npy"), ann_file)


if __name__ == "__main__":
    sys.path.append("..")
    from src.dataset import create_yolo_dataset, data_to_mindrecord_byte_image
    from model_utils.config import config as default_config

    if not os.path.isdir(default_config.eval_mindrecord_dir):
        os.makedirs(default_config.eval_mindrecord_dir)

    yolo_prefix = "yolo.mindrecord"
    mindrecord_file = os.path.join(default_config.eval_mindrecord_dir, yolo_prefix + "0")
    if not os.path.exists(mindrecord_file):
        if os.path.isdir(default_config.image_dir) and os.path.exists(default_config.anno_path):
            print("Create Mindrecord")
            data_to_mindrecord_byte_image(default_config.image_dir,
                                          default_config.anno_path,
                                          default_config.eval_mindrecord_dir,
                                          prefix=yolo_prefix,
                                          file_num=8)
            print("Create Mindrecord Done, at {}".format(default_config.eval_mindrecord_dir))
        else:
            print("image_dir or anno_path not exits")

    generate_data(mindrecord_file)
