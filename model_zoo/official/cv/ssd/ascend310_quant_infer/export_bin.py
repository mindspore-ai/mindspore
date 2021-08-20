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


def generate_data():
    """
    Generate data and label needed for AIR model inference at Ascend310 platform.
    """
    mindrecord_file = create_mindrecord("coco", "ssd_eval.mindrecord", False)
    batch_size = 1
    ds = create_ssd_dataset(mindrecord_file, batch_size=batch_size, repeat_num=1, is_training=False,
                            use_multiprocessing=False)
    cur_dir = os.getcwd()
    image_path = os.path.join(cur_dir, "./data/00_image_data")
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    img_id_path = os.path.join(cur_dir, "./data/01_image_id")
    if not os.path.isdir(img_id_path):
        os.makedirs(img_id_path)
    img_shape_path = os.path.join(cur_dir, "./data/02_image_shape")
    if not os.path.isdir(img_shape_path):
        os.makedirs(img_shape_path)
    total = ds.get_dataset_size()
    iter_num = 0
    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        file_name = "coco_bs_" + str(batch_size) + "_" + str(iter_num) + ".bin"
        img_id = data["img_id"]
        img_np = data["image"]
        img_shape = data["image_shape"]
        img_id.tofile(os.path.join(img_id_path, file_name))
        img_np.tofile(os.path.join(image_path, file_name))
        img_shape.tofile(os.path.join(img_shape_path, file_name))

        iter_num += 1
    print("total images num: ", total)


if __name__ == "__main__":
    sys.path.append("..")
    from src.dataset import create_ssd_dataset, create_mindrecord

    generate_data()
