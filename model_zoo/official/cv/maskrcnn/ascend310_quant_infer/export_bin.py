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
from mindspore import context


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def generate_data(dataset_path):
    """
    Generate data and label needed for AIR model inference at Ascend310 platform.
    """
    batch_size = 1
    ds = create_maskrcnn_dataset(dataset_path, batch_size=batch_size, is_training=False)
    total = ds.get_dataset_size()

    cur_dir = os.getcwd() + "/data"
    img_path = os.path.join(cur_dir, "00_img_data")
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    meta_path = os.path.join(cur_dir, "01_meta_data")
    if not os.path.exists(meta_path):
        os.makedirs(meta_path)
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    bin_prefix = "coco2017_maskrcnn_bs_1_"
    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["image"]
        img_metas = data["image_shape"]
        file_name = bin_prefix + str(i) + ".bin"
        img_data.tofile(os.path.join(img_path, file_name))
        img_metas.tofile(os.path.join(meta_path, file_name))


if __name__ == "__main__":
    sys.path.append("..")
    from src.model_utils.config import config
    from src.dataset import data_to_mindrecord_byte_image, create_maskrcnn_dataset

    prefix = "MaskRcnn_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")

    generate_data(mindrecord_file)
