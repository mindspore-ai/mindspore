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
"""Create Dataset."""
import os
import argparse
import glob
import numpy as np
import PIL.Image as pil_image
from PIL import ImageFile

from mindspore.mindrecord import FileWriter

from src.config import srcnn_cfg as config
from src.utils import convert_rgb_to_y
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Generate dataset file.')
parser.add_argument("--src_folder", type=str, required=True, help="Raw data folder.")
parser.add_argument("--output_folder", type=str, required=True, help="Dataset output path.")

if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    prefix = "srcnn.mindrecord"
    file_num = 32
    patch_size = config.patch_size
    stride = config.stride
    scale = config.scale
    mindrecord_path = os.path.join(args.output_folder, prefix)
    writer = FileWriter(mindrecord_path, file_num)

    srcnn_json = {
        "lr": {"type": "float32", "shape": [1, patch_size, patch_size]},
        "hr": {"type": "float32", "shape": [1, patch_size, patch_size]},
    }
    writer.add_schema(srcnn_json, "srcnn_json")
    image_list = []
    file_list = sorted(os.listdir(args.src_folder))
    for file_name in file_list:
        path = os.path.join(args.src_folder, file_name)
        if os.path.isfile(path):
            image_list.append(path)
        else:
            for image_path in sorted(glob.glob('{}/*'.format(path))):
                image_list.append(image_path)

    print("image_list size ", len(image_list), flush=True)

    for path in image_list:
        hr = pil_image.open(path).convert('RGB')
        hr_width = (hr.width // scale) * scale
        hr_height = (hr.height // scale) * scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - patch_size + 1, stride):
            for j in range(0, lr.shape[1] - patch_size + 1, stride):
                lr_res = np.expand_dims(lr[i:i + patch_size, j:j + patch_size] / 255., 0)
                hr_res = np.expand_dims(hr[i:i + patch_size, j:j + patch_size] / 255., 0)
                row = {"lr": lr_res, "hr": hr_res}
                writer.write_raw_data([row])

    writer.commit()
    print("Finish!")
