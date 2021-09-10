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
    postprocess
"""
import os
import numpy as np
from PIL import Image
from src.utils.config import get_args
from mindspore import Tensor

def save_image(img, img_path):
    """Save a numpy image to the disk

    Parameters:
        img (numpy array / Tensor): image to save.
        image_path (str): the path of the image.
    """
    if isinstance(img, Tensor):
        img = img.asnumpy()
    elif not isinstance(img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))
    img = decode_image(img)

    img_pil = Image.fromarray(img)
    img_pil.save(img_path + ".jpg")

def decode_image(img):
    """Decode a [1, C, H, W] Tensor to image numpy array."""
    mean = 0.5 * 255
    std = 0.5 * 255

    return (img * std + mean).astype(np.uint8).transpose((1, 2, 0))

if __name__ == '__main__':
    args = get_args()

    result_dir = "./result_Files"
    object_imageSize = 256
    rst_path = result_dir

    for i in range(len(os.listdir(rst_path))):
        file_name = os.path.join(rst_path, "Pix2Pix_data_bs" + str(args.batch_size) + '_' + str(i) + '_0.bin')
        output = np.fromfile(file_name, np.float32).reshape(3, object_imageSize, object_imageSize)
        print(output.shape)
        save_image(output, './310_infer_img' + str(i + 1))
        print("=======image", i + 1, "saved success=======")
    print("Generate images success!")
