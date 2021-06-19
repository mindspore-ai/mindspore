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
"""bin for 310 inference"""
import os
import numpy as np
from PIL import Image, ImageOps
from model_utils.config import config


def tf_pil(img):
    """ Letterbox an image to fit in the network """

    net_w, net_h = config.input_shape
    fill_color = 127
    im_w, im_h = img.size

    if im_w == net_w and im_h == net_h:
        return img

    # Rescaling
    if im_w / net_w >= im_h / net_h:
        scale = net_w / im_w
    else:
        scale = net_h / im_h
    if scale != 1:
        resample_mode = Image.NEAREST
        img = img.resize((int(scale * im_w), int(scale * im_h)), resample_mode)
        im_w, im_h = img.size

    if im_w == net_w and im_h == net_h:
        return img

    # Padding
    img_np = np.array(img)
    channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
    pad_w = (net_w - im_w) / 2
    pad_h = (net_h - im_h) / 2
    pad = (int(pad_w), int(pad_h), int(pad_w + .5), int(pad_h + .5))
    img = ImageOps.expand(img, border=pad, fill=(fill_color,) * channels)
    return img


def hwc2chw(img_np):
    return img_np.transpose(2, 0, 1).copy()


def to_tensor(image):
    image = np.asarray(image)
    image = hwc2chw(image)
    image = image / 255.
    return image.astype(np.float32)


if __name__ == '__main__':
    result_path = os.path.join(config.preprocess_path, 'images_bin')
    if not os.path.isdir(result_path):
        os.makedirs(result_path, exist_ok=True)
    data_path = os.path.join(config.preprocess_path, "images")
    files = os.listdir(data_path)
    for file in files:
        img_pil = Image.open(os.path.join(data_path, file)).convert("RGB")
        img_pil = tf_pil(img_pil)
        img_pil = to_tensor(img_pil)
        img_pil.tofile(os.path.join(result_path, file.split('.')[0] + '.bin'))
