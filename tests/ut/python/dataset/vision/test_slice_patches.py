# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Testing SlicePatches Python API
"""
import numpy as np
import os



DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")





















def slice_patches(image, num_h, num_w, pad_or_drop, fill_value):
    """ help function which slice patches with numpy """
    if num_h == 1 and num_w == 1:
        return image
    # (H, W, C)
    H, W, C = image.shape
    patch_h = H // num_h
    patch_w = W // num_w
    if H % num_h != 0:
        if pad_or_drop:
            patch_h += 1
    if W % num_w != 0:
        if pad_or_drop:
            patch_w += 1
    img = image[:, :, :]
    if pad_or_drop:
        img = np.full([patch_h*num_h, patch_w*num_w, C], fill_value, dtype=np.uint8)
        img[:H, :W] = image[:, :, :]
    patches = []
    for top in range(num_h):
        for left in range(num_w):
            patches.append(img[top*patch_h:(top+1)*patch_h,
                               left*patch_w:(left+1)*patch_w, :])

    return (*patches,)
