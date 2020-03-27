# Copyright 2019 Huawei Technologies Co., Ltd
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
Testing RgbToHsv and HsvToRgb op in DE
"""

import numpy as np
from numpy.testing import assert_allclose
import colorsys

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.py_transforms as vision
import mindspore.dataset.transforms.vision.py_transforms_util as util

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

def generate_numpy_random_rgb(shape):
    # Only generate floating points that are fractions like n / 256, since they
    # are RGB pixels. Some low-precision floating point types in this test can't
    # handle arbitrary precision floating points well.
    return np.random.randint(0, 256, shape) / 255.


def test_rgb_hsv_hwc():
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_np = rgb_flat.reshape((8, 8, 3))
    hsv_base = np.array([
        colorsys.rgb_to_hsv(
            r.astype(np.float64), g.astype(np.float64), b.astype(np.float64))
        for r, g, b in rgb_flat
    ])
    hsv_base = hsv_base.reshape((8, 8, 3))
    hsv_de = util.rgb_to_hsvs(rgb_np, True)
    assert hsv_base.shape == hsv_de.shape
    assert_allclose(hsv_base.flatten(), hsv_de.flatten(), rtol=1e-5, atol=0)

    hsv_flat = hsv_base.reshape(64, 3)
    rgb_base = np.array([
        colorsys.hsv_to_rgb(
            h.astype(np.float64), s.astype(np.float64), v.astype(np.float64))
        for h, s, v in hsv_flat
    ])
    rgb_base = rgb_base.reshape((8, 8, 3))
    rgb_de = util.hsv_to_rgbs(hsv_base, True)
    assert rgb_base.shape == rgb_de.shape
    assert_allclose(rgb_base.flatten(), rgb_de.flatten(), rtol=1e-5, atol=0)


def test_rgb_hsv_batch_hwc():
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_np = rgb_flat.reshape((4, 2, 8, 3))
    hsv_base = np.array([
        colorsys.rgb_to_hsv(
            r.astype(np.float64), g.astype(np.float64), b.astype(np.float64))
        for r, g, b in rgb_flat
    ])
    hsv_base = hsv_base.reshape((4, 2, 8, 3))
    hsv_de = util.rgb_to_hsvs(rgb_np, True)
    assert hsv_base.shape == hsv_de.shape
    assert_allclose(hsv_base.flatten(), hsv_de.flatten(), rtol=1e-5, atol=0)

    hsv_flat = hsv_base.reshape((64, 3))
    rgb_base = np.array([
        colorsys.hsv_to_rgb(
            h.astype(np.float64), s.astype(np.float64), v.astype(np.float64))
        for h, s, v in hsv_flat
    ])
    rgb_base = rgb_base.reshape((4, 2, 8, 3))
    rgb_de = util.hsv_to_rgbs(hsv_base, True)
    assert rgb_de.shape == rgb_base.shape
    assert_allclose(rgb_base.flatten(), rgb_de.flatten(), rtol=1e-5, atol=0)


def test_rgb_hsv_chw():
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_np = rgb_flat.reshape((3, 8, 8))
    hsv_base = np.array([
        np.vectorize(colorsys.rgb_to_hsv)(
            rgb_np[0, :, :].astype(np.float64), rgb_np[1, :, :].astype(np.float64), rgb_np[2, :, :].astype(np.float64))
    ])
    hsv_base = hsv_base.reshape((3, 8, 8))
    hsv_de = util.rgb_to_hsvs(rgb_np, False)
    assert hsv_base.shape == hsv_de.shape
    assert_allclose(hsv_base.flatten(), hsv_de.flatten(), rtol=1e-5, atol=0)

    rgb_base = np.array([
        np.vectorize(colorsys.hsv_to_rgb)(
            hsv_base[0, :, :].astype(np.float64), hsv_base[1, :, :].astype(np.float64),
            hsv_base[2, :, :].astype(np.float64))
    ])
    rgb_base = rgb_base.reshape((3, 8, 8))
    rgb_de = util.hsv_to_rgbs(hsv_base, False)
    assert rgb_de.shape == rgb_base.shape
    assert_allclose(rgb_base.flatten(), rgb_de.flatten(), rtol=1e-5, atol=0)


def test_rgb_hsv_batch_chw():
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_imgs = rgb_flat.reshape((4, 3, 2, 8))
    hsv_base_imgs = np.array([
        np.vectorize(colorsys.rgb_to_hsv)(
            img[0, :, :].astype(np.float64), img[1, :, :].astype(np.float64), img[2, :, :].astype(np.float64))
        for img in rgb_imgs
    ])
    hsv_de = util.rgb_to_hsvs(rgb_imgs, False)
    assert hsv_base_imgs.shape == hsv_de.shape
    assert_allclose(hsv_base_imgs.flatten(), hsv_de.flatten(), rtol=1e-5, atol=0)

    rgb_base = np.array([
        np.vectorize(colorsys.hsv_to_rgb)(
            img[0, :, :].astype(np.float64), img[1, :, :].astype(np.float64), img[2, :, :].astype(np.float64))
        for img in hsv_base_imgs
    ])
    rgb_de = util.hsv_to_rgbs(hsv_base_imgs, False)
    assert rgb_base.shape == rgb_de.shape
    assert_allclose(rgb_base.flatten(), rgb_de.flatten(), rtol=1e-5, atol=0)


def test_rgb_hsv_pipeline():
    # First dataset
    transforms1 = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transforms1 = vision.ComposeOp(transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds1 = ds1.map(input_columns=["image"], operations=transforms1())

    # Second dataset
    transforms2 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RgbToHsv(),
        vision.HsvToRgb()
    ]
    transform2 = vision.ComposeOp(transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds2 = ds2.map(input_columns=["image"], operations=transform2())

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(), ds2.create_dict_iterator()):
        num_iter += 1
        ori_img = data1["image"]
        cvt_img = data2["image"]
        assert_allclose(ori_img.flatten(), cvt_img.flatten(), rtol=1e-5, atol=0)
        assert (ori_img.shape == cvt_img.shape)


if __name__ == "__main__":
    test_rgb_hsv_hwc()
    test_rgb_hsv_batch_hwc()
    test_rgb_hsv_chw()
    test_rgb_hsv_batch_chw()
    test_rgb_hsv_pipeline()

