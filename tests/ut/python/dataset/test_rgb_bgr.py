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
# ==============================================================================
"""
Testing RgbToBgr op in DE
"""

import numpy as np
from numpy.testing import assert_allclose
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.vision.py_transforms_util as util

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def generate_numpy_random_rgb(shape):
    # Only generate floating points that are fractions like n / 256, since they
    # are RGB pixels. Some low-precision floating point types in this test can't
    # handle arbitrary precision floating points well.
    return np.random.randint(0, 256, shape) / 255.


def test_rgb_bgr_hwc_py():
    # Eager
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_np = rgb_flat.reshape((8, 8, 3))

    bgr_np_pred = util.rgb_to_bgrs(rgb_np, True)
    r, g, b = rgb_np[:, :, 0], rgb_np[:, :, 1], rgb_np[:, :, 2]
    bgr_np_gt = np.stack((b, g, r), axis=2)
    assert bgr_np_pred.shape == rgb_np.shape
    assert_allclose(bgr_np_pred.flatten(),
                    bgr_np_gt.flatten(),
                    rtol=1e-5,
                    atol=0)


def test_rgb_bgr_hwc_c():
    # Eager
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_np = rgb_flat.reshape((8, 8, 3))

    rgb2bgr_op = vision.RgbToBgr()
    bgr_np_pred = rgb2bgr_op(rgb_np)
    r, g, b = rgb_np[:, :, 0], rgb_np[:, :, 1], rgb_np[:, :, 2]
    bgr_np_gt = np.stack((b, g, r), axis=2)
    assert bgr_np_pred.shape == rgb_np.shape
    assert_allclose(bgr_np_pred.flatten(),
                    bgr_np_gt.flatten(),
                    rtol=1e-5,
                    atol=0)


def test_rgb_bgr_chw_py():
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_np = rgb_flat.reshape((3, 8, 8))

    rgb_np_pred = util.rgb_to_bgrs(rgb_np, False)
    rgb_np_gt = rgb_np[::-1, :, :]
    assert rgb_np_pred.shape == rgb_np.shape
    assert_allclose(rgb_np_pred.flatten(),
                    rgb_np_gt.flatten(),
                    rtol=1e-5,
                    atol=0)


def test_rgb_bgr_pipeline_py():
    # First dataset
    transforms1 = [py_vision.Decode(), py_vision.Resize([64, 64]), py_vision.ToTensor()]
    transforms1 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        py_vision.Decode(),
        py_vision.Resize([64, 64]),
        py_vision.ToTensor(),
        py_vision.RgbToBgr()
    ]
    transforms2 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transforms2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        cvt_img_gt = ori_img[::-1, :, :]
        assert_allclose(cvt_img_gt.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        assert ori_img.shape == cvt_img.shape


def test_rgb_bgr_pipeline_c():
    # First dataset
    transforms1 = [
        vision.Decode(),
        vision.Resize([64, 64])
    ]
    transforms1 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(),
        vision.Resize([64, 64]),
        vision.RgbToBgr()
    ]
    transforms2 = mindspore.dataset.transforms.py_transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transforms2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        cvt_img_gt = ori_img[:, :, ::-1]
        assert_allclose(cvt_img_gt.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        assert ori_img.shape == cvt_img.shape


if __name__ == "__main__":
    test_rgb_bgr_hwc_py()
    test_rgb_bgr_hwc_c()
    test_rgb_bgr_chw_py()
    test_rgb_bgr_pipeline_py()
    test_rgb_bgr_pipeline_c()
