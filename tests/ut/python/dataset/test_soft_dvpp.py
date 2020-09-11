# Copyright 2020 Huawei Technologies Co., Ltd
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
Testing soft dvpp SoftDvppDecodeResizeJpeg and SoftDvppDecodeRandomCropResizeJpeg in DE
"""
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore import log as logger
from util import diff_mse, visualize_image

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_soft_dvpp_decode_resize_jpeg(plot=False):
    """
    Test SoftDvppDecodeResizeJpeg op
    """
    logger.info("test_random_decode_resize_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.Resize((256, 512))
    data1 = data1.map(operations=[decode_op, resize_op], input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    soft_dvpp_decode_resize_op = vision.SoftDvppDecodeResizeJpeg((256, 512))
    data2 = data2.map(operations=soft_dvpp_decode_resize_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        image1 = item1["image"]
        image2 = item2["image"]
        mse = diff_mse(image1, image2)
        assert mse <= 0.02
        logger.info("random_crop_decode_resize_op_{}, mse: {}".format(num_iter + 1, mse))
        if plot:
            visualize_image(image1, image2, mse)
        num_iter += 1


def test_soft_dvpp_decode_random_crop_resize_jpeg(plot=False):
    """
    Test SoftDvppDecodeRandomCropResizeJpeg op
    """
    logger.info("test_random_decode_resize_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_decode_resize_op = vision.RandomCropDecodeResize((256, 512), (1, 1), (0.5, 0.5))
    data1 = data1.map(operations=random_crop_decode_resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    soft_dvpp_random_crop_decode_resize_op = vision.SoftDvppDecodeRandomCropResizeJpeg((256, 512), (1, 1), (0.5, 0.5))
    data2 = data2.map(operations=soft_dvpp_random_crop_decode_resize_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        image1 = item1["image"]
        image2 = item2["image"]
        mse = diff_mse(image1, image2)
        assert mse <= 0.06
        logger.info("random_crop_decode_resize_op_{}, mse: {}".format(num_iter + 1, mse))
        if plot:
            visualize_image(image1, image2, mse)
        num_iter += 1


def test_soft_dvpp_decode_resize_jpeg_supplement(plot=False):
    """
    Test SoftDvppDecodeResizeJpeg op
    """
    logger.info("test_random_decode_resize_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.Resize(1134)
    data1 = data1.map(operations=[decode_op, resize_op], input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    soft_dvpp_decode_resize_op = vision.SoftDvppDecodeResizeJpeg(1134)
    data2 = data2.map(operations=soft_dvpp_decode_resize_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        image1 = item1["image"]
        image2 = item2["image"]
        mse = diff_mse(image1, image2)
        assert mse <= 0.02
        logger.info("random_crop_decode_resize_op_{}, mse: {}".format(num_iter + 1, mse))
        if plot:
            visualize_image(image1, image2, mse)
        num_iter += 1


if __name__ == "__main__":
    test_soft_dvpp_decode_resize_jpeg(plot=True)
    test_soft_dvpp_decode_random_crop_resize_jpeg(plot=True)
    test_soft_dvpp_decode_resize_jpeg_supplement(plot=True)
