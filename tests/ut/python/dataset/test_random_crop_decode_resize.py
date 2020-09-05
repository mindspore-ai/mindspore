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
Testing RandomCropDecodeResize op in DE
"""
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore import log as logger
from util import diff_mse, visualize_image, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def test_random_crop_decode_resize_op(plot=False):
    """
    Test RandomCropDecodeResize op
    """
    logger.info("test_random_decode_resize_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    random_crop_decode_resize_op = vision.RandomCropDecodeResize((256, 512), (1, 1), (0.5, 0.5))
    data1 = data1.map(operations=random_crop_decode_resize_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_resize_op = vision.RandomResizedCrop((256, 512), (1, 1), (0.5, 0.5))
    data2 = data2.map(operations=decode_op, input_columns=["image"])
    data2 = data2.map(operations=random_crop_resize_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        if num_iter > 0:
            break
        image1 = item1["image"]
        image2 = item2["image"]
        mse = diff_mse(image1, image2)
        assert mse == 0
        logger.info("random_crop_decode_resize_op_{}, mse: {}".format(num_iter + 1, mse))
        if plot:
            visualize_image(image1, image2, mse)
        num_iter += 1


def test_random_crop_decode_resize_md5():
    """
    Test RandomCropDecodeResize with md5 check
    """
    logger.info("Test RandomCropDecodeResize with md5 check")
    original_seed = config_get_set_seed(10)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_crop_decode_resize_op = vision.RandomCropDecodeResize((256, 512), (1, 1), (0.5, 0.5))
    data = data.map(operations=random_crop_decode_resize_op, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "random_crop_decode_resize_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


if __name__ == "__main__":
    test_random_crop_decode_resize_op(plot=True)
    test_random_crop_decode_resize_md5()
