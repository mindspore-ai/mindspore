# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import os
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.utils.browse_dataset import imshow_det_bbox


def test_browse_dataset():
    """
    Feature: Browse dataset
    Description: Demo code of visualization on VOC detection dataset
    Expectation: Runs successfully
    """
    # init
    DATA_DIR = "../data/dataset/testVOC2012_2"
    dataset = ds.VOCDataset(DATA_DIR, task="Detection",
                            usage="train", shuffle=False, decode=True, num_samples=3)
    dataset_iter = dataset.create_dict_iterator(
        output_numpy=True, num_epochs=1)

    # iter
    for index, data in enumerate(dataset_iter):
        image = data["image"]
        bbox = data["bbox"]
        label = data["label"]

        masks = np.zeros((4, image.shape[0], image.shape[1]))
        masks[0][0:500, 0:500] = 1
        masks[1][1000:1500, 1000:1500] = 2
        masks[2][0:500, 0:500] = 3
        masks[3][1000:1500, 1000:1500] = 4
        segm = masks

        imshow_det_bbox(image, bbox, label, segm,
                        class_names=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                                     'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                                     'sheep', 'sofa', 'train', 'tvmonitor'],
                        win_name="windows 98",
                        wait_time=5,
                        show=False,
                        out_file="test_browse_dataset_{}.jpg".format(str(index)))

        index += 1

    if os.path.exists("test_browse_dataset_0.jpg"):
        os.remove("test_browse_dataset_0.jpg")
    if os.path.exists("test_browse_dataset_1.jpg"):
        os.remove("test_browse_dataset_1.jpg")
    if os.path.exists("test_browse_dataset_2.jpg"):
        os.remove("test_browse_dataset_2.jpg")


if __name__ == "__main__":
    test_browse_dataset()
