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
# ============================================================================
"""use ImageNetToMR tool generate mindrecord"""
from mindspore.mindrecord import ImageNetToMR

IMAGENET_MAP_FILE = "../../../ut/data/mindrecord/testImageNetDataWhole/labels_map.txt"
IMAGENET_IMAGE_DIR = "../../../ut/data/mindrecord/testImageNetDataWhole/images"
MINDRECORD_FILE = "./imagenet.mindrecord"
PARTITION_NUMBER = 16


def imagenet_to_mindrecord():
    imagenet_transformer = ImageNetToMR(IMAGENET_MAP_FILE,
                                        IMAGENET_IMAGE_DIR,
                                        MINDRECORD_FILE,
                                        PARTITION_NUMBER)
    imagenet_transformer.transform()


if __name__ == '__main__':
    imagenet_to_mindrecord()
