# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Adapter dataset."""
import fnmatch
import io
import os

import numpy as np
from PIL import Image

from ..utils import file_io


def get_raw_samples(data_url):
    """
    Get dataset from raw data.

    Args:
        data_url (str): Dataset path.

    Returns:
        list, a file list.
    """
    def _list_files(dir_path, pattern):
        full_files = []
        _, _, files = next(file_io.walk(dir_path))
        for f in files:
            if fnmatch.fnmatch(f.lower(), pattern.lower()):
                full_files.append(os.path.join(dir_path, f))
        return full_files

    img_files = _list_files(os.path.join(data_url, "Images"), "*.jpg")
    seg_files = _list_files(os.path.join(data_url, "SegmentationClassRaw"), "*.png")

    files = []
    for img_file in img_files:
        _, file_name = os.path.split(img_file)
        name, _ = os.path.splitext(file_name)
        seg_file = os.path.join(data_url, "SegmentationClassRaw", ".".join([name, "png"]))
        if seg_file in seg_files:
            files.append([img_file, seg_file])
    return files


def read_image(img_path):
    """
    Read image from file.

    Args:
        img_path (str): image path.
    """
    img = file_io.read(img_path.strip(), binary=True)
    data = io.BytesIO(img)
    img = Image.open(data)
    return np.array(img)
