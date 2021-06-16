# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""pre process for 310 inference"""
import os
import argparse
import numpy as np
from PIL import Image

import mindspore.dataset.vision.py_transforms as V
import mindspore.dataset.transforms.py_transforms as T


def load_images(paths, batch_size=1):
    '''Load images.'''
    ll = []
    resize = V.Resize((96, 64))
    transform = T.Compose([
        V.ToTensor(),
        V.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    for i, _ in enumerate(paths):
        im = Image.open(paths[i])
        im = resize(im)
        img = np.array(im)
        ts = transform(img)
        ll.append(ts[0])
        if len(ll) == batch_size:
            yield np.stack(ll, axis=0)
            ll.clear()
    if ll:
        yield np.stack(ll, axis=0)


def preprocess_data(args):
    """ preprocess data"""
    root_path = args.data_dir
    root_file_list = os.listdir(root_path)
    ims_info = []
    for sub_path in root_file_list:
        for im_path in os.listdir(os.path.join(root_path, sub_path)):
            ims_info.append((im_path.split('.')[0], os.path.join(root_path, sub_path, im_path)))

    paths = [path for name, path in ims_info]
    names = [name for name, path in ims_info]
    i = 0

    for img in load_images(paths):
        img = img.astype(np.float32)
        file_name = names[i] + ".bin"
        file_path = os.path.join(args.output_path, file_name)
        img.tofile(file_path)
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess data bin')
    parser.add_argument('--data_dir', type=str, default='', help='data dir, e.g. /home/test')
    parser.add_argument('--output_path', type=str, default='', help='output image path, e.g. /home/output')

    arg = parser.parse_args()

    preprocess_data(arg)
