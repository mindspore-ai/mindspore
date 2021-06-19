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
"""pre process for 310 inference"""
import os
import numpy as np
from PIL import Image
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset as de
from model_utils.config import config


class SingleScaleTrans_Infer:
    '''SingleScaleTrans'''

    def __init__(self, resize, max_anno_count=200):
        self.resize = (resize[0], resize[1])
        self.max_anno_count = max_anno_count

    def __call__(self, imgs, ann, image_names, image_size, batch_info):

        decode = P.Decode()
        ret_imgs = []
        ret_anno = []

        for i, image in enumerate(imgs):
            img_pil = decode(image)
            input_data = img_pil, ann[i]
            ret_imgs.append(np.array(input_data[0]))
            ret_anno.append(input_data[1])

        for i, anno in enumerate(ret_anno):
            anno_count = anno.shape[0]
            if anno_count < self.max_anno_count:
                ret_anno[i] = np.concatenate(
                    (ret_anno[i], np.zeros((self.max_anno_count - anno_count, 6), dtype=float)), axis=0)
            else:
                ret_anno[i] = ret_anno[i][:self.max_anno_count]

        return np.array(ret_imgs), np.array(ret_anno), image_names, image_size


def preprocess():
    """preprocess"""
    preprocess_path = config.preprocess_path
    images_path = os.path.join(preprocess_path, 'images')
    if not os.path.isdir(images_path):
        os.makedirs(images_path, exist_ok=True)

    labels_path = os.path.join(preprocess_path, 'labels')
    if not os.path.isdir(labels_path):
        os.makedirs(labels_path, exist_ok=True)

    image_name_path = os.path.join(preprocess_path, 'image_name')
    if not os.path.isdir(image_name_path):
        os.makedirs(image_name_path, exist_ok=True)
    image_size_path = os.path.join(preprocess_path, 'image_size')
    if not os.path.isdir(image_size_path):
        os.makedirs(image_size_path, exist_ok=True)

    ds = de.MindDataset(os.path.join(config.mindrecord_path, "data.mindrecord0"),
                        columns_list=["image", "annotation", "image_name", "image_size"])
    single_scale_trans = SingleScaleTrans_Infer(resize=config.input_shape)
    ds = ds.batch(config.batch_size, per_batch_map=single_scale_trans,
                  input_columns=["image", "annotation", "image_name", "image_size"], num_parallel_workers=8)
    ds = ds.repeat(1)
    for data in ds.create_tuple_iterator(output_numpy=True):
        images, labels, image_name, image_size = data[0:4]
        images = Image.fromarray(images[0].astype('uint8')).convert('RGB')
        images.save(os.path.join(images_path, image_name[0].decode() + ".jpg"))
        labels.tofile(os.path.join(labels_path, image_name[0].decode() + ".bin"))
        image_name.tofile(os.path.join(image_name_path, image_name[0].decode() + ".bin"))
        image_size.tofile(os.path.join(image_size_path, image_name[0].decode() + ".bin"))


if __name__ == '__main__':
    preprocess()
