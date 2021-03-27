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
"""
create train or eval dataset.
"""
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.vision import Inter
import cv2
import numpy as np

ds.config.set_seed(0)

def gaussian_blur(im):
    sigma = 0
    _, w = im.shape[:2]
    kernel_size = int(w // 10)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    return np.array(cv2.GaussianBlur(im, (kernel_size, kernel_size), sigma))

def copy_column(x, y):
    return x, x, y

def create_dataset(args, dataset_mode, repeat_num=1):
    """
    create a train or evaluate cifar10 dataset for resnet50
    """
    if args.dataset_name != "cifar10":
        raise ValueError("Unsupported dataset.")
    if dataset_mode in ("train_endcoder", "train_classifier"):
        dataset_path = args.train_dataset_path
    else:
        dataset_path = args.eval_dataset_path
    if args.run_distribute and args.device_target == "Ascend":
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                     num_shards=args.device_num, shard_id=args.device_id)
    else:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)
    # define map operations
    trans = []
    if dataset_mode == "train_endcoder":
        if args.use_crop:
            trans += [C.Resize(256, interpolation=Inter.BICUBIC)]
            trans += [C.RandomResizedCrop(size=(32, 32), scale=(0.31, 1),
                                          interpolation=Inter.BICUBIC, max_attempts=100)]
        if args.use_flip:
            trans += [C.RandomHorizontalFlip(prob=0.5)]
        if args.use_color_jitter:
            scale = 0.6
            color_jitter = C.RandomColorAdjust(0.8 * scale, 0.8 * scale, 0.8 * scale, 0.2 * scale)
            trans += [C2.RandomApply([color_jitter], prob=0.8)]
        if args.use_color_gray:
            trans += [py_vision.ToPIL(),
                      py_vision.RandomGrayscale(prob=0.2),
                      np.array]  # need to convert PIL image to a NumPy array to pass it to C++ operation
        if args.use_blur:
            trans += [C2.RandomApply([gaussian_blur], prob=0.8)]
        if args.use_norm:
            trans += [C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
        trans += [C2.TypeCast(mstype.float32), C.HWC2CHW()]
    else:
        trans += [C.Resize(32)]
        trans += [C2.TypeCast(mstype.float32)]
        if args.use_norm:
            trans += [C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
        trans += [C.HWC2CHW()]
    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.map(operations=copy_column, input_columns=["image", "label"],
                            output_columns=["image1", "image2", "label"],
                            column_order=["image1", "image2", "label"], num_parallel_workers=8)
    data_set = data_set.map(operations=trans, input_columns=["image1"], num_parallel_workers=8)
    data_set = data_set.map(operations=trans, input_columns=["image2"], num_parallel_workers=8)
    # apply batch operations
    data_set = data_set.batch(args.batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)
    return data_set
