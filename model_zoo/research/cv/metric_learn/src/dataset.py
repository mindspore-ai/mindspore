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
import mindspore.dataset as dss
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.dataset.vision import Inter

def create_dataset0(dataset_generator, do_train, batch_size=80, device_num=1, rank_id=0):
    """softmax dataset"""
    if device_num == 1:
        ds = dss.GeneratorDataset(dataset_generator, ["image", "label"], num_parallel_workers=8, shuffle=True)
    else:
        ds = dss.GeneratorDataset(dataset_generator, ["image", "label"], num_parallel_workers=8, shuffle=True,
                                  num_shards=device_num, shard_id=rank_id)
    trans = []
    if do_train:
        trans += [
            C.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3./4, 4./3), interpolation=Inter.BICUBIC)
        ]
    trans += [
        C.Resize((224, 224)),
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        C.HWC2CHW(),
        C2.TypeCast(mstype.float32)
    ]
    ds = ds.map(operations=trans, input_columns="image", num_parallel_workers=8)
    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

def create_dataset1(dataset_generator, do_train, batch_size=80, device_num=1, rank_id=0):
    """triplet/quadruplet dataset"""
    if device_num == 1:
        ds = dss.GeneratorDataset(dataset_generator, ["image", "label"], num_parallel_workers=8, shuffle=False)
    else:
        ds = dss.GeneratorDataset(dataset_generator, ["image", "label"], num_parallel_workers=8, shuffle=False,
                                  num_shards=device_num, shard_id=rank_id)
    trans = []
    if do_train:
        trans += [
            C.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3./4, 4./3), interpolation=Inter.BICUBIC)
        ]
    trans += [
        C.Resize((224, 224)),
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        C.HWC2CHW(),
        C2.TypeCast(mstype.float32)
    ]
    ds = ds.map(operations=trans, input_columns="image", num_parallel_workers=8)
    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds
