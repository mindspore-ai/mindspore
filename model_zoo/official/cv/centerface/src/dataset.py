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
"""generate dataloader and data processing entry"""

import mindspore.dataset as ds

from src.utils import DistributedSampler

from dependency.centernet.src.lib.datasets.dataset.coco_hp import CenterfaceDataset
from dependency.centernet.src.lib.datasets.sample.multi_pose import preprocess_train

def GetDataLoader(per_batch_size,
                  max_epoch,
                  rank,
                  group_size,
                  config,
                  split='train'):
    """
    Centerface get data loader
    """
    centerface_gen = CenterfaceDataset(config=config, split=split)
    sampler = DistributedSampler(centerface_gen, rank, group_size, shuffle=(split == 'train')) # user defined sampling strategy
    de_dataset = ds.GeneratorDataset(centerface_gen, ["image", "anns"], sampler=sampler, num_parallel_workers=16)

    if group_size > 1:
        num_parallel_workers = 24
    else:
        num_parallel_workers = 64
    if split == 'train':
        compose_map_func = (lambda image, anns: preprocess_train(image, anns, config=config))
        columns = ['image', "hm", 'reg_mask', 'ind', 'wh', 'wight_mask', 'hm_offset', 'hps_mask', 'landmarks']
        de_dataset = de_dataset.map(input_columns=["image", "anns"],
                                    output_columns=columns,
                                    column_order=columns,
                                    operations=compose_map_func,
                                    num_parallel_workers=num_parallel_workers,
                                    python_multiprocessing=True)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=True, num_parallel_workers=8)
    if split == 'train':
        #de_dataset = de_dataset.repeat(1) # if use this, need an additional "for" cycle epoch times
        de_dataset = de_dataset.repeat(max_epoch)

    return de_dataset, de_dataset.get_dataset_size()
