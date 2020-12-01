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
"""Face Recognition dataset."""
import sys
import warnings
from PIL import ImageFile

from mindspore import dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as VC
import mindspore.dataset.transforms.c_transforms as C

sys.path.append('./')
sys.path.append('../data/')
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

def get_de_dataset(args):
    '''Get de_dataset.'''
    transform_label = [C.TypeCast(mstype.int32)]
    transform_img = [VC.Decode(),
                     VC.Resize((96, 64)),
                     VC.RandomColorAdjust(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                     VC.RandomHorizontalFlip(),
                     VC.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
                     VC.HWC2CHW()]

    de_dataset = de.ImageFolderDataset(dataset_dir=args.data_dir, num_shards=args.world_size,
                                       shard_id=args.local_rank, shuffle=True)
    de_dataset = de_dataset.map(input_columns="image", operations=transform_img)
    de_dataset = de_dataset.map(input_columns="label", operations=transform_label)
    de_dataset = de_dataset.project(columns=["image", "label"])
    de_dataset = de_dataset.batch(args.per_batch_size, drop_remainder=True)

    num_iter_per_gpu = de_dataset.get_dataset_size()
    de_dataset = de_dataset.repeat(args.max_epoch)
    num_classes = de_dataset.num_classes()

    return de_dataset, num_iter_per_gpu, num_classes
