# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Prepare Cityscapes dataset"""
import os
import numpy as np
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV

from src.seg_data_base import SegmentationDataset
from src.distributed_sampler import DistributedSampler


__all__ = ['CitySegmentation']


class CitySegmentation(SegmentationDataset):
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/cityscapes'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19

    def __init__(self, args, root='/data/Fast_SCNN/dataset/', split='train', mode=None, **kwargs):
        super(CitySegmentation, self).__init__(root, split, mode, **kwargs)

        self.args = args
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/cityscapes"
        self.images, self.mask_paths = _get_city_pairs(args, self.root, self.split)
        assert len(self.images) == len(self.mask_paths)
        if not self.images:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert value in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            return img, os.path.basename(self.images[index])

        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        return img, mask

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return np.array(target).astype('int32')

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle')

def _get_city_pairs(args, folder, split='train'):
    '''_get_city_pairs'''
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.startswith('._'):
                    continue
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        args.logger.info('cannot find the mask or image:', imgpath, maskpath)
        args.logger.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    assert split == 'trainval'
    args.logger.info('trainval set')
    train_img_folder = os.path.join(folder, 'leftImg8bit/train')
    train_mask_folder = os.path.join(folder, 'gtFine/train')
    val_img_folder = os.path.join(folder, 'leftImg8bit/val')
    val_mask_folder = os.path.join(folder, 'gtFine/val')
    train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
    val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
    img_paths = train_img_paths + val_img_paths
    mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths

def create_CitySegmentation(args, data_path='../dataset/', split='train', mode=None, \
                            transform=None, base_size=1024, crop_size=(512, 1024), \
                            batch_size=2, device_num=1, rank=0, shuffle=True):
    '''create_CitySegmentation'''
    dataset = CitySegmentation(args, root=data_path, split=split, mode=mode, \
                               base_size=base_size, crop_size=crop_size)
    dataset_len = len(dataset)
    distributed_sampler = DistributedSampler(dataset_len, device_num, rank, shuffle=shuffle)

    data_set = ds.GeneratorDataset(dataset, column_names=["image", "label"], num_parallel_workers=8, \
                                   shuffle=shuffle, sampler=distributed_sampler)
    # general resize, normalize and toTensor
    if transform is not None:
        data_set = data_set.map(input_columns=["image"], operations=transform, num_parallel_workers=8)
    else:
        hwc_to_chw = CV.HWC2CHW()
        data_set = data_set.map(input_columns=["image"], operations=hwc_to_chw, num_parallel_workers=8)

    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set, dataset_len

if __name__ == '__main__':

    ds = create_CitySegmentation(
        data_path=r"./dataset/leftImg8bit_trainvaltest",
        split='train',
        mode='train',
        base_size=1024,
        crop_size=(512, 1024),
        batch_size=1,
        device_num=1,
        rank=0,
        shuffle=True)
    data_loader = ds.create_dict_iterator()
    for i, data in enumerate(data_loader):
        print(data['image'])
        print(data['label'])
        print(i)
