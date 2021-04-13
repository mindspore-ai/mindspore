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

import os
from collections import deque
import cv2
import numpy as np
from PIL import Image, ImageSequence
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.vision.utils import Inter
from mindspore.communication.management import get_rank, get_group_size


def _load_multipage_tiff(path):
    """Load tiff images containing many images in the channel dimension"""
    return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])

def _get_val_train_indices(length, fold, ratio=0.8):
    assert 0 < ratio <= 1, "Train/total data ratio must be in range (0.0, 1.0]"
    np.random.seed(0)
    indices = np.arange(0, length, 1, dtype=np.int)
    np.random.shuffle(indices)

    if fold is not None:
        indices = deque(indices)
        indices.rotate(fold * round((1.0 - ratio) * length))
        indices = np.array(indices)
        train_indices = indices[:round(ratio * len(indices))]
        val_indices = indices[round(ratio * len(indices)):]
    else:
        train_indices = indices
        val_indices = []
    return train_indices, val_indices

def data_post_process(img, mask):

    img = np.expand_dims(img, axis=0)
    mask = (mask > 0.5).astype(np.int)
    mask = (np.arange(mask.max() + 1) == mask[..., None]).astype(int)
    mask = mask.transpose(2, 0, 1).astype(np.float32)
    return img, mask


def train_data_augmentation(img, mask):

    h_flip = np.random.random()
    if h_flip > 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    v_flip = np.random.random()
    if v_flip > 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)

    left = int(np.random.uniform()*0.3*572)
    right = int((1-np.random.uniform()*0.3)*572)
    top = int(np.random.uniform()*0.3*572)
    bottom = int((1-np.random.uniform()*0.3)*572)


    img = img[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    #adjust brightness
    brightness = np.random.uniform(-0.2, 0.2)
    img = np.float32(img+brightness*np.ones(img.shape))
    img = np.clip(img, -1.0, 1.0)

    return img, mask


def create_dataset(data_dir, repeat=400, train_batch_size=16, augment=False, cross_val_ind=1, run_distribute=False,
                   do_crop=None, img_size=None):

    images = _load_multipage_tiff(os.path.join(data_dir, 'train-volume.tif'))
    masks = _load_multipage_tiff(os.path.join(data_dir, 'train-labels.tif'))

    train_indices, val_indices = _get_val_train_indices(len(images), cross_val_ind)
    train_images = images[train_indices]
    train_masks = masks[train_indices]
    train_images = np.repeat(train_images, repeat, axis=0)
    train_masks = np.repeat(train_masks, repeat, axis=0)
    val_images = images[val_indices]
    val_masks = masks[val_indices]

    train_image_data = {"image": train_images}
    train_mask_data = {"mask": train_masks}
    valid_image_data = {"image": val_images}
    valid_mask_data = {"mask": val_masks}


    ds_train_images = ds.NumpySlicesDataset(data=train_image_data, sampler=None, shuffle=False)
    ds_train_masks = ds.NumpySlicesDataset(data=train_mask_data, sampler=None, shuffle=False)

    if run_distribute:
        rank_id = get_rank()
        rank_size = get_group_size()
        ds_train_images = ds.NumpySlicesDataset(data=train_image_data,
                                                sampler=None,
                                                shuffle=False,
                                                num_shards=rank_size,
                                                shard_id=rank_id)
        ds_train_masks = ds.NumpySlicesDataset(data=train_mask_data,
                                               sampler=None,
                                               shuffle=False,
                                               num_shards=rank_size,
                                               shard_id=rank_id)

    ds_valid_images = ds.NumpySlicesDataset(data=valid_image_data, sampler=None, shuffle=False)
    ds_valid_masks = ds.NumpySlicesDataset(data=valid_mask_data, sampler=None, shuffle=False)

    if do_crop:
        resize_size = [int(img_size[x] * do_crop[x]) for x in range(len(img_size))]
    else:
        resize_size = img_size
    c_resize_op = c_vision.Resize(size=(resize_size[0], resize_size[1]), interpolation=Inter.BILINEAR)
    c_pad = c_vision.Pad(padding=(img_size[0] - resize_size[0]) // 2)
    c_rescale_image = c_vision.Rescale(1.0/127.5, -1)
    c_rescale_mask = c_vision.Rescale(1.0/255.0, 0)

    c_trans_normalize_img = [c_rescale_image, c_resize_op, c_pad]
    c_trans_normalize_mask = [c_rescale_mask, c_resize_op, c_pad]
    c_center_crop = c_vision.CenterCrop(size=388)

    train_image_ds = ds_train_images.map(input_columns="image", operations=c_trans_normalize_img)
    train_mask_ds = ds_train_masks.map(input_columns="mask", operations=c_trans_normalize_mask)
    train_ds = ds.zip((train_image_ds, train_mask_ds))
    train_ds = train_ds.project(columns=["image", "mask"])
    if augment:
        augment_process = train_data_augmentation
        c_resize_op = c_vision.Resize(size=(img_size[0], img_size[1]), interpolation=Inter.BILINEAR)
        train_ds = train_ds.map(input_columns=["image", "mask"], operations=augment_process)
        train_ds = train_ds.map(input_columns="image", operations=c_resize_op)
        train_ds = train_ds.map(input_columns="mask", operations=c_resize_op)

    if do_crop:
        train_ds = train_ds.map(input_columns="mask", operations=c_center_crop)
    post_process = data_post_process
    train_ds = train_ds.map(input_columns=["image", "mask"], operations=post_process)
    train_ds = train_ds.shuffle(repeat*24)
    train_ds = train_ds.batch(batch_size=train_batch_size, drop_remainder=True)

    valid_image_ds = ds_valid_images.map(input_columns="image", operations=c_trans_normalize_img)
    valid_mask_ds = ds_valid_masks.map(input_columns="mask", operations=c_trans_normalize_mask)
    valid_ds = ds.zip((valid_image_ds, valid_mask_ds))
    valid_ds = valid_ds.project(columns=["image", "mask"])
    if do_crop:
        valid_ds = valid_ds.map(input_columns="mask", operations=c_center_crop)
    post_process = data_post_process
    valid_ds = valid_ds.map(input_columns=["image", "mask"], operations=post_process)
    valid_ds = valid_ds.batch(batch_size=1, drop_remainder=True)

    return train_ds, valid_ds

class CellNucleiDataset:
    """
    Cell nuclei dataset preprocess class.
    """
    def __init__(self, data_dir, repeat, is_train=False, split=0.8):
        self.data_dir = data_dir
        self.img_ids = sorted(next(os.walk(self.data_dir))[1])
        self.train_ids = self.img_ids[:int(len(self.img_ids) * split)] * repeat
        np.random.shuffle(self.train_ids)
        self.val_ids = self.img_ids[int(len(self.img_ids) * split):]
        self.is_train = is_train
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        for img_id in self.img_ids:
            path = os.path.join(self.data_dir, img_id)
            if (not os.path.exists(os.path.join(path, "image.png"))) or \
                (not os.path.exists(os.path.join(path, "mask.png"))):
                img = cv2.imread(os.path.join(path, "images", img_id + ".png"))
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=-1)
                    img = np.concatenate([img, img, img], axis=-1)
                mask = []
                for mask_file in next(os.walk(os.path.join(path, "masks")))[2]:
                    mask_ = cv2.imread(os.path.join(path, "masks", mask_file), cv2.IMREAD_GRAYSCALE)
                    mask.append(mask_)
                mask = np.max(mask, axis=0)
                cv2.imwrite(os.path.join(path, "image.png"), img)
                cv2.imwrite(os.path.join(path, "mask.png"), mask)

    def _read_img_mask(self, img_id):
        path = os.path.join(self.data_dir, img_id)
        img = cv2.imread(os.path.join(path, "image.png"))
        mask = cv2.imread(os.path.join(path, "mask.png"), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def __getitem__(self, index):
        if self.is_train:
            return self._read_img_mask(self.train_ids[index])
        return self._read_img_mask(self.val_ids[index])

    @property
    def column_names(self):
        column_names = ['image', 'mask']
        return column_names

    def __len__(self):
        if self.is_train:
            return len(self.train_ids)
        return len(self.val_ids)

def preprocess_img_mask(img, mask, img_size, augment=False, eval_resize=False):
    """
    Preprocess for cell nuclei dataset.
    Random crop and flip images and masks when augment is True.
    """
    if augment:
        img_size_w = int(np.random.randint(img_size[0], img_size[0] * 1.5, 1))
        img_size_h = int(np.random.randint(img_size[1], img_size[1] * 1.5, 1))
        img = cv2.resize(img, (img_size_w, img_size_h))
        mask = cv2.resize(mask, (img_size_w, img_size_h))
        dw = int(np.random.randint(0, img_size_w - img_size[0] + 1, 1))
        dh = int(np.random.randint(0, img_size_h - img_size[1] + 1, 1))
        img = img[dh:dh+img_size[1], dw:dw+img_size[0], :]
        mask = mask[dh:dh+img_size[1], dw:dw+img_size[0]]
        if np.random.random() > 0.5:
            flip_code = int(np.random.randint(-1, 2, 1))
            img = cv2.flip(img, flip_code)
            mask = cv2.flip(mask, flip_code)
    else:
        img = cv2.resize(img, img_size)
        if not eval_resize:
            mask = cv2.resize(mask, img_size)
    img = (img.astype(np.float32) - 127.5) / 127.5
    img = img.transpose(2, 0, 1)
    mask = mask.astype(np.float32) / 255
    mask = (mask > 0.5).astype(np.int)
    mask = (np.arange(2) == mask[..., None]).astype(int)
    mask = mask.transpose(2, 0, 1).astype(np.float32)
    return img, mask

def create_cell_nuclei_dataset(data_dir, img_size, repeat, batch_size, is_train=False, augment=False, eval_resize=False,
                               split=0.8, rank=0, group_size=1, python_multiprocessing=True, num_parallel_workers=8):
    """
    Get generator dataset for cell nuclei dataset.
    """
    cell_dataset = CellNucleiDataset(data_dir, repeat, is_train, split)
    sampler = ds.DistributedSampler(group_size, rank, shuffle=is_train)
    dataset = ds.GeneratorDataset(cell_dataset, cell_dataset.column_names, sampler=sampler)
    compose_map_func = (lambda image, mask: preprocess_img_mask(image, mask, tuple(img_size), augment and is_train,
                                                                eval_resize))
    dataset = dataset.map(operations=compose_map_func, input_columns=cell_dataset.column_names,
                          output_columns=cell_dataset.column_names, column_order=cell_dataset.column_names,
                          python_multiprocessing=python_multiprocessing,
                          num_parallel_workers=num_parallel_workers)
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    dataset = dataset.repeat(1)
    return dataset
