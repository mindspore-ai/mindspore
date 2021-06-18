#!/usr/bin/env python3
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

import os
import random
import glob
import io

import numpy as np
import cv2
import PIL
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C

def create_train_dataset(data_path, model_type, noise_level=25, batch_size=128):
    # define dataset
    if model_type == "DnCNN-S":
        dataset = DnCNN_train_Dataset(data_path, model_type, batch_size, noise_level, patch_shape=(40, 40))
    if model_type in ["DnCNN-B", "DnCNN-3"]:
        dataset = DnCNN_train_Dataset(data_path, model_type, batch_size, patch_shape=(50, 50))

    print("total training patch numbers per epoch", len(dataset))
    dataloader = ds.GeneratorDataset(dataset, ["noisy", "gt"])
    # apply map operations on images
    dataloader = dataloader.map(input_columns="noisy", operations=C.TypeCast(mindspore.float32))
    dataloader = dataloader.map(input_columns="gt", operations=C.TypeCast(mindspore.float32))

    # apply DatasetOps
    dataloader = dataloader.shuffle(buffer_size=10000)
    dataloader = dataloader.batch(batch_size, drop_remainder=True)
    dataloader = dataloader.repeat(1) #here 400 images as an epoch , on the paper 128x1600 patches as a epoch
    return dataloader

class DnCNN_train_Dataset():
    def __init__(self, dataset_path, model_type, batch_size=128, noise_level=25, \
                 image_shape=(180, 180), patch_shape=(50, 50)):
        #DnCNN-S/B uses 200 training and 200 test images in BDS500 as training set
        #DnCNN-3 uses 200 training images in BDS500 and T91 images as training set
        self.im_list = []
        self.im_list.extend(glob.glob(os.path.join(dataset_path, "*/*.jpg")))
        self.im_list.extend(glob.glob(os.path.join(dataset_path, "*/*.bmp")))
        self.im_list.extend(glob.glob(os.path.join(dataset_path, "*/*.png")))
        self.patch_shape = patch_shape
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.model_type = model_type
        self.noise_level = noise_level
        self.scales = [0.8, 1, 1.25]
        self.sr_scales = [2, 3, 4]
        self.compression_range = (5, 100)
        self.sigma_range = (0, 55)

    def __getitem__(self, i):
        i = i % len(self.im_list)
        img = cv2.imread(self.im_list[i], 0)
        #random scale
        scale = random.choice(self.scales)
        img_h = int(self.image_shape[1]*scale)
        img_w = int(self.image_shape[0]*scale)
        img = cv2.resize(img, (img_w, img_h))

        #crop random patch
        start_w = random.randint(0, img_w - self.patch_shape[0])
        start_h = random.randint(0, img_h - self.patch_shape[1])
        patch = img[start_h:start_h+self.patch_shape[1], start_w:start_w+self.patch_shape[0]]

        #random flip & rotation
        patch = self.data_augment(patch)

        #add noise
        if self.model_type == "DnCNN-S":
            #add specific level of noise
            noisy = self.add_noise(patch, self.noise_level)
        elif self.model_type == "DnCNN-B":
            #add random level of noise
            sigma = random.uniform(*self.sigma_range)
            noisy = self.add_noise(patch, sigma)
        elif self.model_type == "DnCNN-3":
            #randomly add noise, down-up sampling blur and jpeg bloc
            noisy = self.add_random_type_noise(patch)
        else:
            print("wrong type")
            exit()
        noise = np.float32(noisy) - np.float32(patch)

        #add channel dimension
        noisy = noisy[np.newaxis, :, :]
        noise = noise[np.newaxis, :, :]

        #normalize
        noisy = noisy / 255.0
        noise = noise / 255.0

        return noisy, noise

    def __len__(self):
        #To build the same epoch size as original paper
        #on the paper, DnCNN-S has 1600 iteratisons per epoch
        #DnCNN-B has 3000 and DnCNN-3 has 8000
        if self.model_type == "DnCNN-S":
            epoch_size = self.batch_size * 1600
        elif self.model_type == "DnCNN-B":
            epoch_size = self.batch_size * 3000
        elif self.model_type == "DnCNN-3":
            epoch_size = self.batch_size * 8000
        else:
            print("wrong model type")
            exit()
        return epoch_size

    def data_augment(self, patch):
        #random flip & rotation
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        if random.random() < 0.5:
            patch = np.flipud(patch)
        alea = random.random()
        if alea > 0.25:
            patch = np.rot90(patch)
        elif alea > 0.5:
            patch = np.rot90(patch, k=2)
        elif alea > 0.75:
            patch = np.rot90(patch, k=3)
        return patch

    def add_random_type_noise(self, patch):
        #random noise/down-up resampling/JPEG compression
        alea = random.random()
        if alea < 0.33: #generate noisy image
            sigma = random.uniform(*self.sigma_range)
            noisy = self.add_noise(patch, sigma)
        elif alea < 0.66: #generate blur image
            sr_scale = random.choice(self.sr_scales)
            noisy = cv2.resize(patch, (int(self.patch_shape[0]/sr_scale), int(self.patch_shape[1]/sr_scale)))
            noisy = cv2.resize(noisy, (int(self.patch_shape[0]), int(self.patch_shape[1])))
        else: #generate JPEG blocking image
            compression_level = random.randint(*self.compression_range)
            noisy = self.jpeg_compression(patch, compression_level)
        return noisy

    def add_noise(self, im, sigma):
        gauss = np.random.normal(0, sigma, self.patch_shape)
        noisy = im + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype('uint8')
        return noisy

    def jpeg_compression(self, img, quality):
        im_pil = PIL.Image.fromarray(img)
        output = io.BytesIO()
        im_pil.save(output, 'JPEG', quality=quality)
        im_pil = PIL.Image.open(output)
        img_np = np.asarray(im_pil)
        return img_np


if __name__ == "__main__":
    #only for test
    test_dataset_path = "/code/BSR_bsds500/BSR/BSDS500/data/images/"
    ds_train = create_train_dataset(test_dataset_path, "DnCNN-S", batch_size=128)
    print("batch number:", ds_train.get_dataset_size())

    for data in ds_train.create_dict_iterator():
        print(type(data))
        print(data["noisy"].shape)
        print(data["gt"].shape)
        print(type(data["noisy"]))
        break
