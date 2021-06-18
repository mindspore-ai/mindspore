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
import io
import argparse
import glob

import PIL
import numpy as np
import cv2
import skimage.metrics
import mindspore
import mindspore.dataset as ds
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.transforms.c_transforms as C
from src.model import DnCNN

class DnCNN_eval_Dataset():
    def __init__(self, dataset_path, task_type, noise_level):
        self.im_list = []
        self.im_list.extend(glob.glob(os.path.join(dataset_path, "*.png")))
        self.im_list.extend(glob.glob(os.path.join(dataset_path, "*.bmp")))
        self.im_list.extend(glob.glob(os.path.join(dataset_path, "*.jpg")))
        self.task_type = task_type
        self.noise_level = noise_level

    def __getitem__(self, i):
        img = cv2.imread(self.im_list[i], 0)

        if self.task_type == "denoise":
            noisy = self.add_noise(img, self.noise_level)
        elif self.task_type == "super-resolution":
            h, w = img.shape
            noisy = cv2.resize(img, (int(w/self.noise_level), int(h/self.noise_level)))
            noisy = cv2.resize(noisy, (w, h))
        elif self.task_type == "jpeg-deblock":
            noisy = self.jpeg_compression(img, self.noise_level)

        #add channel dimension
        noisy = noisy[np.newaxis, :, :]
        noisy = noisy / 255.0
        return noisy, img

    def __len__(self):
        return len(self.im_list)

    def add_noise(self, im, sigma):
        gauss = np.random.normal(0, sigma, im.shape)
        noisy = im + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype('float32')
        return noisy

    def jpeg_compression(self, img, severity):
        im_pil = PIL.Image.fromarray(img)
        output = io.BytesIO()
        im_pil.save(output, 'JPEG', quality=severity)
        im_pil = PIL.Image.open(output)
        img_np = np.asarray(im_pil)
        return img_np


def create_eval_dataset(data_path, task_type, noise_level, batch_size=1):
    # define dataset
    dataset = DnCNN_eval_Dataset(data_path, task_type, noise_level)
    dataloader = ds.GeneratorDataset(dataset, ["noisy", "clear"])
    # apply map operations on images
    dataloader = dataloader.map(input_columns="noisy", operations=C.TypeCast(mindspore.float32))
    dataloader = dataloader.map(input_columns="clear", operations=C.TypeCast(mindspore.uint8))
    dataloader = dataloader.batch(batch_size, drop_remainder=False)
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DnCNN")
    parser.add_argument("--dataset_path", type=str, default="/code/12imgs-TestingSet/", help='eval image path')
    parser.add_argument('--ckpt_path', type=str, default=False, help='trained ckpt')
    parser.add_argument('--model_type', type=str, default='DnCNN-S', \
                        choices=['DnCNN-S', 'DnCNN-B', 'DnCNN-3'], help='type of DnCNN')
    parser.add_argument('--noise_type', type=str, default=False, \
                        choices=["denoise", "super-resolution", "jpeg-deblock"], help='trained ckpt')
    parser.add_argument('--noise_level', type=int, default=False, help='trained ckpt')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ds_eval = create_eval_dataset(args.dataset_path, args.noise_type, args.noise_level, batch_size=1)
    print("evaluation image number:", ds_eval.get_dataset_size())

    if args.model_type == 'DnCNN-S':
        network = DnCNN(1, num_of_layers=17)
    elif args.model_type == 'DnCNN-3' or args.model_type == 'DnCNN-B':
        network = DnCNN(1, num_of_layers=20)
    else:
        print("wrong model type")
        exit()

    # load parameter to the network
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)

    mean_psnr = 0
    mean_ssim = 0
    count = 0
    for data in ds_eval.create_dict_iterator():
        clear = data["clear"].asnumpy()
        #get denoised image
        residual = network(data["noisy"]).asnumpy() * 255
        noisy_img = data["noisy"].asnumpy() * 255
        denoised = np.clip(noisy - residual, 0, 255).astype("uint8")
        denoised = np.squeeze(denoised)
        clear = np.squeeze(clear)
        noisy_img = np.squeeze(noisy_img)
        if count == 0: #save example result
            cv2.imwrite("noisy.jpg", noisy_img.astype("uint8"))
            cv2.imwrite("denoised.jpg", denoised)
            cv2.imwrite("original.jpg", clear)

        #calculate psnr
        mse = np.mean((clear - denoised)**2)
        psnr = 10*np.log10(255*255/mse)
        #calculate ssim
        ssim = skimage.metrics.structural_similarity(clear, denoised, data_range=255) #skimage 0.18

        mean_psnr += psnr
        mean_ssim += ssim
        count += 1

    mean_psnr = mean_psnr / count
    mean_ssim = mean_ssim / count
    print("mean psnr", mean_psnr)
    print("mean_ssim", mean_ssim)
