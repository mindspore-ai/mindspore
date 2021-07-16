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
"""eval midas."""
import glob
import csv
import os
import struct
import json
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.train import serialization
import mindspore.ops as ops
from src.util import depth_read_kitti, depth_read_sintel, BadPixelMetric
from src.midas_net import MidasNet
from src.config import config
from src.utils import transforms
from scipy.io import loadmat
import cv2
from PIL import Image
import h5py


def eval_Kitti(data_path, net):
    """eval Kitti."""
    img_input_1 = transforms.Resize(config.img_width,
                                    config.img_height,
                                    resize_target=None,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=32,
                                    resize_method="lower_bound",
                                    image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = transforms.NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = transforms.PrepareForNet()
    metric = BadPixelMetric(1.25, 80, 'KITTI')
    loss_sum = 0
    sample = {}
    image_path = glob.glob(os.path.join(data_path, '*', 'image', '*.png'))
    num = 0
    for file_name in image_path:
        num += 1
        print(f"processing: {num} / {len(image_path)}")
        image = np.array(Image.open(file_name)).astype(float)  # (436,1024,3)
        image = image / 255
        print(file_name)
        all_path = file_name.split('/')
        depth_path_name = all_path[-1].split('.')[0]

        depth = depth_read_kitti(os.path.join(data_path, all_path[-3], 'depth', depth_path_name + '.png'))  # (436,1024)
        mask = (depth > 0) & (depth < 80)
        sample['image'] = image
        sample["depth"] = depth
        sample["mask"] = mask
        sample = img_input_1(sample)
        sample = img_input_2(sample)
        sample = img_input_3(sample)
        # print('transform later', sample['image'].shape)
        sample['image'] = Tensor([sample["image"]], mstype.float32)
        sample['depth'] = Tensor([sample["depth"]], mstype.float32)
        sample['mask'] = Tensor([sample["mask"]], mstype.int32)

        print(sample['image'].shape, sample['depth'].shape)
        prediction = net(sample['image'])

        mask = sample['mask'].asnumpy()
        depth = sample['depth'].asnumpy()

        expand_dims = ops.ExpandDims()
        prediction = expand_dims(prediction, 0)
        resize_bilinear = ops.ResizeBilinear(mask.shape[1:])
        prediction = resize_bilinear(prediction)
        prediction = np.squeeze(prediction.asnumpy())
        loss = metric(prediction, depth, mask)

        print('loss is ', loss)
        loss_sum += loss

    print(f"Kitti bad pixel: {loss_sum / num:.3f}")
    return loss_sum / num


def eval_TUM(datapath, net):
    """eval TUM."""
    img_input_1 = transforms.Resize(config.img_width,
                                    config.img_height,
                                    resize_target=None,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=32,
                                    resize_method="upper_bound",
                                    image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = transforms.NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = transforms.PrepareForNet()
    # get data
    metric = BadPixelMetric(1.25, 10, 'TUM')
    loss_sum = 0
    sample = {}
    file_path = glob.glob(os.path.join(datapath, '*_person', 'associate.txt'))

    num = 0
    for ind in file_path:
        all_path = ind.split('/')

        for line in open(ind):
            num += 1
            print(f"processing: {num}")
            data = line.split('\n')[0].split(' ')
            image_path = os.path.join(datapath, all_path[-2], data[0])  # (480,640,3)
            depth_path = os.path.join(datapath, all_path[-2], data[1])  # (480,640,3)
            image = cv2.imread(image_path) / 255
            depth = cv2.imread(depth_path)[:, :, 0] / 5000
            mask = (depth > 0) & (depth < 10)
            print('mask is ', np.unique(mask))
            sample['image'] = image
            sample["depth"] = depth
            sample["mask"] = mask

            sample = img_input_1(sample)
            sample = img_input_2(sample)
            sample = img_input_3(sample)

            sample['image'] = Tensor([sample["image"]], mstype.float32)
            sample['depth'] = Tensor([sample["depth"]], mstype.float32)
            sample['mask'] = Tensor([sample["mask"]], mstype.int32)

            print(sample['image'].shape, sample['depth'].shape)
            prediction = net(sample['image'])
            mask = sample['mask'].asnumpy()
            depth = sample['depth'].asnumpy()
            expand_dims = ops.ExpandDims()
            prediction = expand_dims(prediction, 0)
            print(prediction.shape, mask.shape)
            resize_bilinear = ops.ResizeBilinear(mask.shape[1:])
            prediction = resize_bilinear(prediction)
            prediction = np.squeeze(prediction.asnumpy())

            loss = metric(prediction, depth, mask)

            print('loss is ', loss)
            loss_sum += loss

    print(f"TUM bad pixel: {loss_sum / num:.2f}")

    return loss_sum / num


def eval_Sintel(datapath, net):
    """eval Sintel."""
    img_input_1 = transforms.Resize(config.img_width,
                                    config.img_height,
                                    resize_target=None,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=32,
                                    resize_method="upper_bound",
                                    image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = transforms.NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = transforms.PrepareForNet()
    # get data
    metric = BadPixelMetric(1.25, 72, 'sintel')
    loss_sum = 0
    sample = {}
    image_path = glob.glob(os.path.join(datapath, 'final_left', '*', '*.png'))

    num = 0
    for file_name in image_path:
        num += 1
        print(f"processing: {num} / {len(image_path)}")
        image = np.array(Image.open(file_name)).astype(float)  # (436,1024,3)
        image = image / 255
        print(file_name)
        all_path = file_name.split('/')
        depth_path_name = all_path[-1].split('.')[0]

        depth = depth_read_sintel(os.path.join(datapath, 'depth', all_path[-2], depth_path_name + '.dpt'))  # (436,1024)

        mask1 = np.array(Image.open(os.path.join(datapath, 'occlusions', all_path[-2], all_path[-1]))).astype(int)
        mask1 = mask1 / 255

        mask = (mask1 == 1) & (depth > 0) & (depth < 72)
        sample['image'] = image
        sample["depth"] = depth
        sample["mask"] = mask
        sample = img_input_1(sample)
        sample = img_input_2(sample)
        sample = img_input_3(sample)
        sample['image'] = Tensor([sample["image"]], mstype.float32)
        sample['depth'] = Tensor([sample["depth"]], mstype.float32)
        sample['mask'] = Tensor([sample["mask"]], mstype.int32)

        print(sample['image'].shape, sample['depth'].shape)
        prediction = net(sample['image'])

        mask = sample['mask'].asnumpy()
        depth = sample['depth'].asnumpy()

        expand_dims = ops.ExpandDims()
        prediction = expand_dims(prediction, 0)
        resize_bilinear = ops.ResizeBilinear(mask.shape[1:])
        prediction = resize_bilinear(prediction)
        prediction = np.squeeze(prediction.asnumpy())
        loss = metric(prediction, depth, mask)

        print('loss is ', loss)
        loss_sum += loss

    print(f"sintel bad pixel: {loss_sum / len(image_path):.3f}")
    return loss_sum / len(image_path)


def eval_ETH3D(datapath, net):
    """eval ETH3D."""
    img_input_1 = transforms.Resize(config.img_width,
                                    config.img_height,
                                    resize_target=True,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=32,
                                    resize_method="upper_bound",
                                    image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = transforms.NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = transforms.PrepareForNet()
    metric = BadPixelMetric(1.25, 72, 'ETH3D')

    loss_sum = 0
    sample = {}
    image_path = glob.glob(os.path.join(datapath, '*', 'images', 'dslr_images', '*.JPG'))
    num = 0
    for file_name in image_path:
        num += 1
        print(f"processing: {num} / {len(image_path)}")
        image = cv2.imread(file_name) / 255
        all_path = file_name.split('/')
        depth_path = os.path.join(datapath, all_path[-4], "ground_truth_depth", 'dslr_images', all_path[-1])
        depth = []
        with open(depth_path, 'rb') as f:
            data = f.read(4)
            while data:
                depth.append(struct.unpack('f', data))
                data = f.read(4)
            depth = np.reshape(np.array(depth), (4032, -1))
        mask = (depth > 0) & (depth < 72)
        sample['image'] = image
        sample["depth"] = depth
        sample["mask"] = mask

        sample = img_input_1(sample)
        sample = img_input_2(sample)
        sample = img_input_3(sample)
        sample['image'] = Tensor([sample["image"]], mstype.float32)
        sample['depth'] = Tensor([sample["depth"]], mstype.float32)
        sample['mask'] = Tensor([sample["mask"]], mstype.int32)

        prediction = net(sample['image'])

        mask = sample['mask'].asnumpy()
        depth = sample['depth'].asnumpy()

        expand_dims = ops.ExpandDims()
        prediction = expand_dims(prediction, 0)
        resize_bilinear = ops.ResizeBilinear(mask.shape[1:])
        prediction = resize_bilinear(prediction)
        prediction = np.squeeze(prediction.asnumpy())
        loss = metric(prediction, depth, mask)

        print('loss is ', loss)
        loss_sum += loss

    print(f"ETH3D bad pixel: {loss_sum / num:.3f}")

    return loss_sum / num


def eval_DIW(datapath, net):
    """eval DIW."""
    img_input_1 = transforms.Resize(config.img_width,
                                    config.img_height,
                                    resize_target=True,
                                    keep_aspect_ratio=True,
                                    ensure_multiple_of=32,
                                    resize_method="upper_bound",
                                    image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = transforms.NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = transforms.PrepareForNet()
    loss_sum = 0
    num = 0
    sample = {}
    file_path = os.path.join(datapath, 'DIW_Annotations', 'DIW_test.csv')
    with open(file_path) as f:
        reader = list(csv.reader(f))
        for (i, row) in enumerate(reader):
            if i % 2 == 0:
                path = row[0].split('/')
                sample['file_name'] = os.path.join(datapath, path[-2], path[-1])
                sample['image'] = cv2.imread(sample['file_name']) / 255
            else:
                sample['depths'] = row
                if not os.path.exists(sample['file_name']):
                    continue
                num += 1  # 图片个数+1
                print(f"processing: {num}")
                sample = img_input_1(sample)
                sample = img_input_2(sample)
                sample = img_input_3(sample)
                sample['image'] = Tensor([sample["image"]], mstype.float32)
                prediction = net(sample['image'])
                shape_w, shape_h = [int(sample['depths'][-2]), int(sample['depths'][-1])]
                expand_dims = ops.ExpandDims()
                prediction = expand_dims(prediction, 0)
                resize_bilinear = ops.ResizeBilinear((shape_h, shape_w))
                prediction = resize_bilinear(prediction)
                prediction = np.squeeze(prediction.asnumpy())

                pixtel_a = prediction[int(sample['depths'][0]) - 1][int(sample['depths'][1]) - 1]
                pixtel_b = prediction[int(sample['depths'][2]) - 1][int(sample['depths'][3]) - 1]
                if pixtel_a > pixtel_b:
                    if sample['depths'][4] == '>':
                        loss_sum += 1
                if pixtel_a < pixtel_b:
                    if sample['depths'][4] == '<':
                        loss_sum += 1
                print(f"bad pixel: {(num - loss_sum) / num:.4f}")
    return (num - loss_sum) / num


def eval_NYU(datamat, splitmat, net):
    """eval NYU."""
    img_input_1 = Resize(config.img_width,
                         config.img_height,
                         resize_target=None,
                         keep_aspect_ratio=True,
                         ensure_multiple_of=32,
                         resize_method="upper_bound",
                         image_interpolation_method=cv2.INTER_CUBIC)
    img_input_2 = NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
    img_input_3 = PrepareForNet()

    # get data

    metric = BadPixelMetric(1.25, 10, 'NYU')
    loss_sum = 0
    sample = {}
    mat = loadmat(splitmat)
    indices = [ind[0] - 1 for ind in mat["testNdxs"]]
    num = 0
    with h5py.File(datamat, "r") as f:
        for ind in indices:
            num += 1
            print(num)
            image = np.swapaxes(f["images"][ind], 0, 2)
            image = image / 255
            depth = np.swapaxes(f["rawDepths"][ind], 0, 1)
            mask = (depth > 0) & (depth < 10)

            # mask = mask1
            sample['image'] = image
            sample["depth"] = depth
            sample["mask"] = mask
            sample = img_input_1(sample)
            sample = img_input_2(sample)
            sample = img_input_3(sample)
            sample['image'] = Tensor([sample["image"]], mstype.float32)
            sample['depth'] = Tensor([sample["depth"]], mstype.float32)
            sample['mask'] = Tensor([sample["mask"]], mstype.int32)

            print(sample['image'].shape, sample['depth'].shape)
            prediction = net(sample['image'])

            mask = sample['mask'].asnumpy()
            depth = sample['depth'].asnumpy()

            expand_dims = ops.ExpandDims()
            prediction = expand_dims(prediction, 0)
            resize_bilinear = ops.ResizeBilinear(mask.shape[1:])
            prediction = resize_bilinear(prediction)
            prediction = np.squeeze(prediction.asnumpy())
            loss = metric(prediction, depth, mask)

            print('loss is ', loss)
            loss_sum += loss

    print(f"bad pixel: {loss_sum / num:.3f}")
    return loss_sum / num


def run_eval():
    """run."""
    datapath_TUM = config.train_data_dir+config.datapath_TUM
    datapath_Sintel = config.train_data_dir+config.datapath_Sintel
    datapath_ETH3D = config.train_data_dir+config.datapath_ETH3D
    datapath_Kitti = config.train_data_dir+config.datapath_Kitti
    datapath_DIW = config.train_data_dir+config.datapath_DIW
    datamat = config.train_data_dir+config.datapath_NYU[0]
    splitmat = config.train_data_dir+config.datapath_NYU[1]

    net = MidasNet()
    param_dict = serialization.load_checkpoint(config.train_data_dir+config.ckpt_path)
    serialization.load_param_into_net(net, param_dict)
    results = {}
    if config.data_name == 'Sintel' or config.data_name == "all":
        result_sintel = eval_Sintel(datapath_Sintel, net)
        results['Sintel'] = result_sintel
    if config.data_name == 'Kitti' or config.data_name == "all":
        result_kitti = eval_Kitti(datapath_Kitti, net)
        results['Kitti'] = result_kitti
    if config.data_name == 'TUM' or config.data_name == "all":
        result_tum = eval_TUM(datapath_TUM, net)
        results['TUM'] = result_tum
    if config.data_name == 'DIW' or config.data_name == "all":
        result_DIW = eval_DIW(datapath_DIW, net)
        results['DIW'] = result_DIW
    if config.data_name == 'ETH3D' or config.data_name == "all":
        result_ETH3D = eval_ETH3D(datapath_ETH3D, net)
        results['ETH3D'] = result_ETH3D
    if config.data_name == 'NYU' or config.data_name == "all":
        result_NYU = eval_NYU(datamat, splitmat, net)
        results['NYU'] = result_NYU

    print(results)
    json.dump(results, open(config.ann_file, 'w'))


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)
    run_eval()
