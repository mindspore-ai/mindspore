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
import glob
import numpy as np
import PIL.Image as Image
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
from src.distributed_sampler import DistributedSampler

class BRDNetDataset:
    """ BRDNetDataset.
    Args:
        data_path: path of images, must end by '/'
        sigma: noise level
        channel: 3 for color, 1 for gray
    """
    def __init__(self, data_path, sigma, channel):
        images = []
        file_dictory = glob.glob(data_path+'*.bmp') #notice the data format
        for file in file_dictory:
            images.append(file)
        self.images = images
        self.sigma = sigma
        self.channel = channel
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            get_batch_x: image with noise
            get_batch_y: image without noise
        """
        if self.channel == 3:
            get_batch_y = np.array(Image.open(self.images[index]), dtype='uint8')
        else:
            get_batch_y = np.expand_dims(np.array(Image.open(self.images[index]).convert('L'), dtype='uint8'), axis=2)

        get_batch_y = get_batch_y.astype('float32')/255.0
        noise = np.random.normal(0, self.sigma/255.0, get_batch_y.shape).astype('float32')    # noise
        get_batch_x = get_batch_y + noise  # input image = clean image + noise
        return get_batch_x, get_batch_y

    def __len__(self):
        return len(self.images)

def create_BRDNetDataset(data_path, sigma, channel, batch_size, device_num, rank, shuffle):

    dataset = BRDNetDataset(data_path, sigma, channel)
    dataset_len = len(dataset)
    distributed_sampler = DistributedSampler(dataset_len, device_num, rank, shuffle=shuffle)
    hwc_to_chw = CV.HWC2CHW()
    data_set = ds.GeneratorDataset(dataset, column_names=["image", "label"], \
               shuffle=shuffle, sampler=distributed_sampler)
    data_set = data_set.map(input_columns=["image"], operations=hwc_to_chw, num_parallel_workers=8)
    data_set = data_set.map(input_columns=["label"], operations=hwc_to_chw, num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set, dataset_len
