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
"""Face Recognition infer preprocess."""
import os
import shutil

import math
from pprint import pformat
import cv2

import mindspore.dataset as de

from src.my_logging import get_logger
from model_utils.config import config

class TxtDataset():
    '''TxtDataset'''
    def __init__(self, root_all, filenames):
        super(TxtDataset, self).__init__()
        self.imgs = []
        self.labels = []
        self.path = []
        for root, filename in zip(root_all, filenames):
            fin = open(filename, "r")
            for line in fin:
                self.imgs.append(os.path.join(root, line.strip().split(" ")[0]))
                self.labels.append(line.strip())
                self.path.append(os.path.join(root, line.strip().split(" ")[0]))
            fin.close()

    def __getitem__(self, index):
        try:
            img = cv2.cvtColor(cv2.imread(self.imgs[index]), cv2.COLOR_BGR2RGB)
            path = self.path[index]
        except:
            print(self.imgs[index])
            print(self.path[index])
            raise
        return img, path, index

    def __len__(self):
        return len(self.imgs)

    def get_all_labels(self):
        return self.labels

class DistributedSampler():
    '''DistributedSampler'''
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_replicas = 1
        self.rank = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def get_dataloader(img_predix_all, img_list_all):
    dataset = TxtDataset(img_predix_all, img_list_all)
    sampler = DistributedSampler(dataset)
    dataset_column_names = ["image", "path", "index"]
    ds = de.GeneratorDataset(dataset, column_names=dataset_column_names, sampler=sampler)
    ds = ds.batch(1, num_parallel_workers=8, drop_remainder=False)
    ds = ds.repeat(1)

    return ds, len(dataset), dataset.get_all_labels()

def merge_data(test_img_predix, test_img_list, dis_img_predix, dis_img_list):
    '''extract data.'''
    ds, _, _ = get_dataloader(test_img_predix, test_img_list)
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
    for _, data in enumerate(data_loader):
        _, path, _ = data["image"], data["path"], data["index"]
        path_01 = str(path).split('\'')[-2]
        new_name = path_01.split('/')[-3] + '_' + path_01.split('/')[-2] + '_' +path_01.split('/')[-1]
        path_new = './preprocess_Result' + '/' + new_name
        shutil.copyfile(path_01, path_new)

    # for dis images
    ds_dis, _, _ = get_dataloader(dis_img_predix, dis_img_list)
    data_loader = ds_dis.create_dict_iterator(output_numpy=True, num_epochs=1)
    for _, data in enumerate(data_loader):
        _, path, _ = data["image"], data["path"], data["index"]
        path_01 = str(path).split('\'')[-2]
        new_name = path_01.split('/')[-3] + '_' + path_01.split('/')[-2] + '_' +path_01.split('/')[-1]
        path_new = './preprocess_Result' + '/' + new_name
        shutil.copyfile(path_01, path_new)

if __name__ == '__main__':
    config.test_img_predix = [os.path.join(config.test_dir, 'test_dataset/'),
                              os.path.join(config.test_dir, 'test_dataset/')]

    config.test_img_list = [os.path.join(config.test_dir, 'lists/jk_list.txt'),
                            os.path.join(config.test_dir, 'lists/zj_list.txt')]
    config.dis_img_predix = [os.path.join(config.test_dir, 'dis_dataset/'),]
    config.dis_img_list = [os.path.join(config.test_dir, 'lists/dis_list.txt'),]

    log_path = os.path.join(config.ckpt_path, 'logs')
    config.logger = get_logger(log_path, config.local_rank)

    config.logger.info('Config %s', pformat(config))

    merge_data(config.test_img_predix, config.test_img_list, config.dis_img_predix, config.dis_img_list)
