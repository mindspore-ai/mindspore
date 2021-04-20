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
dataset iter generator script.
"""
import numpy as np
from tqdm import tqdm


class IterDatasetGenerator:
    """
    dataloader class
    """
    def __init__(self, data, classes_per_it, num_samples, iterations):
        self.__iterations = iterations
        self.__data = data.x
        self.__labels = data.y
        self.__iter = 0
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.classes, self.counts = np.unique(self.__labels, return_counts=True)
        self.idxs = range(len(self.__labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.numel_per_class = np.zeros_like(self.classes)
        for idx, label in tqdm(enumerate(self.__labels)):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] = int(self.numel_per_class[label_idx]) + 1

        print('init end')


    def __next__(self):
        spc = self.sample_per_class
        cpi = self.classes_per_it

        if self.__iter >= self.__iterations:
            raise StopIteration
        batch_size = spc * cpi
        batch = np.random.randint(low=batch_size, high=10 * batch_size, size=(batch_size), dtype=np.int32)
        c_idxs = np.random.permutation(len(self.classes))[:cpi]
        for indx, c in enumerate(self.classes[c_idxs]):
            index = indx*spc
            ci = [c_i for c_i in range(len(self.classes)) if self.classes[c_i] == c][0]
            label_idx = list(range(len(self.classes)))[ci]
            sample_idxs = np.random.permutation(int(self.numel_per_class[label_idx]))[:spc]
            ind = 0
            for sid in sample_idxs:
                batch[index+ind] = self.indexes[label_idx][sid]
                ind = ind + 1
        batch = batch[np.random.permutation(len(batch))]
        data_x = []
        data_y = []
        for b in batch:
            data_x.append(self.__data[b])
            data_y.append(self.__labels[b])
        self.__iter += 1
        data_y = np.asarray(data_y, np.int32)
        data_class = np.asarray(np.unique(data_y), np.int32)
        item = (data_x, data_y, data_class)
        return item

    def __iter__(self):
        self.__iter = 0
        return self

    def __len__(self):
        return self.__iterations
