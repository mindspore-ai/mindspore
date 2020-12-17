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
import os
import math
import pickle
from collections import defaultdict
import numpy as np

from PIL import Image, ImageFile
from mindspore.communication.management import get_group_size, get_rank
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['DistributedCustomSampler', 'CustomDataset']
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class DistributedCustomSampler:
    '''DistributedCustomSampler'''
    def __init__(self, dataset, num_replicas=None, rank=None, is_distributed=1, shuffle=True, k=2):
        assert isinstance(dataset, CustomDataset), 'Custom Sampler is Only Support Custom Dataset!!!'
        if is_distributed:
            if num_replicas is None:
                num_replicas = get_group_size()
            if rank is None:
                rank = get_rank()
        else:
            if num_replicas is None:
                num_replicas = 1
            if rank is None:
                rank = 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.ratio = 4.0
        self.data_len = len(self.dataset.classes)
        self.num_ids = int(math.ceil(self.data_len * 1.0 / self.num_replicas))
        self.total_ids = self.num_ids * self.num_replicas
        self.num_samples = math.ceil(len(self.dataset) * 1.0 / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.k = k
        self.epoch_gen = 1

    def _sample_(self, indices):
        sampled = []
        for indice in indices:
            sampled_id = indice
            sampled.extend(np.random.choice(self.dataset.id2range[sampled_id][:], self.k).tolist())

        return sampled

    def __iter__(self):
        if self.shuffle:
            # Note, the self.epoch parameter does not get updated in DE
            self.epoch_gen = (self.epoch_gen + 1) & 0xffffffff
            np.random.seed(self.epoch_gen)
            indices = np.random.permutation(len(self.dataset.classes))
            indices = indices.tolist()
        else:
            indices = list(range(len(self.dataset.classes)))

        indices += indices[:(self.total_ids - len(indices))]
        assert len(indices) == self.total_ids

        indices = indices[self.rank*self.num_ids:(self.rank+1)*self.num_ids]
        assert len(indices) == self.num_ids
        sampled_idxs = self._sample_(indices)
        return iter(sampled_idxs)

    def __len__(self):
        return self.num_ids * self.k

    def set_epoch(self, epoch):
        self.epoch = epoch

    def merge_indices(self, list1, list2):
        '''merge_indices'''
        list_result = []
        ct_1, ct_2 = 0, 0
        for i in range(self.data_len):
            if (i+1) % int(self.ratio+1) == 0:
                list_result.append(list2[ct_2])
                ct_2 += 1
            else:
                list_result.append(list1[ct_1])
                ct_1 += 1
        return list_result


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir_1, class_to_idx, extensions=None, is_valid_file=None):
    '''make_dataset'''
    images = []
    dir_1 = os.path.expanduser(dir_1)
    if not (extensions is None) ^ (is_valid_file is None):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def f(x):
            return has_file_allowed_extension(x, extensions)
        is_valid_file = f
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir_1, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class ImageFolderDataset:
    '''ImageFolderDataset'''
    def __init__(self, root, cache_path, is_distributed):

        if not os.path.isfile(cache_path):
            self.classes, self.classes_to_idx = self._find_classes(root)
            self.samples = make_dataset(root, self.classes_to_idx, IMG_EXTENSIONS, None)
            self.id2range = self._build_id2range()
            cache = dict()
            cache['classes'] = self.classes
            cache['classes_to_idx'] = self.classes_to_idx
            cache['samples'] = self.samples
            cache['id2range'] = self.id2range
            if is_distributed:
                print("******* TODO: All workers will write cache... Need to only dump when rank == 0 ******")
                if get_rank() == 0:
                    with open(cache_path, 'wb') as fw:
                        pickle.dump(cache, fw)
                    print('local dump cache:{}'.format(cache_path))
            else:
                with open(cache_path, 'wb') as fw:
                    pickle.dump(cache, fw)
                print('local dump cache:{}'.format(cache_path))
        else:
            print('loading cache from %s'%cache_path)
            with open(cache_path, 'rb') as fr:
                cache = pickle.load(fr)
                self.classes, self.classes_to_idx, self.samples, self.id2range = cache['classes'], \
                                                                                 cache['classes_to_idx'], \
                                                                                 cache['samples'], cache['id2range']

        self.all_image_idxs = range(len(self.samples))

        self.classes = list(self.id2range.keys())

    def _find_classes(self, dir_1):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir_1) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir_1) if os.path.isdir(os.path.join(dir_1, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _build_id2range(self):
        '''_build_id2range'''
        id2range = defaultdict(list)
        ret_range = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            label = sample[1]
            id2range[label].append((sample, idx))
        for key in id2range:
            id2range[key].sort(key=lambda x: int(os.path.basename(x[0][0]).split('.')[0]))
            for item in id2range[key]:
                ret_range[key].append(item[1])
        return ret_range

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def pil_loader(path):
    """
    Loads the image
    Args:
        path: path to the image
    Returns:
        Object: pil_loader
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CustomDataset:
    '''CustomDataset'''
    def __init__(self, root, cache_path, is_distributed=1, transform=None, target_transform=None,
                 loader=pil_loader):
        self.dataset = ImageFolderDataset(root, cache_path, is_distributed)
        print('CustomDataset len(dataset):{}'.format(len(self.dataset)))
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.classes = self.dataset.classes
        self.id2range = self.dataset.id2range

    def __getitem__(self, index):
        path, target = self.dataset[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)
