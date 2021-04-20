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
dataset for ProtoNet
"""
import os
from PIL import Image
import numpy as np

IMG_CACHE = {}


class OmniglotDataset():
    """
    Omniglot dataset class
    """

    splits_folder = os.path.join('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, mode='train', root='.' + os.sep + 'dataset', transform=None, target_transform=None):
        self.root = root
        print(self.root)
        self.transform = transform
        self.target_transform = target_transform

        self.classes = get_current_classes(os.path.join(
            self.root, self.splits_folder, mode + '.txt'))
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder), self.classes)

        self.idx_classes = index_classes(self.all_items)
        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))])
        self.x = map(load_img, paths, range(len(paths)))
        self.x = list(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        filename = self.all_items[index][0]
        rot = self.all_items[index][-1]
        img = str.join(os.sep, [self.all_items[index][2], filename]) + rot
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def find_items(root_dir, classes):
    """
    function to find items
    """
    retour = []
    rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
    for (root, _, files) in os.walk(root_dir):
        for f in files:
            r = root.split(os.sep)
            lr = len(r)
            label = r[lr - 2] + os.sep + r[lr - 1]
            for rot in rots:
                if label + rot in classes and (f.endswith("png")):
                    retour.extend([(f, label, root, rot)])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    """
    how mach items and classes dataset have
    """
    idx = {}
    for i in items:
        if not i[1] + i[-1] in idx:
            idx[i[1] + i[-1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    """
    get current classes
    """
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    return classes


def load_img(path, idx):
    """
    function to load images
    """
    path, rot = path.split(os.sep + 'rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))

    shape = 1, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - x
    x = x.T.reshape(shape)

    return x
