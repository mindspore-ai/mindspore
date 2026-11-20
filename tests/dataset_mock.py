# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
'''MindData stubs and private fixture sources shared by UT and ST.'''
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import numpy as np

from mindspore import Tensor
import mindspore.dataset as ds
from mindspore.dataset.vision import Decode


class MindData:
    """ Stub for MindData """

    def __init__(self, size=1, batch_size=None, repeat_count=1,
                 np_types=None, output_shapes=None, input_indexs=()):
        self._size = size
        self._batch_size = batch_size
        self._repeat_count = repeat_count
        self._np_types = np_types
        self._output_shapes = output_shapes
        self._input_indexs = input_indexs
        self._iter_num = 0
        self._global_step = 0

    def get_dataset_size(self):
        return self._size

    def get_repeat_count(self):
        return self._repeat_count

    def get_batch_size(self):
        return self._batch_size

    def output_types(self):
        return self._np_types

    def output_shapes(self):
        return self._output_shapes

    @property
    def input_indexs(self):
        return self._input_indexs

    def device_que(self, send_epoch_end=True, create_data_info_queue=False, queue_name=""):
        self.queue_name = '6ba41974-209e-11ea-88b0-a24efeb2c736'
        self.send_epoch_end = send_epoch_end
        return self

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        return self.__iter__()

    def send(self, num_epochs=-1):
        pass

    def stop_send(self):
        pass

    def release(self):
        pass

    def continue_send(self):
        pass

    def get_data_info(self):
        pass

    def get_mbuf_queue_size(self):
        pass

    def get_send_info(self):
        pass

    def __len__(self):
        return self._size

    def __iter__(self):
        return self

    def __next__(self):
        if self._size < self._iter_num:
            raise StopIteration
        self._iter_num += 1
        next_value = []
        for shape, typ in zip(self._output_shapes, self._np_types):
            next_value.append(Tensor(np.ndarray(shape, typ)))

        return tuple(next_value)

    def next(self):
        return self.__next__()

    def reset(self):
        self._iter_num = 0

    def get_init_step(self):
        return self._global_step


# Private fixture sources use retained GeneratorDataset; these are not public reader replacements.
class _Columns:
    def __init__(self, *columns):
        self.columns = columns

    def __len__(self):
        return len(self.columns[0])

    def __getitem__(self, index):
        return tuple(column[index] for column in self.columns)


def mnist_source(dataset_dir, usage=None, **kwargs):
    """Read IDX fixtures without a native benchmark reader."""
    root = Path(dataset_dir)
    images, labels = [], []
    prefixes = ('train', 't10k') if usage in (None, 'all') else (('t10k',) if usage == 'test' else ('train',))
    for prefix in prefixes:
        for file in sorted(root.glob(prefix+'*images*idx3*')):
            with file.open('rb') as stream:
                magic, count, height, width = struct.unpack('>IIII', stream.read(16))
                if magic != 2051:
                    raise ValueError(f'Invalid IDX image fixture: {file}')
                image = np.frombuffer(stream.read(), np.uint8).reshape(count, height, width, 1)
            matches = sorted(root.glob(prefix+'*labels*idx1*'))
            if len(matches) != 1:
                raise ValueError(f'Expected one IDX label file under {root}')
            with matches[0].open('rb') as stream:
                magic, label_count = struct.unpack('>II', stream.read(8))
                if magic != 2049 or label_count != count:
                    raise ValueError('Mismatched IDX fixture labels')
                label = np.frombuffer(stream.read(), np.uint8).astype(np.uint32)
            images.append(image)
            labels.append(label)
    if not images:
        raise ValueError(f'No IDX image fixtures under {root}')
    return ds.GeneratorDataset(_Columns(np.concatenate(images), np.concatenate(labels)), ['image', 'label'], **kwargs)


def cifar_source(dataset_dir, usage=None, *, coarse=False, **kwargs):
    """Load binary image fixtures with the original HWC layout and uint32 labels."""
    root = Path(dataset_dir)
    if coarse:
        candidates = [root/'train.bin', root/'test.bin']
    else:
        candidates = sorted(root.glob('data_batch*.bin')) + sorted(root.glob('test_batch*.bin'))
    files = [p for p in candidates if p.exists() and
             (usage in (None, 'all') or ('test' in p.name) == (usage == 'test'))]
    if not files:
        raise ValueError(f'No binary image fixtures under {root}')
    label_size = 2 if coarse else 1
    rows = np.concatenate([np.fromfile(p, np.uint8).reshape(-1, 3072+label_size) for p in files])
    images = rows[:, label_size:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    if coarse:
        source = _Columns(images, rows[:, 0].astype(np.uint32), rows[:, 1].astype(np.uint32))
        columns = ['image', 'coarse_label', 'fine_label']
    else:
        source = _Columns(images, rows[:, 0].astype(np.uint32))
        columns = ['image', 'label']
    return ds.GeneratorDataset(source, columns, **kwargs)


def cifar100_source(dataset_dir, usage=None, **kwargs):
    """Read the coarse/fine-label binary fixtures."""
    return cifar_source(dataset_dir, usage, coarse=True, **kwargs)


def _image(path, decode):
    data = np.fromfile(path, dtype=np.uint8)
    return Decode()(data) if decode else data


class _ImageRows:
    """Decode fixture image paths while preserving annotation arrays."""
    def __init__(self, rows, decode):
        self.rows = rows
        self.decode = decode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return tuple(_image(value, self.decode) if isinstance(value, Path) else value
                     for value in self.rows[index])


def voc_source(dataset_dir, task='Segmentation', usage='train', decode=False, **kwargs):
    """Load only the existing detection/segmentation fixtures for vision tests."""
    root = Path(dataset_dir)
    split = 'Main' if task == 'Detection' else 'Segmentation'
    ids = (root/'ImageSets'/split/(usage+'.txt')).read_text().split()
    if task == 'Segmentation':
        rows = [(root/'JPEGImages'/(name+'.jpg'), root/'SegmentationClass'/(name+'.png')) for name in ids]
        return ds.GeneratorDataset(_ImageRows(rows, decode), ['image', 'target'], **kwargs)
    if task != 'Detection':
        raise ValueError('The fixture supports Detection and Segmentation only')
    objects = [ET.parse(root/'Annotations'/(name+'.xml')).getroot().findall('object')
               for name in ids]
    class_names = sorted({o.findtext('name') for row in objects for o in row})
    labels = {name: index for index, name in enumerate(class_names)}
    rows = []
    for name, row in zip(ids, objects):
        if not row:
            continue
        boxes = []
        for obj in row:
            box = obj.find('bndbox')
            xmin, ymin, xmax, ymax = [float(box.findtext(key)) for key in ('xmin','ymin','xmax','ymax')]
            boxes.append([xmin, ymin, xmax-xmin, ymax-ymin])
        rows.append((root/'JPEGImages'/(name+'.jpg'), np.array(boxes, np.float32),
                     np.array([[labels[o.findtext('name')]] for o in row], np.uint32),
                     np.array([[int(o.findtext('difficult', '0'))] for o in row], np.uint32),
                     np.array([[int(o.findtext('truncated', '0'))] for o in row], np.uint32)))
    return ds.GeneratorDataset(_ImageRows(rows, decode), ['image','bbox','label','difficult','truncate'], **kwargs)


def coco_source(dataset_dir, annotation_file, task='Detection', decode=False, extra_metadata=False, **kwargs):
    """Read the detection-only JSON fixtures used by bbox transform tests."""
    if task != 'Detection':
        raise ValueError('The fixture supports Detection only')
    data = json.loads(Path(annotation_file).read_text(encoding='utf-8'))
    rows = []
    for image in data['images']:
        annotations = [a for a in data['annotations'] if a['image_id'] == image['id']]
        if not annotations:
            continue
        row = (Path(dataset_dir)/image['file_name'],
               np.array([a['bbox'] for a in annotations], np.float32),
               np.array([[a['category_id']] for a in annotations], np.uint32),
               np.array([[a['iscrowd']] for a in annotations], np.uint32))
        if extra_metadata:
            row += (np.asarray(Path(image['file_name']).stem),)
        rows.append(row)
    columns = ['image','bbox','category_id','iscrowd']
    if extra_metadata:
        columns.append('_meta-filename')
    return ds.GeneratorDataset(_ImageRows(rows, decode), columns, **kwargs)


def celeba_source(dataset_dir, usage=None, decode=False, **kwargs):
    """Read the attribute fixture; model/transform assertions remain unchanged."""
    root = Path(dataset_dir)
    lines = (root/'list_attr_celeba.txt').read_text().splitlines()[2:]
    partitions = dict(line.split() for line in (root/'list_eval_partition.txt').read_text().splitlines())
    subset = {'train':'0', 'valid':'1', 'test':'2'}
    rows=[]
    for line in lines:
        name, *attrs = line.split()
        if usage not in (None, 'all') and partitions[name] != subset[usage]:
            continue
        path = root/name
        if not path.exists():
            path = root/'img_align_celeba'/name
        rows.append((path, np.array([int(a)>0 for a in attrs], np.uint32)))
    return ds.GeneratorDataset(_ImageRows(rows, decode), ['image','attr'], **kwargs)


class _SyntheticImages:
    def __init__(self, count, shape, classes, seed):
        self.count, self.shape, self.classes, self.seed = count, shape, classes, seed

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed + index)
        return rng.randint(0, 256, self.shape, dtype=np.uint8), np.uint32(index % self.classes)


def synthetic_image_source(num_images=1000, image_size=(224,224,3), num_classes=10, base_seed=0, **kwargs):
    source = _SyntheticImages(num_images, image_size, num_classes, base_seed)
    return ds.GeneratorDataset(source, ['image','label'], **kwargs)
