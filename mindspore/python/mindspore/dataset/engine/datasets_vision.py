# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
This file contains specific vision dataset loading classes. You can easily use
these classes to load the prepared dataset. For example:
    ImageFolderDataset: which is about ImageNet dataset.
    Cifar10Dataset: which is cifar10 binary version dataset.
    Cifar100Dataset: which is cifar100 binary version dataset.
    MnistDataset: which is mnist dataset.
    ...
After declaring the dataset object, you can further apply dataset operations
(e.g. filter, skip, concat, map, batch) on it.
"""
import os
import numpy as np
from scipy.io import loadmat
from PIL import Image

import mindspore._c_dataengine as cde

from .datasets import VisionBaseDataset, SourceDataset, MappableDataset, Shuffle, Schema
from .datasets_user_defined import GeneratorDataset
from .validators import check_caltech101_dataset, check_caltech256_dataset, check_celebadataset, \
    check_cityscapes_dataset, check_cocodataset, check_div2k_dataset, check_emnist_dataset, check_fake_image_dataset, \
    check_flickr_dataset, check_flowers102dataset, check_food101_dataset, check_imagefolderdataset, \
    check_kittidataset, check_lfw_dataset, check_lsun_dataset, check_manifestdataset, check_mnist_cifar_dataset, \
    check_omniglotdataset, check_photo_tour_dataset, check_places365_dataset, check_qmnist_dataset, \
    check_random_dataset, check_rendered_sst2_dataset, check_sb_dataset, check_sbu_dataset, check_semeion_dataset, \
    check_stl10_dataset, check_sun397_dataset, check_svhn_dataset, check_usps_dataset, check_vocdataset, \
    check_wider_face_dataset

from ..core.validator_helpers import replace_none


class _Caltech101Dataset:
    """
    Mainly for loading Caltech101 Dataset, and return two rows each time.
    """

    def __init__(self, dataset_dir, target_type="category", decode=False):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.annotation_dir = os.path.join(self.dataset_dir, "Annotations")
        self.target_type = target_type
        if self.target_type == "category":
            self.column_names = ["image", "category"]
        elif self.target_type == "annotation":
            self.column_names = ["image", "annotation"]
        else:
            self.column_names = ["image", "category", "annotation"]
        self.decode = decode
        self.classes = sorted(os.listdir(self.image_dir))
        if "BACKGROUND_Google" in self.classes:
            self.classes.remove("BACKGROUND_Google")
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_classes = [name_map[class_name] if class_name in name_map else class_name
                                   for class_name in self.classes]
        self.image_index = []
        self.image_label = []
        for i, image_class in enumerate(self.classes):
            sub_dir = os.path.join(self.image_dir, image_class)
            if not os.path.isdir(sub_dir) or not os.access(sub_dir, os.R_OK):
                continue
            num_images = len(os.listdir(sub_dir))
            self.image_index.extend(range(1, num_images + 1))
            self.image_label.extend(num_images * [i])

    def __getitem__(self, index):
        image_file = os.path.join(self.image_dir, self.classes[self.image_label[index]],
                                  "image_{:04d}.jpg".format(self.image_index[index]))
        if not os.path.exists(image_file):
            raise ValueError("The image file {} does not exist or permission denied!".format(image_file))
        if self.decode:
            image = np.asarray(Image.open(image_file).convert("RGB"))
        else:
            image = np.fromfile(image_file, dtype=np.uint8)

        if self.target_type == "category":
            return image, self.image_label[index]
        annotation_file = os.path.join(self.annotation_dir, self.annotation_classes[self.image_label[index]],
                                       "annotation_{:04d}.mat".format(self.image_index[index]))
        if not os.path.exists(annotation_file):
            raise ValueError("The annotation file {} does not exist or permission denied!".format(annotation_file))
        annotation = loadmat(annotation_file)["obj_contour"]

        if self.target_type == "annotation":
            return image, annotation
        return image, self.image_label[index], annotation

    def __len__(self):
        return len(self.image_index)


class Caltech101Dataset(GeneratorDataset):
    """
    A source dataset that reads and parses Caltech101 dataset.

    The columns of the generated dataset depend on the value of `target_type` .

    - When `target_type` is 'category', the columns are :py:obj:`[image, category]` .
    - When `target_type` is 'annotation', the columns are :py:obj:`[image, annotation]` .
    - When `target_type` is 'all', the columns are :py:obj:`[image, category, annotation]` .

    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`category` is of the uint32 type.
    The tensor of column :py:obj:`annotation` is a 2-dimensional ndarray that stores the contour of the image
    and consists of a series of points.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset. This root directory contains two
            subdirectories, one is called 101_ObjectCategories, which stores images,
            and the other is called Annotations, which stores annotations.
        target_type (str, optional): Target of the image. If `target_type` is 'category', return category represents
            the target class. If `target_type` is 'annotation', return annotation.
            If `target_type` is 'all', return category and annotation. Default: None, means 'category'.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Whether or not to decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `target_type` is not set correctly.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> caltech101_dataset_directory = "/path/to/caltech101_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in caltech101_dataset_directory with 8 threads
        >>> dataset = ds.Caltech101Dataset(dataset_dir=caltech101_dataset_directory, num_parallel_workers=8)
        >>>
        >>> # 2) Read all samples (image files) with the target_type "annotation"
        >>> dataset = ds.Caltech101Dataset(dataset_dir=caltech101_dataset_directory, target_type="annotation")

    About Caltech101Dataset:

    Pictures of objects belonging to 101 categories, about 40 to 800 images per category.
    Most categories have about 50 images. The size of each image is roughly 300 x 200 pixels.
    The official provides the contour data of each object in each picture, which is the annotation.

    Here is the original Caltech101 dataset structure,
    and you can unzip the dataset files into the following directory structure, which are read by MindSpore API.

    .. code-block::

        .
        └── caltech101_dataset_directory
            ├── 101_ObjectCategories
            │    ├── Faces
            │    │    ├── image_0001.jpg
            │    │    ├── image_0002.jpg
            │    │    ...
            │    ├── Faces_easy
            │    │    ├── image_0001.jpg
            │    │    ├── image_0002.jpg
            │    │    ...
            │    ├── ...
            └── Annotations
                 ├── Airplanes_Side_2
                 │    ├── annotation_0001.mat
                 │    ├── annotation_0002.mat
                 │    ...
                 ├── Faces_2
                 │    ├── annotation_0001.mat
                 │    ├── annotation_0002.mat
                 │    ...
                 ├── ...

    Citation:

    .. code-block::

        @article{FeiFei2004LearningGV,
        author    = {Li Fei-Fei and Rob Fergus and Pietro Perona},
        title     = {Learning Generative Visual Models from Few Training Examples:
                    An Incremental Bayesian Approach Tested on 101 Object Categories},
        journal   = {Computer Vision and Pattern Recognition Workshop},
        year      = {2004},
        url       = {https://data.caltech.edu/records/mzrjq-6wc02},
        }
    """

    @check_caltech101_dataset
    def __init__(self, dataset_dir, target_type=None, num_samples=None, num_parallel_workers=1,
                 shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None):
        self.dataset_dir = dataset_dir
        self.target_type = replace_none(target_type, "category")
        self.decode = replace_none(decode, False)
        dataset = _Caltech101Dataset(self.dataset_dir, self.target_type, self.decode)
        super().__init__(dataset, column_names=dataset.column_names, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
        """
        class_dict = {'Faces': 0, 'Faces_easy': 1, 'Leopards': 2, 'Motorbikes': 3, 'accordion': 4, 'airplanes': 5,
                      'anchor': 6, 'ant': 7, 'barrel': 8, 'bass': 9, 'beaver': 10, 'binocular': 11, 'bonsai': 12,
                      'brain': 13, 'brontosaurus': 14, 'buddha': 15, 'butterfly': 16, 'camera': 17, 'cannon': 18,
                      'car_side': 19, 'ceiling_fan': 20, 'cellphone': 21, 'chair': 22, 'chandelier': 23,
                      'cougar_body': 24, 'cougar_face': 25, 'crab': 26, 'crayfish': 27, 'crocodile': 28,
                      'crocodile_head': 29, 'cup': 30, 'dalmatian': 31, 'dollar_bill': 32, 'dolphin': 33,
                      'dragonfly': 34, 'electric_guitar': 35, 'elephant': 36, 'emu': 37, 'euphonium': 38, 'ewer': 39,
                      'ferry': 40, 'flamingo': 41, 'flamingo_head': 42, 'garfield': 43, 'gerenuk': 44, 'gramophone': 45,
                      'grand_piano': 46, 'hawksbill': 47, 'headphone': 48, 'hedgehog': 49, 'helicopter': 50, 'ibis': 51,
                      'inline_skate': 52, 'joshua_tree': 53, 'kangaroo': 54, 'ketch': 55, 'lamp': 56, 'laptop': 57,
                      'llama': 58, 'lobster': 59, 'lotus': 60, 'mandolin': 61, 'mayfly': 62, 'menorah': 63,
                      'metronome': 64, 'minaret': 65, 'nautilus': 66, 'octopus': 67, 'okapi': 68, 'pagoda': 69,
                      'panda': 70, 'pigeon': 71, 'pizza': 72, 'platypus': 73, 'pyramid': 74, 'revolver': 75,
                      'rhino': 76, 'rooster': 77, 'saxophone': 78, 'schooner': 79, 'scissors': 80, 'scorpion': 81,
                      'sea_horse': 82, 'snoopy': 83, 'soccer_ball': 84, 'stapler': 85, 'starfish': 86,
                      'stegosaurus': 87, 'stop_sign': 88, 'strawberry': 89, 'sunflower': 90, 'tick': 91,
                      'trilobite': 92, 'umbrella': 93, 'watch': 94, 'water_lilly': 95, 'wheelchair': 96, 'wild_cat': 97,
                      'windsor_chair': 98, 'wrench': 99, 'yin_yang': 100}
        return class_dict


class Caltech256Dataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses Caltech256 dataset.

    The generated dataset has two columns: :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Whether or not to decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `target_type` is not 'category', 'annotation' or 'all'.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> caltech256_dataset_dir = "/path/to/caltech256_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in caltech256_dataset_dir with 8 threads
        >>> dataset = ds.Caltech256Dataset(dataset_dir=caltech256_dataset_dir, num_parallel_workers=8)

    About Caltech256Dataset:

    Caltech-256 is an object recognition dataset containing 30,607 real-world images, of different sizes,
    spanning 257 classes (256 object classes and an additional clutter class).
    Each class is represented by at least 80 images. The dataset is a superset of the Caltech-101 dataset.

    .. code-block::

        .
        └── caltech256_dataset_directory
             ├── 001.ak47
             │    ├── 001_0001.jpg
             │    ├── 001_0002.jpg
             │    ...
             ├── 002.american-flag
             │    ├── 002_0001.jpg
             │    ├── 002_0002.jpg
             │    ...
             ├── 003.backpack
             │    ├── 003_0001.jpg
             │    ├── 003_0002.jpg
             │    ...
             ├── ...

    Citation:

    .. code-block::

        @article{griffin2007caltech,
        title     = {Caltech-256 object category dataset},
        added-at  = {2021-01-21T02:54:42.000+0100},
        author    = {Griffin, Gregory and Holub, Alex and Perona, Pietro},
        biburl    = {https://www.bibsonomy.org/bibtex/21f746f23ff0307826cca3e3be45f8de7/s364315},
        interhash = {bfe1e648c1778c04baa60f23d1223375},
        intrahash = {1f746f23ff0307826cca3e3be45f8de7},
        publisher = {California Institute of Technology},
        timestamp = {2021-01-21T02:54:42.000+0100},
        year      = {2007}
        }
    """

    @check_caltech256_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.Caltech256Node(self.dataset_dir, self.decode, self.sampler)


class CelebADataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses CelebA dataset.
    Only support to read `list_attr_celeba.txt` currently, which is the attribute annotations of the dataset.

    The generated dataset has two columns: :py:obj:`[image, attr]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`attr` is of the uint32 type and one hot encoded.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_parallel_workers (int, optional): Number of workers to read the data. Default: None, will use value set in
            the config.
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None.
        usage (str, optional): Specify the 'train', 'valid', 'test' part or 'all' parts of dataset.
            Default: 'all', will read all samples.
        sampler (Sampler, optional): Object used to choose samples from the dataset. Default: None.
        decode (bool, optional): Whether to decode the images after reading. Default: False.
        extensions (list[str], optional): List of file extensions to be included in the dataset. Default: None.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will include all images.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.
        decrypt (callable, optional): Image decryption function, which accepts the path of the encrypted image file
            and returns the decrypted bytes data. Default: None, no decryption.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `usage` is not 'train', 'valid', 'test' or 'all'.

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> celeba_dataset_dir = "/path/to/celeba_dataset_directory"
        >>>
        >>> # Read 5 samples from CelebA dataset
        >>> dataset = ds.CelebADataset(dataset_dir=celeba_dataset_dir, usage='train', num_samples=5)
        >>>
        >>> # Note: In celeba dataset, each data dictionary owns keys "image" and "attr"

    About CelebA dataset:

    CelebFaces Attributes Dataset (CelebA) is a large-scale dataset
    with more than 200K celebrity images, each with 40 attribute annotations.

    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including

    * 10,177 number of identities,
    * 202,599 number of images,
    * 5 landmark locations, 40 binary attributes annotations per image.

    The dataset can be employed as the training and test sets for the following computer
    vision tasks: attribute recognition, detection, landmark (or facial part) and
    localization.

    Original CelebA dataset structure:

    .. code-block::

        .
        └── CelebA
             ├── README.md
             ├── Img
             │    ├── img_celeba.7z
             │    ├── img_align_celeba_png.7z
             │    └── img_align_celeba.zip
             ├── Eval
             │    └── list_eval_partition.txt
             └── Anno
                  ├── list_landmarks_celeba.txt
                  ├── list_landmarks_align_celeba.txt
                  ├── list_bbox_celeba.txt
                  ├── list_attr_celeba.txt
                  └── identity_CelebA.txt

    You can unzip the dataset files into the following structure and read by MindSpore's API.

    .. code-block::

        .
        └── celeba_dataset_directory
            ├── list_attr_celeba.txt
            ├── 000001.jpg
            ├── 000002.jpg
            ├── 000003.jpg
            ├── ...

    Citation:

    .. code-block::

        @article{DBLP:journals/corr/LiuLWT14,
        author        = {Ziwei Liu and Ping Luo and Xiaogang Wang and Xiaoou Tang},
        title         = {Deep Learning Attributes in the Wild},
        journal       = {CoRR},
        volume        = {abs/1411.7766},
        year          = {2014},
        url           = {http://arxiv.org/abs/1411.7766},
        archivePrefix = {arXiv},
        eprint        = {1411.7766},
        timestamp     = {Tue, 10 Dec 2019 15:37:26 +0100},
        biburl        = {https://dblp.org/rec/journals/corr/LiuLWT14.bib},
        bibsource     = {dblp computer science bibliography, https://dblp.org},
        howpublished  = {http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html}
        }
    """

    @check_celebadataset
    def __init__(self, dataset_dir, num_parallel_workers=None, shuffle=None, usage='all', sampler=None, decode=False,
                 extensions=None, num_samples=None, num_shards=None, shard_id=None, cache=None, decrypt=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.decode = replace_none(decode, False)
        self.extensions = replace_none(extensions, [])
        self.usage = replace_none(usage, "all")
        self.decrypt = decrypt

    def parse(self, children=None):
        if self.usage != "all":
            dataset_dir = os.path.realpath(self.dataset_dir)
            partition_file = os.path.join(dataset_dir, "list_eval_partition.txt")
            if os.path.exists(partition_file) is False:
                raise RuntimeError("Partition file can not be found when usage is not 'all'.")
        return cde.CelebANode(self.dataset_dir, self.usage, self.sampler, self.decode,
                              self.extensions, self.decrypt)



class Cifar10Dataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses Cifar10 dataset.
    This api only supports parsing Cifar10 file in binary version now.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all' . 'train' will read from 50,000
            train samples, 'test' will read from 10,000 test samples, 'all' will read from all 60,000 samples.
            Default: None, all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `usage` is not 'train', 'test' or 'all'.

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> cifar10_dataset_dir = "/path/to/cifar10_dataset_directory"
        >>>
        >>> # 1) Get all samples from CIFAR10 dataset in sequence
        >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from CIFAR10 dataset
        >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from CIFAR10 dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, num_shards=2, shard_id=0)
        >>>
        >>> # In CIFAR10 dataset, each dictionary has keys "image" and "label"

    About CIFAR-10 dataset:

    The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes,
    with 6000 images per class. There are 50000 training images and 10000 test images.
    The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

    Here is the original CIFAR-10 dataset structure.
    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── cifar-10-batches-bin
             ├── data_batch_1.bin
             ├── data_batch_2.bin
             ├── data_batch_3.bin
             ├── data_batch_4.bin
             ├── data_batch_5.bin
             ├── test_batch.bin
             ├── readme.html
             └── batches.meta.txt

    Citation:

    .. code-block::

        @techreport{Krizhevsky09,
        author       = {Alex Krizhevsky},
        title        = {Learning multiple layers of features from tiny images},
        institution  = {},
        year         = {2009},
        howpublished = {http://www.cs.toronto.edu/~kriz/cifar.html}
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.Cifar10Node(self.dataset_dir, self.usage, self.sampler)


class Cifar100Dataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses Cifar100 dataset.

    The generated dataset has three columns :py:obj:`[image, coarse_label, fine_label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`coarse_label` and :py:obj:`fine_labels` are each a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all' . 'train' will read from 50,000
            train samples, 'test' will read from 10,000 test samples, 'all' will read from all 60,000 samples.
            Default: None, all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, 'num_samples' reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `usage` is not 'train', 'test' or 'all'.

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and shuffle
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> cifar100_dataset_dir = "/path/to/cifar100_dataset_directory"
        >>>
        >>> # 1) Get all samples from CIFAR100 dataset in sequence
        >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from CIFAR100 dataset
        >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # In CIFAR100 dataset, each dictionary has 3 keys: "image", "fine_label" and "coarse_label"

    About CIFAR-100 dataset:

    This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images
    each. There are 500 training images and 100 testing images per class. The 100 classes in
    the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the
    class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Here is the original CIFAR-100 dataset structure.
    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── cifar-100-binary
            ├── train.bin
            ├── test.bin
            ├── fine_label_names.txt
            └── coarse_label_names.txt

    Citation:

    .. code-block::

        @techreport{Krizhevsky09,
        author       = {Alex Krizhevsky},
        title        = {Learning multiple layers of features from tiny images},
        institution  = {},
        year         = {2009},
        howpublished = {http://www.cs.toronto.edu/~kriz/cifar.html}
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.Cifar100Node(self.dataset_dir, self.usage, self.sampler)


class CityscapesDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses Cityscapes dataset.

    The generated dataset has two columns :py:obj:`[image, task]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`task` is of the uint8 type if task is not 'polygon' otherwise task is
    a string tensor with serialize json.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include 'train', 'test', 'val' or 'all' if quality_mode is 'fine'
            otherwise 'train', 'train_extra', 'val' or 'all'. Default: 'train'.
        quality_mode (str, optional): Acceptable quality_modes include 'fine' or 'coarse'. Default: 'fine'.
        task (str, optional): Acceptable tasks include 'instance',
            'semantic', 'polygon' or 'color'. Default: 'instance'.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` is invalid or does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `dataset_dir` is not exist.
        ValueError: If `task` is invalid.
        ValueError: If `quality_mode` is invalid.
        ValueError: If `usage` is invalid.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> cityscapes_dataset_dir = "/path/to/cityscapes_dataset_directory"
        >>>
        >>> # 1) Get all samples from Cityscapes dataset in sequence
        >>> dataset = ds.CityscapesDataset(dataset_dir=cityscapes_dataset_dir, task="instance", quality_mode="fine",
        ...                                usage="train", shuffle=False, num_parallel_workers=1)
        >>>
        >>> # 2) Randomly select 350 samples from Cityscapes dataset
        >>> dataset = ds.CityscapesDataset(dataset_dir=cityscapes_dataset_dir, num_samples=350, shuffle=True,
        ...                                num_parallel_workers=1)
        >>>
        >>> # 3) Get samples from Cityscapes dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.CityscapesDataset(dataset_dir=cityscapes_dataset_dir, num_shards=2, shard_id=0,
        ...                                num_parallel_workers=1)
        >>>
        >>> # In Cityscapes dataset, each dictionary has keys "image" and "task"

    About Cityscapes dataset:

    The Cityscapes dataset consists of 5000 color images with high quality dense pixel annotations and
    19998 color images with coarser polygonal annotations in 50 cities. There are 30 classes in this
    dataset and the polygonal annotations include dense semantic segmentation and instance segmentation
    for vehicle and people.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    Taking the quality_mode of `fine` as an example.

    .. code-block::

        .
        └── Cityscapes
             ├── leftImg8bit
             |    ├── train
             |    |    ├── aachen
             |    |    |    ├── aachen_000000_000019_leftImg8bit.png
             |    |    |    ├── aachen_000001_000019_leftImg8bit.png
             |    |    |    ├── ...
             |    |    ├── bochum
             |    |    |    ├── ...
             |    |    ├── ...
             |    ├── test
             |    |    ├── ...
             |    ├── val
             |    |    ├── ...
             └── gtFine
                  ├── train
                  |    ├── aachen
                  |    |    ├── aachen_000000_000019_gtFine_color.png
                  |    |    ├── aachen_000000_000019_gtFine_instanceIds.png
                  |    |    ├── aachen_000000_000019_gtFine_labelIds.png
                  |    |    ├── aachen_000000_000019_gtFine_polygons.json
                  |    |    ├── aachen_000001_000019_gtFine_color.png
                  |    |    ├── aachen_000001_000019_gtFine_instanceIds.png
                  |    |    ├── aachen_000001_000019_gtFine_labelIds.png
                  |    |    ├── aachen_000001_000019_gtFine_polygons.json
                  |    |    ├── ...
                  |    ├── bochum
                  |    |    ├── ...
                  |    ├── ...
                  ├── test
                  |    ├── ...
                  └── val
                       ├── ...

    Citation:

    .. code-block::

        @inproceedings{Cordts2016Cityscapes,
        title       = {The Cityscapes Dataset for Semantic Urban Scene Understanding},
        author      = {Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler,
                        Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
        booktitle   = {Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year        = {2016}
        }
    """

    @check_cityscapes_dataset
    def __init__(self, dataset_dir, usage="train", quality_mode="fine", task="instance", num_samples=None,
                 num_parallel_workers=None, shuffle=None, decode=None, sampler=None, num_shards=None,
                 shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.task = task
        self.quality_mode = quality_mode
        self.usage = usage
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.CityscapesNode(self.dataset_dir, self.usage, self.quality_mode, self.task, self.decode, self.sampler)


class CocoDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses COCO dataset.

    CocoDataset supports five kinds of tasks, which are Object Detection, Keypoint Detection, Stuff Segmentation,
    Panoptic Segmentation and Captioning of 2017 Train/Val/Test dataset.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        annotation_file (str): Path to the annotation JSON file.
        task (str, optional): Set the task type for reading COCO data. Supported task types:
            'Detection', 'Stuff', 'Panoptic', 'Keypoint' and 'Captioning'. Default: 'Detection'.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` uration file.
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.
        extra_metadata(bool, optional): Flag to add extra meta-data to row. If True, an additional column will be
            output at the end :py:obj:`[_meta-filename, dtype=string]` . Default: False.
        decrypt (callable, optional): Image decryption function, which accepts the path of the encrypted image file
            and returns the decrypted bytes data. Default: None, no decryption.

    The generated dataset with different task setting has different output columns:

    +-------------------------+----------------------------------------------+
    | `task`                  |   Output column                              |
    +=========================+==============================================+
    | Detection               |   [image, dtype=uint8]                       |
    |                         |                                              |
    |                         |   [bbox, dtype=float32]                      |
    |                         |                                              |
    |                         |   [category_id, dtype=uint32]                |
    |                         |                                              |
    |                         |   [iscrowd, dtype=uint32]                    |
    +-------------------------+----------------------------------------------+
    | Stuff                   |   [image, dtype=uint8]                       |
    |                         |                                              |
    |                         |   [segmentation, dtype=float32]              |
    |                         |                                              |
    |                         |   [iscrowd, dtype=uint32]                    |
    +-------------------------+----------------------------------------------+
    | Keypoint                |   [image, dtype=uint8]                       |
    |                         |                                              |
    |                         |   [keypoints, dtype=float32]                 |
    |                         |                                              |
    |                         |   [num_keypoints, dtype=uint32]              |
    +-------------------------+----------------------------------------------+
    | Panoptic                |   [image, dtype=uint8]                       |
    |                         |                                              |
    |                         |   [bbox, dtype=float32]                      |
    |                         |                                              |
    |                         |   [category_id, dtype=uint32]                |
    |                         |                                              |
    |                         |   [iscrowd, dtype=uint32]                    |
    |                         |                                              |
    |                         |   [area, dtype=uint32]                       |
    +-------------------------+----------------------------------------------+

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        RuntimeError: If parse JSON file failed.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `task` is not in ['Detection', 'Stuff', 'Panoptic', 'Keypoint', 'Captioning'].
        ValueError: If `annotation_file` is not exist.
        ValueError: If `dataset_dir` is not exist.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - Column '[_meta-filename, dtype=string]' won't be output unless an explicit rename dataset op is added
          to remove the prefix('_meta-').
        - Not support `mindspore.dataset.PKSampler` for `sampler` parameter yet.
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> coco_dataset_dir = "/path/to/coco_dataset_directory/images"
        >>> coco_annotation_file = "/path/to/coco_dataset_directory/annotation_file"
        >>>
        >>> # 1) Read COCO data for Detection task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Detection')
        >>>
        >>> # 2) Read COCO data for Stuff task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Stuff')
        >>>
        >>> # 3) Read COCO data for Panoptic task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Panoptic')
        >>>
        >>> # 4) Read COCO data for Keypoint task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Keypoint')
        >>>
        >>> # 5) Read COCO data for Captioning task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Captioning')
        >>>
        >>> # In COCO dataset, each dictionary has keys "image" and "annotation"

    About COCO dataset:

    COCO(Microsoft Common Objects in Context) is a large-scale object detection, segmentation, and captioning dataset
    with several features: Object segmentation, Recognition in context, Superpixel stuff segmentation,
    330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories,
    5 captions per image, 250,000 people with keypoints. In contrast to the popular ImageNet dataset, COCO has fewer
    categories but more instances in per category.

    You can unzip the original COCO-2017 dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── coco_dataset_directory
             ├── train2017
             │    ├── 000000000009.jpg
             │    ├── 000000000025.jpg
             │    ├── ...
             ├── test2017
             │    ├── 000000000001.jpg
             │    ├── 000000058136.jpg
             │    ├── ...
             ├── val2017
             │    ├── 000000000139.jpg
             │    ├── 000000057027.jpg
             │    ├── ...
             └── annotations
                  ├── captions_train2017.json
                  ├── captions_val2017.json
                  ├── instances_train2017.json
                  ├── instances_val2017.json
                  ├── person_keypoints_train2017.json
                  └── person_keypoints_val2017.json

    Citation:

    .. code-block::

        @article{DBLP:journals/corr/LinMBHPRDZ14,
        author        = {Tsung{-}Yi Lin and Michael Maire and Serge J. Belongie and
                        Lubomir D. Bourdev and  Ross B. Girshick and James Hays and
                        Pietro Perona and Deva Ramanan and Piotr Doll{\'{a}}r and C. Lawrence Zitnick},
        title         = {Microsoft {COCO:} Common Objects in Context},
        journal       = {CoRR},
        volume        = {abs/1405.0312},
        year          = {2014},
        url           = {http://arxiv.org/abs/1405.0312},
        archivePrefix = {arXiv},
        eprint        = {1405.0312},
        timestamp     = {Mon, 13 Aug 2018 16:48:13 +0200},
        biburl        = {https://dblp.org/rec/journals/corr/LinMBHPRDZ14.bib},
        bibsource     = {dblp computer science bibliography, https://dblp.org}
        }
    """

    @check_cocodataset
    def __init__(self, dataset_dir, annotation_file, task="Detection", num_samples=None, num_parallel_workers=None,
                 shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None,
                 extra_metadata=False, decrypt=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.annotation_file = annotation_file
        self.task = replace_none(task, "Detection")
        self.decode = replace_none(decode, False)
        self.extra_metadata = extra_metadata
        self.decrypt = decrypt

    def parse(self, children=None):
        return cde.CocoNode(self.dataset_dir, self.annotation_file, self.task, self.decode, self.sampler,
                            self.extra_metadata, self.decrypt)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-list<int> mapping from label name to index.

        Examples:
            >>> coco_dataset_dir = "/path/to/coco_dataset_directory/images"
            >>> coco_annotation_file = "/path/to/coco_dataset_directory/annotation_file"
            >>>
            >>> # Read COCO data for Detection task
            >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
            ...                          annotation_file=coco_annotation_file,
            ...                          task='Detection')
            >>>
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.task not in {"Detection", "Panoptic"}:
            raise NotImplementedError("Only 'Detection' and 'Panoptic' support get_class_indexing.")
        if self._class_indexing is None:
            runtime_getter = self._init_tree_getters()
            self._class_indexing = dict(runtime_getter[0].GetClassIndexing())
        return self._class_indexing


class DIV2KDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses DIV2KDataset dataset.

    The generated dataset has two columns :py:obj:`[hr_image, lr_image]` .
    The tensor of column :py:obj:`hr_image` and the tensor of column :py:obj:`lr_image` are of the uint8 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include 'train', 'valid' or 'all'. Default: 'train'.
        downgrade (str, optional): Acceptable downgrades include 'bicubic', 'unknown', 'mild', 'difficult' or
            'wild'. Default: 'bicubic'.
        scale (str, optional): Acceptable scales include 2, 3, 4 or 8. Default: 2.
            When `downgrade` is 'bicubic', scale can be 2, 3, 4, 8.
            When `downgrade` is 'unknown', scale can only be 2, 3, 4.
            When `downgrade` is 'mild', 'difficult' or 'wild', scale can only be 4.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` is invalid or does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `dataset_dir` is not exist.
        ValueError: If `usage` is invalid.
        ValueError: If `downgrade` is invalid.
        ValueError: If `scale` is invalid.
        ValueError: If `scale` equal to 8 and downgrade not equal to 'bicubic'.
        ValueError: If `downgrade` in ['mild', 'difficult', 'wild'] and `scale` not equal to 4.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> div2k_dataset_dir = "/path/to/div2k_dataset_directory"
        >>>
        >>> # 1) Get all samples from DIV2K dataset in sequence
        >>> dataset = ds.DIV2KDataset(dataset_dir=div2k_dataset_dir, usage="train", scale=2, downgrade="bicubic",
        ...                           shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from DIV2K dataset
        >>> dataset = ds.DIV2KDataset(dataset_dir=div2k_dataset_dir, usage="train", scale=2, downgrade="bicubic",
        ...                           num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from DIV2K dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.DIV2KDataset(dataset_dir=div2k_dataset_dir, usage="train", scale=2, downgrade="bicubic",
        ...                           num_shards=2, shard_id=0)
        >>>
        >>> # In DIV2K dataset, each dictionary has keys "hr_image" and "lr_image"

    About DIV2K dataset:

    The DIV2K dataset consists of 1000 2K resolution images, among which 800 images are for training, 100 images
    are for validation and 100 images are for testing. NTIRE 2017 and NTIRE 2018 include only training dataset
    and validation dataset.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    Take the training set as an example.

    .. code-block::

        .
        └── DIV2K
             ├── DIV2K_train_HR
             |    ├── 0001.png
             |    ├── 0002.png
             |    ├── ...
             ├── DIV2K_train_LR_bicubic
             |    ├── X2
             |    |    ├── 0001x2.png
             |    |    ├── 0002x2.png
             |    |    ├── ...
             |    ├── X3
             |    |    ├── 0001x3.png
             |    |    ├── 0002x3.png
             |    |    ├── ...
             |    └── X4
             |         ├── 0001x4.png
             |         ├── 0002x4.png
             |         ├── ...
             ├── DIV2K_train_LR_unknown
             |    ├── X2
             |    |    ├── 0001x2.png
             |    |    ├── 0002x2.png
             |    |    ├── ...
             |    ├── X3
             |    |    ├── 0001x3.png
             |    |    ├── 0002x3.png
             |    |    ├── ...
             |    └── X4
             |         ├── 0001x4.png
             |         ├── 0002x4.png
             |         ├── ...
             ├── DIV2K_train_LR_mild
             |    ├── 0001x4m.png
             |    ├── 0002x4m.png
             |    ├── ...
             ├── DIV2K_train_LR_difficult
             |    ├── 0001x4d.png
             |    ├── 0002x4d.png
             |    ├── ...
             ├── DIV2K_train_LR_wild
             |    ├── 0001x4w.png
             |    ├── 0002x4w.png
             |    ├── ...
             └── DIV2K_train_LR_x8
                  ├── 0001x8.png
                  ├── 0002x8.png
                  ├── ...

    Citation:

    .. code-block::

        @InProceedings{Agustsson_2017_CVPR_Workshops,
        author    = {Agustsson, Eirikur and Timofte, Radu},
        title     = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        url       = "http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf",
        month     = {July},
        year      = {2017}
        }
    """

    @check_div2k_dataset
    def __init__(self, dataset_dir, usage="train", downgrade="bicubic", scale=2, num_samples=None,
                 num_parallel_workers=None, shuffle=None, decode=None, sampler=None, num_shards=None,
                 shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = usage
        self.scale = scale
        self.downgrade = downgrade
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.DIV2KNode(self.dataset_dir, self.usage, self.downgrade, self.scale, self.decode, self.sampler)


class EMnistDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the EMNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        name (str): Name of splits for this dataset, can be 'byclass', 'bymerge', 'balanced', 'letters', 'digits'
            or 'mnist'.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all'.'train' will read from 60,000
            train samples, 'test' will read from 10,000 test samples, 'all' will read from all 70,000 samples.
            Default: None, will read all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> emnist_dataset_dir = "/path/to/emnist_dataset_directory"
        >>>
        >>> # Read 3 samples from EMNIST dataset
        >>> dataset = ds.EMnistDataset(dataset_dir=emnist_dataset_dir, name="mnist", num_samples=3)
        >>>
        >>> # Note: In emnist_dataset dataset, each dictionary has keys "image" and "label"

    About EMNIST dataset:

    The EMNIST dataset is a set of handwritten character digits derived from the NIST Special
    Database 19 and converted to a 28x28 pixel image format and dataset structure that directly
    matches the MNIST dataset. Further information on the dataset contents and conversion process
    can be found in the paper available at https://arxiv.org/abs/1702.05373v1.

    The numbers of characters and classes of each split of EMNIST are as follows:

    By Class: 814,255 characters and 62 unbalanced classes.
    By Merge: 814,255 characters and 47 unbalanced classes.
    Balanced: 131,600 characters and 47 balanced classes.
    Letters: 145,600 characters and 26 balanced classes.
    Digits: 280,000 characters and 10 balanced classes.
    MNIST: 70,000 characters and 10 balanced classes.

    Here is the original EMNIST dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── mnist_dataset_dir
             ├── emnist-mnist-train-images-idx3-ubyte
             ├── emnist-mnist-train-labels-idx1-ubyte
             ├── emnist-mnist-test-images-idx3-ubyte
             ├── emnist-mnist-test-labels-idx1-ubyte
             ├── ...

    Citation:

    .. code-block::

        @article{cohen_afshar_tapson_schaik_2017,
        title        = {EMNIST: Extending MNIST to handwritten letters},
        DOI          = {10.1109/ijcnn.2017.7966217},
        journal      = {2017 International Joint Conference on Neural Networks (IJCNN)},
        author       = {Cohen, Gregory and Afshar, Saeed and Tapson, Jonathan and Schaik, Andre Van},
        year         = {2017},
        howpublished = {https://www.westernsydney.edu.au/icns/reproducible_research/
                        publication_support_materials/emnist}
        }
    """

    @check_emnist_dataset
    def __init__(self, dataset_dir, name, usage=None, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.name = name
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.EMnistNode(self.dataset_dir, self.name, self.usage, self.sampler)


class FakeImageDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset for generating fake images.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        num_images (int, optional): Number of images to generate in the dataset. Default: 1000.
        image_size (tuple, optional):  Size of the fake image. Default: (224, 224, 3).
        num_classes (int, optional): Number of classes in the dataset. Default: 10.
        base_seed (int, optional): Offsets the index-based random seed used to generate each image. Default: 0.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> # Read 3 samples from FakeImage dataset
        >>> dataset = ds.FakeImageDataset(num_images=1000, image_size=(224,224,3),
        ...                               num_classes=10, base_seed=0, num_samples=3)
    """

    @check_fake_image_dataset
    def __init__(self, num_images=1000, image_size=(224, 224, 3), num_classes=10, base_seed=0, num_samples=None,
                 num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.num_images = num_images
        self.image_size = image_size
        self.num_classes = num_classes
        self.base_seed = base_seed

    def parse(self, children=None):
        return cde.FakeImageNode(self.num_images, self.image_size, self.num_classes, self.base_seed, self.sampler)


class FashionMnistDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the Fashion-MNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all'. 'train' will read from 60,000
            train samples, 'test' will read from 10,000 test samples, 'all' will read from all 70,000 samples.
            Default: None, will read all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> fashion_mnist_dataset_dir = "/path/to/fashion_mnist_dataset_directory"
        >>>
        >>> # Read 3 samples from FASHIONMNIST dataset
        >>> dataset = ds.FashionMnistDataset(dataset_dir=fashion_mnist_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In FASHIONMNIST dataset, each dictionary has keys "image" and "label"

    About Fashion-MNIST dataset:

    Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and
    a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
    We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking
    machine learning algorithms. It shares the same image size and structure of training and testing splits.

    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── fashionmnist_dataset_dir
             ├── t10k-images-idx3-ubyte
             ├── t10k-labels-idx1-ubyte
             ├── train-images-idx3-ubyte
             └── train-labels-idx1-ubyte

    Citation:

    .. code-block::

        @online{xiao2017/online,
          author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
          title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
          date         = {2017-08-28},
          year         = {2017},
          eprintclass  = {cs.LG},
          eprinttype   = {arXiv},
          eprint       = {cs.LG/1708.07747},
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.FashionMnistNode(self.dataset_dir, self.usage, self.sampler)


class FlickrDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses Flickr8k and Flickr30k dataset.

    The generated dataset has two columns :py:obj:`[image, annotation]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`annotation` is a tensor which contains 5 annotations string,
    such as ["a", "b", "c", "d", "e"].

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        annotation_file (str): Path to the root directory that contains the annotation.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: None.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` is not valid or does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `dataset_dir` is not exist.
        ValueError: If `annotation_file` is not exist.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> flickr_dataset_dir = "/path/to/flickr_dataset_directory"
        >>> annotation_file = "/path/to/flickr_annotation_file"
        >>>
        >>> # 1) Get all samples from FLICKR dataset in sequence
        >>> dataset = ds.FlickrDataset(dataset_dir=flickr_dataset_dir,
        ...                            annotation_file=annotation_file,
        ...                            shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from FLICKR dataset
        >>> dataset = ds.FlickrDataset(dataset_dir=flickr_dataset_dir,
        ...                            annotation_file=annotation_file,
        ...                            num_samples=350,
        ...                            shuffle=True)
        >>>
        >>> # 3) Get samples from FLICKR dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.FlickrDataset(dataset_dir=flickr_dataset_dir,
        ...                            annotation_file=annotation_file,
        ...                            num_shards=2,
        ...                            shard_id=0)
        >>>
        >>> # In FLICKR dataset, each dictionary has keys "image" and "annotation"

    About Flickr8k dataset:

    The Flickr8k dataset consists of 8092 color images. There are 40460 annotations in the Flickr8k.token.txt,
    each image has 5 annotations.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── Flickr8k
             ├── Flickr8k_Dataset
             │    ├── 1000268201_693b08cb0e.jpg
             │    ├── 1001773457_577c3a7d70.jpg
             │    ├── ...
             └── Flickr8k.token.txt

    Citation:

    .. code-block::

        @article{DBLP:journals/jair/HodoshYH13,
        author    = {Micah Hodosh and Peter Young and Julia Hockenmaier},
        title     = {Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics},
        journal   = {J. Artif. Intell. Res.},
        volume    = {47},
        pages     = {853--899},
        year      = {2013},
        url       = {https://doi.org/10.1613/jair.3994},
        doi       = {10.1613/jair.3994},
        timestamp = {Mon, 21 Jan 2019 15:01:17 +0100},
        biburl    = {https://dblp.org/rec/journals/jair/HodoshYH13.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
        }

    About Flickr30k dataset:

    The Flickr30k dataset consists of 31783 color images. There are 158915 annotations in
    the results_20130124.token, each image has 5 annotations.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── Flickr30k
             ├── flickr30k-images
             │    ├── 1000092795.jpg
             │    ├── 10002456.jpg
             │    ├── ...
             └── results_20130124.token

    Citation:

    .. code-block::

        @article{DBLP:journals/tacl/YoungLHH14,
        author    = {Peter Young and Alice Lai and Micah Hodosh and Julia Hockenmaier},
        title     = {From image descriptions to visual denotations: New similarity metrics
                     for semantic inference over event descriptions},
        journal   = {Trans. Assoc. Comput. Linguistics},
        volume    = {2},
        pages     = {67--78},
        year      = {2014},
        url       = {https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/229},
        timestamp = {Wed, 17 Feb 2021 21:55:25 +0100},
        biburl    = {https://dblp.org/rec/journals/tacl/YoungLHH14.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
        }
    """

    @check_flickr_dataset
    def __init__(self, dataset_dir, annotation_file, num_samples=None, num_parallel_workers=None, shuffle=None,
                 decode=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.annotation_file = annotation_file
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.FlickrNode(self.dataset_dir, self.annotation_file, self.decode, self.sampler)


class _Flowers102Dataset:
    """
    Mainly for loading Flowers102 Dataset, and return one row each time.
    """

    def __init__(self, dataset_dir, task, usage, decode):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.task = task
        self.usage = usage
        self.decode = decode

        if self.task == "Classification":
            self.column_names = ["image", "label"]
        else:
            self.column_names = ["image", "segmentation", "label"]

        labels_path = os.path.join(self.dataset_dir, "imagelabels.mat")
        setid_path = os.path.join(self.dataset_dir, "setid.mat")
        # minus one to transform 1~102 to 0 ~ 101
        self.labels = (loadmat(labels_path)["labels"][0] - 1).astype(np.uint32)
        self.setid = loadmat(setid_path)

        if self.usage == 'train':
            self.indices = self.setid["trnid"][0].tolist()
        elif self.usage == 'test':
            self.indices = self.setid["tstid"][0].tolist()
        elif self.usage == 'valid':
            self.indices = self.setid["valid"][0].tolist()
        elif self.usage == 'all':
            self.indices = self.setid["trnid"][0].tolist()
            self.indices += self.setid["tstid"][0].tolist()
            self.indices += self.setid["valid"][0].tolist()
        else:
            raise ValueError("Input usage is not within the valid set of ['train', 'valid', 'test', 'all'].")

    def __getitem__(self, index):
        # range: 1 ~ 8189
        image_path = os.path.join(self.dataset_dir, "jpg", "image_" + str(self.indices[index]).zfill(5) + ".jpg")
        if not os.path.exists(image_path):
            raise RuntimeError("Can not find image file: " + image_path)

        if self.decode is True:
            image = np.asarray(Image.open(image_path).convert("RGB"))
        else:
            image = np.fromfile(image_path, dtype=np.uint8)

        label = self.labels[self.indices[index] - 1]

        if self.task == "Segmentation":
            segmentation_path = \
                os.path.join(self.dataset_dir, "segmim", "segmim_" + str(self.indices[index]).zfill(5) + ".jpg")
            if not os.path.exists(segmentation_path):
                raise RuntimeError("Can not find segmentation file: " + segmentation_path)
            if self.decode is True:
                segmentation = np.asarray(Image.open(segmentation_path).convert("RGB"))
            else:
                segmentation = np.fromfile(segmentation_path, dtype=np.uint8)
            return image, segmentation, label

        return image, label

    def __len__(self):
        return len(self.indices)


class Flowers102Dataset(GeneratorDataset):
    """
    A source dataset that reads and parses Flowers102 dataset.

    According to the given `task` configuration, the generated dataset has different output columns:
    - `task` = 'Classification', output columns: `[image, dtype=uint8]` , `[label, dtype=uint32]` .
    - `task` = 'Segmentation',
    output columns: `[image, dtype=uint8]` , `[segmentation, dtype=uint8]` , `[label, dtype=uint32]` .

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        task (str, optional): Specify the 'Classification' or 'Segmentation' task. Default: 'Classification'.
        usage (str, optional): Specify the 'train', 'valid', 'test' part or 'all' parts of dataset.
            Default: 'all', will read all samples.
        num_samples (int, optional): The number of samples to be included in the dataset. Default: None, all images.
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Whether or not to decode the images and segmentations after reading. Default: False.
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This argument must be specified only
            when `num_shards` is also specified.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> flowers102_dataset_dir = "/path/to/flowers102_dataset_directory"
        >>> dataset = ds.Flowers102Dataset(dataset_dir=flowers102_dataset_dir,
        ...                                task="Classification",
        ...                                usage="all",
        ...                                decode=True)

    About Flowers102 dataset:

    Flowers102 dataset consists of 102 flower categories.
    The flowers commonly occur in the United Kingdom.
    Each class consists of between 40 and 258 images.

    Here is the original Flowers102 dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── flowes102_dataset_dir
             ├── imagelabels.mat
             ├── setid.mat
             ├── jpg
                  ├── image_00001.jpg
                  ├── image_00002.jpg
                  ├── ...
             ├── segmim
                  ├── segmim_00001.jpg
                  ├── segmim_00002.jpg
                  ├── ...

    Citation:

    .. code-block::

        @InProceedings{Nilsback08,
          author       = "Maria-Elena Nilsback and Andrew Zisserman",
          title        = "Automated Flower Classification over a Large Number of Classes",
          booktitle    = "Indian Conference on Computer Vision, Graphics and Image Processing",
          month        = "Dec",
          year         = "2008",
        }
    """

    @check_flowers102dataset
    def __init__(self, dataset_dir, task="Classification", usage="all", num_samples=None, num_parallel_workers=1,
                 shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.task = replace_none(task, "Classification")
        self.usage = replace_none(usage, "all")
        self.decode = replace_none(decode, False)
        dataset = _Flowers102Dataset(self.dataset_dir, self.task, self.usage, self.decode)
        super().__init__(dataset, column_names=dataset.column_names, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
        """
        class_names = [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
            "sweet pea", "english marigold", "tiger lily", "moon orchid",
            "bird of paradise", "monkshood", "globe thistle", "snapdragon",
            "colt's foot", "king protea", "spear thistle", "yellow iris",
            "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
            "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
            "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
            "stemless gentian", "artichoke", "sweet william", "carnation",
            "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
            "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
            "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
            "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
            "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
            "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
            "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
            "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
            "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
            "azalea", "water lily", "rose", "thorn apple", "morning glory",
            "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
            "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
            "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
            "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
            "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
            "blackberry lily"
        ]

        class_dict = {}
        for i, class_name in enumerate(class_names):
            class_dict[class_name] = i

        return class_dict


class Food101Dataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses Food101 dataset.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test', or 'all'. 'train' will read
            from 75,750 samples, 'test' will read from 25,250 samples, and 'all' will read all 'train'
            and 'test' samples. Default: None, will be set to 'all'.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. When this argument
            is specified, `num_samples` reflects the maximum sample number of per shard. Default: None.
        shard_id (int, optional): The shard ID within `num_shards` . This argument can only be specified
            when `num_shards` is also specified. Default: None.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If the value of `usage` is not 'train', 'test', or 'all'.
        ValueError: If `dataset_dir` is not exist.

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> food101_dataset_dir = "/path/to/food101_dataset_directory"
        >>>
        >>> # Read 3 samples from Food101 dataset
        >>> dataset = ds.Food101Dataset(dataset_dir=food101_dataset_dir, num_samples=3)

    About Food101 dataset:

    The Food101 is a challenging dataset of 101 food categories, with 101,000 images.
    There are 250 test imgaes and 750 training images in each class. All images were rescaled
    to have a maximum side length of 512 pixels.

    The following is the original Food101 dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── food101_dir
             ├── images
             │    ├── apple_pie
             │    │    ├── 1005649.jpg
             │    │    ├── 1014775.jpg
             │    │    ├──...
             │    ├── baby_back_rips
             │    │    ├── 1005293.jpg
             │    │    ├── 1007102.jpg
             │    │    ├──...
             │    └──...
             └── meta
                  ├── train.txt
                  ├── test.txt
                  ├── classes.txt
                  ├── train.json
                  ├── test.json
                  └── train.txt

    Citation:

    .. code-block::

        @inproceedings{bossard14,
        title     = {Food-101 -- Mining Discriminative Components with Random Forests},
        author    = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
        booktitle = {European Conference on Computer Vision},
        year      = {2014}
        }
    """

    @check_food101_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 decode=False, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.Food101Node(self.dataset_dir, self.usage, self.decode, self.sampler)


class ImageFolderDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads images from a tree of directories.
    All images within one folder have the same label.

    The generated dataset has two columns: :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        extensions (list[str], optional): List of file extensions to be
            included in the dataset. Default: None.
        class_indexing (dict, optional): A str-to-int mapping from folder name to index
            Default: None, the folder names will be sorted
            alphabetically and each class will be given a
            unique index starting from 0.
        decode (bool, optional): Decode the images after reading. Default: False.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.
        decrypt (callable, optional): Image decryption function, which accepts the path of the encrypted image file
            and returns the decrypted bytes data. Default: None, no decryption.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        RuntimeError: If `class_indexing` is not a dictionary.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - The shape of the image column is [image_size] if decode flag is False, or [H,W,C] otherwise.
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in image_folder_dataset_dir with 8 threads
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 num_parallel_workers=8)
        >>>
        >>> # 2) Read all samples (image files) from folder cat and folder dog with label 0 and 1
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 class_indexing={"cat":0, "dog":1})
        >>>
        >>> # 3) Read all samples (image files) in image_folder_dataset_dir with extensions .JPEG
        >>> #    and .png (case sensitive)
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 extensions=[".JPEG", ".png"])

    About ImageFolderDataset:

    You can construct the following directory structure from your dataset files and read by MindSpore's API.

    .. code-block::

        .
        └── image_folder_dataset_directory
             ├── class1
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class2
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class3
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── classN
             ├── ...
    """

    @check_imagefolderdataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None,
                 extensions=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None,
                 decrypt=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.extensions = replace_none(extensions, [])
        self.class_indexing = replace_none(class_indexing, {})
        self.decode = replace_none(decode, False)
        self.decrypt = decrypt

    def parse(self, children=None):
        return cde.ImageFolderNode(self.dataset_dir, self.decode, self.sampler, self.extensions, self.class_indexing,
                                   self.decrypt)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.

        Examples:
            >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
            >>>
            >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir)
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.class_indexing is None or not self.class_indexing:
            runtime_getter = self._init_tree_getters()
            _class_indexing = runtime_getter[0].GetClassIndexing()
            for pair in _class_indexing:
                self.class_indexing[pair[0]] = pair[1][0]
        return self.class_indexing


class KITTIDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the KITTI dataset.

    When `usage` is "train", the generated dataset has multiple columns: :py:obj:`[image, label, truncated,
    occluded, alpha, bbox, dimensions, location, rotation_y]` ; When `usage` is "test", the generated dataset
    has only one column: :py:obj:`[image]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of the uint32 type.
    The tensor of column :py:obj:`truncated` is of the float32 type.
    The tensor of column :py:obj:`occluded` is of the uint32 type.
    The tensor of column :py:obj:`alpha` is of the float32 type.
    The tensor of column :py:obj:`bbox` is of the float32 type.
    The tensor of column :py:obj:`dimensions` is of the float32 type.
    The tensor of column :py:obj:`location` is of the float32 type.
    The tensor of column :py:obj:`rotation_y` is of the float32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train` or `test` . `train` will read 7481
            train samples, `test` will read from 7518 test samples without label. Default: None, will use `train` .
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will include all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards. Default: None. This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `dataset_dir` is not exist.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> kitti_dataset_dir = "/path/to/kitti_dataset_directory"
        >>>
        >>> # 1) Read all KITTI train dataset samples in kitti_dataset_dir in sequence
        >>> dataset = ds.KITTIDataset(dataset_dir=kitti_dataset_dir, usage="train")
        >>>
        >>> # 2) Read then decode all KITTI test dataset samples in kitti_dataset_dir in sequence
        >>> dataset = ds.KITTIDataset(dataset_dir=kitti_dataset_dir, usage="test",
        ...                           decode=True, shuffle=False)

    About KITTI dataset:

    KITTI (Karlsruhe Institute of Technology and Toyota Technological Institute) is one of the most popular
    datasets for use in mobile robotics and autonomous driving. It consists of hours of traffic scenarios
    recorded with a variety of sensor modalities, including high-resolution RGB, grayscale stereo cameras,
    and a 3D laser scanner. Despite its popularity, the dataset itself does not contain ground truth for
    semantic segmentation. However, various researchers have manually annotated parts of the dataset to fit
    their necessities. Álvarez et al. generated ground truth for 323 images from the road detection challenge
    with three classes: road, vehicles and sky. Zhang et al. annotated 252 (140 for training and 112 for testing)
    acquisitions – RGB and Velodyne scans – from the tracking challenge for ten object categories: building, sky,
    road, vegetation, sidewalk, car, pedestrian, cyclist, sign/pole, and fence.

    You can unzip the original KITTI dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── kitti_dataset_directory
            ├── data_object_image_2
            │    ├──training
            │    │    ├──image_2
            │    │    │    ├── 000000000001.jpg
            │    │    │    ├── 000000000002.jpg
            │    │    │    ├── ...
            │    ├──testing
            │    │    ├── image_2
            │    │    │    ├── 000000000001.jpg
            │    │    │    ├── 000000000002.jpg
            │    │    │    ├── ...
            ├── data_object_label_2
            │    ├──training
            │    │    ├──label_2
            │    │    │    ├── 000000000001.jpg
            │    │    │    ├── 000000000002.jpg
            │    │    │    ├── ...

    Citation:

    .. code-block::

        @INPROCEEDINGS{Geiger2012CVPR,
        author={Andreas Geiger and Philip Lenz and Raquel Urtasun},
        title={Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2012}
        }
    """

    @check_kittidataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 decode=False, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "train")
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.KITTINode(self.dataset_dir, self.usage, self.decode, self.sampler)


class KMnistDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the KMNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all' . 'train' will read from 60,000
            train samples, 'test' will read from 10,000 test samples, 'all' will read from all 70,000 samples.
            Default: None, will read all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and sharding are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> kmnist_dataset_dir = "/path/to/kmnist_dataset_directory"
        >>>
        >>> # Read 3 samples from KMNIST dataset
        >>> dataset = ds.KMnistDataset(dataset_dir=kmnist_dataset_dir, num_samples=3)

    About KMNIST dataset:

    KMNIST is a dataset, adapted from Kuzushiji Dataset, as a drop-in replacement for MNIST dataset,
    which is the most famous dataset in the machine learning community.

    Here is the original KMNIST dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── kmnist_dataset_dir
             ├── t10k-images-idx3-ubyte
             ├── t10k-labels-idx1-ubyte
             ├── train-images-idx3-ubyte
             └── train-labels-idx1-ubyte

    Citation:

    .. code-block::

        @online{clanuwat2018deep,
          author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and
                           Alex Lamb and Kazuaki Yamamoto and David Ha},
          title        = {Deep Learning for Classical Japanese Literature},
          date         = {2018-12-03},
          year         = {2018},
          eprintclass  = {cs.CV},
          eprinttype   = {arXiv},
          eprint       = {cs.CV/1812.01718},
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.KMnistNode(self.dataset_dir, self.usage, self.sampler)


class LFWDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the LFW dataset.

    When `task` is 'people', the generated dataset has two columns: :py:obj:`[image, label]`;
    When `task` is 'pairs', the generated dataset has three columns: :py:obj:`[image1, image2, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`image1` is of the uint8 type.
    The tensor of column :py:obj:`image2` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        task (str, optional): Set the task type of reading lfw data, support 'people' and 'pairs'.
            Default: None, means 'people'.
        usage (str, optional): The image split to use, support '10fold', 'train', 'test' and 'all'.
            Default: None, will read samples including train and test.
        image_set (str, optional): Type of image funneling to use, support 'original', 'funneled' or
            'deepfunneled'. Default: None, will use 'funneled'.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards. Default: None. This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> # 1) Read LFW People dataset
        >>> lfw_people_dataset_dir = "/path/to/lfw_people_dataset_directory"
        >>> dataset = ds.LFWDataset(dataset_dir=lfw_people_dataset_dir, task="people", usage="10fold",
        ...                         image_set="original")
        >>>
        >>> # 2) Read LFW Pairs dataset
        >>> lfw_pairs_dataset_dir = "/path/to/lfw_pairs_dataset_directory"
        >>> dataset = ds.LFWDataset(dataset_dir=lfw_pairs_dataset_dir, task="pairs", usage="test", image_set="funneled")

    About LFW dataset:

    LFW (Labelled Faces in the Wild) dataset is one of the most commonly used and widely open datasets in
    the field of face recognition. It was released by Gary B. Huang and his team at Massachusetts Institute
    of Technology in 2007. The dataset includes nearly 50,000 images of 13,233 individuals, which are sourced
    from various internet platforms and contain diverse environmental factors such as different poses, lighting
    conditions, and angles. Most of the images in the dataset are frontal and cover a wide range of ages, genders,
    and ethnicities.

    You can unzip the original LFW dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── lfw_dataset_directory
            ├── lfw
            │    ├──Aaron_Eckhart
            │    │    ├──Aaron_Eckhart_0001.jpg
            │    │    ├──...
            │    ├──Abbas_Kiarostami
            │    │    ├── Abbas_Kiarostami_0001.jpg
            │    │    ├──...
            │    ├──...
            ├── lfw-deepfunneled
            │    ├──Aaron_Eckhart
            │    │    ├──Aaron_Eckhart_0001.jpg
            │    │    ├──...
            │    ├──Abbas_Kiarostami
            │    │    ├── Abbas_Kiarostami_0001.jpg
            │    │    ├──...
            │    ├──...
            ├── lfw_funneled
            │    ├──Aaron_Eckhart
            │    │    ├──Aaron_Eckhart_0001.jpg
            │    │    ├──...
            │    ├──Abbas_Kiarostami
            │    │    ├── Abbas_Kiarostami_0001.jpg
            │    │    ├──...
            │    ├──...
            ├── lfw-names.txt
            ├── pairs.txt
            ├── pairsDevTest.txt
            ├── pairsDevTrain.txt
            ├── people.txt
            ├── peopleDevTest.txt
            ├── peopleDevTrain.txt

    Citation:

    .. code-block::

        @TechReport{LFWTech,
            title={LFW: A Database for Studying Recognition in Unconstrained Environments},
            author={Gary B. Huang and Manu Ramesh and Tamara Berg and Erik Learned-Miller},
            institution ={University of Massachusetts, Amherst},
            year={2007}
            number={07-49},
            month={October},
            howpublished = {http://vis-www.cs.umass.edu/lfw}
        }
    """

    @check_lfw_dataset
    def __init__(self, dataset_dir, task=None, usage=None, image_set=None, num_samples=None, num_parallel_workers=None,
                 shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.task = replace_none(task, "people")
        self.usage = replace_none(usage, "all")
        self.image_set = replace_none(image_set, "funneled")
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.LFWNode(self.dataset_dir, self.task, self.usage, self.image_set, self.decode, self.sampler)


class LSUNDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the LSUN dataset.

    The generated dataset has two columns: :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train` , `test` , `valid` or `all`
            Default: None, will be set to `all` .
        classes (Union[str, list[str]], optional): Choose the specific classes to load. Default: None, means loading
            all classes in root directory.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards. Default: None. This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If 'sampler' and 'shuffle' are specified at the same time.
        RuntimeError: If 'sampler' and sharding are specified at the same time.
        RuntimeError: If 'num_shards' is specified but 'shard_id' is None.
        RuntimeError: If 'shard_id' is specified but 'num_shards' is None.
        ValueError: If 'shard_id' is invalid (< 0 or >= `num_shards` ).
        ValueError: If 'usage' or 'classes' is invalid (not in specific types).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> lsun_dataset_dir = "/path/to/lsun_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in lsun_dataset_dir with 8 threads
        >>> dataset = ds.LSUNDataset(dataset_dir=lsun_dataset_dir,
        ...                          num_parallel_workers=8)
        >>>
        >>> # 2) Read all train samples (image files) from folder "bedroom" and "classroom"
        >>> dataset = ds.LSUNDataset(dataset_dir=lsun_dataset_dir, usage="train",
        ...                          classes=["bedroom", "classroom"])

    About LSUN dataset:

    The LSUN (Large-Scale Scene Understanding) is a large-scale dataset used for indoors scene
    understanding. It was originally launched by Stanford University in 2015 with the aim of
    providing a challenging and diverse dataset for research in computer vision and machine
    learning. The main application of this dataset for research is indoor scene analysis.

    This dataset contains ten different categories of scenes, including bedrooms, living rooms,
    restaurants, lounges, studies, kitchens, bathrooms, corridors, children's room, and outdoors.
    Each category contains tens of thousands of images from different perspectives, and these
    images are high-quality, high-resolusion real-world images.

    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── lsun_dataset_directory
            ├── test
            │    ├── ...
            ├── bedroom_train
            │    ├── 1_1.jpg
            │    ├── 1_2.jpg
            ├── bedroom_val
            │    ├── ...
            ├── classroom_train
            │    ├── ...
            ├── classroom_val
            │    ├── ...

    Citation:

    .. code-block::

        article{yu15lsun,
            title={LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop},
            author={Yu, Fisher and Zhang, Yinda and Song, Shuran and Seff, Ari and Xiao, Jianxiong},
            journal={arXiv preprint arXiv:1506.03365},
            year={2015}
        }
    """

    @check_lsun_dataset
    def __init__(self, dataset_dir, usage=None, classes=None, num_samples=None, num_parallel_workers=None,
                 shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")
        self.classes = replace_none(classes, [])
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.LSUNNode(self.dataset_dir, self.usage, self.classes, self.decode, self.sampler)


class ManifestDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset for reading images from a Manifest file.

    The generated dataset has two columns: :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of uint64 type.

    Args:
        dataset_file (str): File to be read.
        usage (str, optional): Acceptable usages include 'train', 'eval' and 'inference'. Default: 'train'.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will include all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, will use value set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        class_indexing (dict, optional): A str-to-int mapping from label name to index.
            Default: None, the folder names will be sorted alphabetically and each
            class will be given a unique index starting from 0.
        decode (bool, optional): decode the images after reading. Default: False.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the max number of samples per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If dataset_files are not valid or do not exist.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        RuntimeError: If class_indexing is not a dictionary.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - The shape of the image column is [image_size] if decode flag is False, or [H,W,C] otherwise.
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> manifest_dataset_dir = "/path/to/manifest_dataset_file"
        >>>
        >>> # 1) Read all samples specified in manifest_dataset_dir dataset with 8 threads for training
        >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, usage="train", num_parallel_workers=8)
        >>>
        >>> # 2) Read samples (specified in manifest_file.manifest) for shard 0 in a 2-way distributed training setup
        >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, num_shards=2, shard_id=0)

    About Manifest dataset:

    Manifest file contains a list of files included in a dataset, including basic file info such as File name and File
    ID, along with extended file metadata. Manifest is a data format file supported by Huawei Modelarts. For details,
    see `Specifications for Importing the Manifest File <https://support.huaweicloud.com/engineers-modelarts/
    modelarts_23_0009.html>`_ .

    .. code-block::

        .
        └── manifest_dataset_directory
            ├── train
            │    ├── 1.JPEG
            │    ├── 2.JPEG
            │    ├── ...
            ├── eval
            │    ├── 1.JPEG
            │    ├── 2.JPEG
            │    ├── ...
    """

    @check_manifestdataset
    def __init__(self, dataset_file, usage="train", num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_file = dataset_file
        self.decode = replace_none(decode, False)
        self.usage = replace_none(usage, "train")
        self.class_indexing = replace_none(class_indexing, {})

    def parse(self, children=None):
        return cde.ManifestNode(self.dataset_file, self.usage, self.sampler, self.class_indexing, self.decode)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.

        Examples:
            >>> manifest_dataset_dir = "/path/to/manifest_dataset_file"
            >>>
            >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir)
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.class_indexing is None or not self.class_indexing:
            if self._class_indexing is None:
                runtime_getter = self._init_tree_getters()
                self._class_indexing = runtime_getter[0].GetClassIndexing()
            self.class_indexing = {}
            for pair in self._class_indexing:
                self.class_indexing[pair[0]] = pair[1][0]
        return self.class_indexing


class MnistDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the MNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all' . 'train' will read from 60,000
            train samples, 'test' will read from 10,000 test samples, 'all' will read from all 70,000 samples.
            Default: None, will read all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, will use value set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `usage` is not 'train'、'test' or 'all'.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but shard_id is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
        >>>
        >>> # Read 3 samples from MNIST dataset
        >>> dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In mnist_dataset dataset, each dictionary has keys "image" and "label"

    About MNIST dataset:

    The MNIST database of handwritten digits has a training set of 60,000 examples,
    and a test set of 10,000 examples. It is a subset of a larger set available from
    NIST. The digits have been size-normalized and centered in a fixed-size image.

    Here is the original MNIST dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── mnist_dataset_dir
             ├── t10k-images-idx3-ubyte
             ├── t10k-labels-idx1-ubyte
             ├── train-images-idx3-ubyte
             └── train-labels-idx1-ubyte

    Citation:

    .. code-block::

        @article{lecun2010mnist,
        title        = {MNIST handwritten digit database},
        author       = {LeCun, Yann and Cortes, Corinna and Burges, CJ},
        journal      = {ATT Labs [Online]},
        volume       = {2},
        year         = {2010},
        howpublished = {http://yann.lecun.com/exdb/mnist}
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.MnistNode(self.dataset_dir, self.usage, self.sampler)


class OmniglotDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the Omniglot dataset.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        background (bool, optional): Whether to create dataset from the "background" set.
            Otherwise create from the "evaluation" set. Default: None, set to True.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards. Default: None. This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `sharding` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a sampler. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> omniglot_dataset_dir = "/path/to/omniglot_dataset_directory"
        >>> dataset = ds.OmniglotDataset(dataset_dir=omniglot_dataset_dir,
        ...                              num_parallel_workers=8)

    About Omniglot dataset:

    The Omniglot dataset is designed for developing more human-like learning algorithms. It contains 1623 different
    handwritten characters from 50 different alphabets. Each of the 1623 characters was drawn online via Amazon's
    Mechanical Turk by 20 different people. Each image is paired with stroke data, a sequences of [x, y, t] coordinates
    with time in milliseconds.

    You can unzip the original Omniglot dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── omniglot_dataset_directory
             ├── images_background/
             │    ├── character_class1/
             ├    ├──── 01.jpg
             │    ├──── 02.jpg
             │    ├── character_class2/
             ├    ├──── 01.jpg
             │    ├──── 02.jpg
             │    ├── ...
             ├── images_evaluation/
             │    ├── character_class1/
             ├    ├──── 01.jpg
             │    ├──── 02.jpg
             │    ├── character_class2/
             ├    ├──── 01.jpg
             │    ├──── 02.jpg
             │    ├── ...

    Citation:

    .. code-block::

        @article{lake2015human,
            title={Human-level concept learning through probabilistic program induction},
            author={Lake, Brenden M and Salakhutdinov, Ruslan and Tenenbaum, Joshua B},
            journal={Science},
            volume={350},
            number={6266},
            pages={1332--1338},
            year={2015},
            publisher={American Association for the Advancement of Science}
        }
    """

    @check_omniglotdataset
    def __init__(self, dataset_dir, background=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 decode=False, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.background = replace_none(background, True)
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.OmniglotNode(self.dataset_dir, self.background, self.decode, self.sampler)


class PhotoTourDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the PhotoTour dataset.

    According to the given `usage` configuration, the generated dataset has different output columns:
    - `usage` = 'train', output columns: `[image, dtype=uint8]` .
    - `usage` ≠ 'train', output columns: `[image1, dtype=uint8]` , `[image2, dtype=uint8]` , `[matches, dtype=uint32]` .

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        name (str): Name of the dataset to load,
            should be one of 'notredame', 'yosemite', 'liberty', 'notredame_harris',
            'yosemite_harris' or 'liberty_harris'.
        usage (str, optional): Usage of the dataset, can be 'train' or 'test'. Default: None, will be set to 'train'.
            When usage is 'train', number of samples for each `name` is
            {'notredame': 468159, 'yosemite': 633587, 'liberty': 450092, 'liberty_harris': 379587,
            'yosemite_harris': 450912, 'notredame_harris': 325295}.
            When usage is 'test', will read 100,000 samples for testing.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` ..
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `dataset_dir` is not exist.
        ValueError: If `usage` is not in ["train", "test"].
        ValueError: If name is not in ["notredame", "yosemite", "liberty",
            "notredame_harris", "yosemite_harris", "liberty_harris"].
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive. The table
          below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 64 64 1
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> # Read 3 samples from PhotoTour dataset.
        >>> dataset = ds.PhotoTourDataset(dataset_dir="/path/to/photo_tour_dataset_directory",
        ...                               name='liberty', usage='train', num_samples=3)

    About PhotoTour dataset:

    The data is taken from Photo Tourism reconstructions from Trevi Fountain (Rome), Notre Dame (Paris) and Half
    Dome (Yosemite). Each dataset consists of a series of corresponding patches, which are obtained by projecting
    3D points from Photo Tourism reconstructions back into the original images.

    The dataset consists of 1024 x 1024 bitmap (.bmp) images, each containing a 16 x 16 array of image patches.
    Each patch is sampled as 64 x 64 grayscale, with a canonical scale and orientation. For details of how the scale
    and orientation is established, please see the paper. An associated metadata file info.txt contains the match
    information. Each row of info.txt corresponds to a separate patch, with the patches ordered from left to right and
    top to bottom in each bitmap image. The first number on each row of info.txt is the 3D point ID from which that
    patch was sampled -- patches with the same 3D point ID are projected from the same 3D point (into different images).
    The second number in info.txt corresponds to the image from which the patch was sampled, and is not used at present.

    You can unzip the original PhotoTour dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── photo_tour_dataset_directory
            ├── liberty/
            │    ├── info.txt                 // two columns: 3D_point_ID, unused
            │    ├── m50_100000_100000_0.txt  // seven columns: patch_ID1, 3D_point_ID1, unused1,
            │    │                            // patch_ID2, 3D_point_ID2, unused2, unused3
            │    ├── patches0000.bmp          // 1024*1024 pixels, with 16 * 16 patches.
            │    ├── patches0001.bmp
            │    ├── ...
            ├── yosemite/
            │    ├── ...
            ├── notredame/
            │    ├── ...
            ├── liberty_harris/
            │    ├── ...
            ├── yosemite_harris/
            │    ├── ...
            ├── notredame_harris/
            │    ├── ...

    Citation:

    .. code-block::

        @INPROCEEDINGS{4269996,
            author={Winder, Simon A. J. and Brown, Matthew},
            booktitle={2007 IEEE Conference on Computer Vision and Pattern Recognition},
            title={Learning Local Image Descriptors},
            year={2007},
            volume={},
            number={},
            pages={1-8},
            doi={10.1109/CVPR.2007.382971}
        }
    """

    @check_photo_tour_dataset
    def __init__(self, dataset_dir, name, usage=None, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.name = name
        self.usage = replace_none(usage, "train")

    def parse(self, children=None):
        return cde.PhotoTourNode(self.dataset_dir, self.name, self.usage, self.sampler)


class Places365Dataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the Places365 dataset.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train-standard', 'train-challenge' or 'val'.
            Default: None, will be set to 'train-standard'.
        small (bool, optional): Use 256 * 256 images (True) or high resolution images (False). Default: False.
        decode (bool, optional): Decode the images after reading. Default: False.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, will use value set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `usage` is not in ["train-standard", "train-challenge", "val"].

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> place365_dataset_dir = "/path/to/place365_dataset_directory"
        >>>
        >>> # Read 3 samples from Places365 dataset
        >>> dataset = ds.Places365Dataset(dataset_dir=place365_dataset_dir, usage='train-standard',
        ...                               small=True, decode=True, num_samples=3)

    About Places365 dataset:

    Convolutional neural networks (CNNs) trained on the Places2 Database can be used for scene recognition as well as
    generic deep scene features for visual recognition.

    The author releases the data of Places365-Standard and the data of Places365-Challenge to the public.
    Places365-Standard is the core set of Places2 Database, which has been used to train the Places365-CNNs. The author
    will add other kinds of annotation on the Places365-Standard in the future. Places365-Challenge is the competition
    set of Places2 Database, which has 6.2 million extra images compared to the Places365-Standard.
    The Places365-Challenge will be used for the Places Challenge 2016.

    You can unzip the original Places365 dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── categories_places365
            ├── places365_train-standard.txt
            ├── places365_train-challenge.txt
            ├── val_large/
            │    ├── Places365_val_00000001.jpg
            │    ├── Places365_val_00000002.jpg
            │    ├── Places365_val_00000003.jpg
            │    ├── ...
            ├── val_256/
            │    ├── ...
            ├── data_large_standard/
            │    ├── ...
            ├── data_256_standard/
            │    ├── ...
            ├── data_large_challenge/
            │    ├── ...
            ├── data_256_challenge /
            │    ├── ...

    Citation:

    .. code-block::

        article{zhou2017places,
            title={Places: A 10 million Image Database for Scene Recognition},
            author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
            journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
            year={2017},
            publisher={IEEE}
        }
    """

    @check_places365_dataset
    def __init__(self, dataset_dir, usage=None, small=True, decode=False, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = os.path.abspath(dataset_dir)
        self.usage = replace_none(usage, "train-standard")
        self.small = small
        self.decode = decode

    def parse(self, children=None):
        return cde.Places365Node(self.dataset_dir, self.usage, self.small, self.decode, self.sampler)


class QMnistDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the QMNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test', 'test10k', 'test50k', 'nist'
            or 'all'. Default: None, will read all samples.
        compat (bool, optional): Whether the label for each example is class number (compat=True) or the full QMNIST
            information (compat=False). Default: True.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, will use value set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> qmnist_dataset_dir = "/path/to/qmnist_dataset_directory"
        >>>
        >>> # Read 3 samples from QMNIST train dataset
        >>> dataset = ds.QMnistDataset(dataset_dir=qmnist_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In QMNIST dataset, each dictionary has keys "image" and "label"

    About QMNIST dataset:

    The QMNIST dataset was generated from the original data found in the NIST Special Database 19 with the goal to
    match the MNIST preprocessing as closely as possible.
    Through an iterative process, researchers tried to generate an additional 50k images of MNIST-like data.
    They started with a reconstruction process given in the paper and used the Hungarian algorithm to find the best
    matches between the original MNIST samples and their reconstructed samples.

    Here is the original QMNIST dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── qmnist_dataset_dir
             ├── qmnist-train-images-idx3-ubyte
             ├── qmnist-train-labels-idx2-int
             ├── qmnist-test-images-idx3-ubyte
             ├── qmnist-test-labels-idx2-int
             ├── xnist-images-idx3-ubyte
             └── xnist-labels-idx2-int

    Citation:

    .. code-block::

        @incollection{qmnist-2019,
           title = "Cold Case: The Lost MNIST Digits",
           author = "Chhavi Yadav and L\'{e}on Bottou",\
           booktitle = {Advances in Neural Information Processing Systems 32},
           year = {2019},
           publisher = {Curran Associates, Inc.},
        }
    """

    @check_qmnist_dataset
    def __init__(self, dataset_dir, usage=None, compat=True, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")
        self.compat = compat

    def parse(self, children=None):
        return cde.QMnistNode(self.dataset_dir, self.usage, self.compat, self.sampler)


class RandomDataset(SourceDataset, VisionBaseDataset):
    """
    A source dataset that generates random data.

    Args:
        total_rows (int, optional): Number of samples for the dataset to generate.
            Default: None, number of samples is random.
        schema (Union[str, Schema], optional): Data format policy, which specifies the data types and shapes of the data
            column to be read. Both JSON file path and objects constructed by mindspore.dataset.Schema are acceptable.
            Default: None.
        columns_list (list[str], optional): List of column names of the dataset.
            Default: None, the columns will be named like this "c0", "c1", "c2" etc.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, all samples.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, 'num_samples' reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.

    Raises:
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        TypeError: If `total_rows` is not of type int.
        TypeError: If `num_shards` is not of type int.
        TypeError: If `num_parallel_workers` is not of type int.
        TypeError: If `shuffle` is not of type bool.
        TypeError: If `columns_list` is not of type list.

    Examples:
        >>> from mindspore import dtype as mstype
        >>> import mindspore.dataset as ds
        >>>
        >>> schema = ds.Schema()
        >>> schema.add_column('image', de_type=mstype.uint8, shape=[2])
        >>> schema.add_column('label', de_type=mstype.uint8, shape=[1])
        >>> # apply dataset operations
        >>> ds1 = ds.RandomDataset(schema=schema, total_rows=50, num_parallel_workers=4)
    """

    @check_random_dataset
    def __init__(self, total_rows=None, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None,
                 cache=None, shuffle=None, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.total_rows = replace_none(total_rows, 0)
        if self.total_rows != 0 and self.num_samples != 0:
            self.total_rows = min(self.total_rows, self.num_samples)
        if self.num_samples != 0:
            self.total_rows = self.num_samples
        if schema is not None:
            self.total_rows = replace_none(total_rows, Schema.get_num_rows(schema))
        self.schema = replace_none(schema, "")
        self.columns_list = replace_none(columns_list, [])

    def parse(self, children=None):
        schema = self.schema.cpp_schema if isinstance(self.schema, Schema) else self.schema
        return cde.RandomNode(self.total_rows, schema, self.columns_list)


class RenderedSST2Dataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses RenderedSST2 dataset.

    The generated dataset has two columns: :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'val', 'test' or 'all'.
            Default: None, will read all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will include all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Whether or not to decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard. Default: None.
        shard_id (int, optional): The shard ID within `num_shards` . This
            argument can only be specified when `num_shards` is also specified. Default: None.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        ValueError: If `usage` is not 'train', 'test', 'val' or 'all'.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> rendered_sst2_dataset_dir = "/path/to/rendered_sst2_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in rendered_sst2_dataset_dir with 8 threads
        >>> dataset = ds.RenderedSST2Dataset(dataset_dir=rendered_sst2_dataset_dir,
        ...                                  usage="all", num_parallel_workers=8)

    About RenderedSST2Dataset:

    Rendered SST2 is an image classification dataset which was generated by rendering sentences in the Standford
    Sentiment Treebank v2 dataset. There are three splits in this dataset and each split contains two classes
    (positive and negative): a train split containing 6920 images (3610 positive and 3310 negative), a validation
    split containing 872 images (444 positive and 428 negative), and a test split containing 1821 images
    (909 positive and 912 negative).

    Here is the original RenderedSST2 dataset structure.
    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── rendered_sst2_dataset_directory
             ├── train
             │    ├── negative
             │    │    ├── 0001.jpg
             │    │    ├── 0002.jpg
             │    │    ...
             │    └── positive
             │         ├── 0001.jpg
             │         ├── 0002.jpg
             │         ...
             ├── test
             │    ├── negative
             │    │    ├── 0001.jpg
             │    │    ├── 0002.jpg
             │    │    ...
             │    └── positive
             │         ├── 0001.jpg
             │         ├── 0002.jpg
             │         ...
             └── valid
                  ├── negative
                  │    ├── 0001.jpg
                  │    ├── 0002.jpg
                  │    ...
                  └── positive
                       ├── 0001.jpg
                       ├── 0002.jpg
                       ...

    Citation:

    .. code-block::

        @inproceedings{socher-etal-2013-recursive,
            title     = {Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank},
            author    = {Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning,
                          Christopher D. and Ng, Andrew and Potts, Christopher},
            booktitle = {Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing},
            month     = oct,
            year      = {2013},
            address   = {Seattle, Washington, USA},
            publisher = {Association for Computational Linguistics},
            url       = {https://www.aclweb.org/anthology/D13-1170},
            pages     = {1631--1642},
        }
    """

    @check_rendered_sst2_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 decode=False, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.RenderedSST2Node(self.dataset_dir, self.usage, self.decode, self.sampler)


class _SBDataset:
    """
    Dealing with the data file with .mat extension, and return one row in tuple (image, task) each time.
    """

    def __init__(self, dataset_dir, task, usage, decode):
        self.column_list = ['image', 'task']
        self.task = task
        self.images_path = os.path.join(dataset_dir, 'img')
        self.cls_path = os.path.join(dataset_dir, 'cls')
        self._loadmat = loadmat
        self.categories = 20
        self.decode = replace_none(decode, False)

        if usage == "all":
            image_names = []
            for item in ["train", "val"]:
                usage_path = os.path.join(dataset_dir, item + '.txt')
                if not os.path.exists(usage_path):
                    raise FileNotFoundError("SBDataset: {0} not found".format(usage_path))
                with open(usage_path, 'r') as f:
                    image_names += [x.strip() for x in f.readlines()]
        else:
            usage_path = os.path.join(dataset_dir, usage + '.txt')
            if not os.path.exists(usage_path):
                raise FileNotFoundError("SBDataset: {0} not found".format(usage_path))
            with open(usage_path, 'r') as f:
                image_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.images_path, i + ".jpg") for i in image_names]
        self.clss = [os.path.join(self.cls_path, i + ".mat") for i in image_names]

        if len(self.images) != len(self.clss):
            raise ValueError("SBDataset: images count not equal to cls count")

        self._get_data = self._get_boundaries_data if self.task == "Boundaries" else self._get_segmentation_data
        self._get_item = self._get_decode_item if self.decode else self._get_undecode_item

    def _get_boundaries_data(self, mat_path):
        mat_data = self._loadmat(mat_path)
        return np.concatenate([np.expand_dims(mat_data['GTcls'][0][self.task][0][i][0].toarray(), axis=0)
                               for i in range(self.categories)], axis=0)

    def _get_segmentation_data(self, mat_path):
        mat_data = self._loadmat(mat_path)
        return Image.fromarray(mat_data['GTcls'][0][self.task][0])

    def _get_decode_item(self, idx):
        return Image.open(self.images[idx]).convert('RGB'), self._get_data(self.clss[idx])

    def _get_undecode_item(self, idx):
        return np.fromfile(self.images[idx], dtype=np.uint8), self._get_data(self.clss[idx])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self._get_item(idx)


class SBDataset(GeneratorDataset):
    """
    A source dataset that reads and parses Semantic Boundaries Dataset.

    By configuring the 'Task' parameter, the generated dataset has different output columns.

    - 'task' = 'Boundaries' , there are two output columns: the 'image' column has the data type uint8 and
      the 'label' column contains one image of the data type uint8.
    - 'task' = 'Segmentation' , there are two output columns: the 'image' column has the data type uint8 and
      the 'label' column contains 20 images of the data type uint8.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        task (str, optional): Acceptable tasks include 'Boundaries' or 'Segmentation'. Default: 'Boundaries'.
        usage (str, optional): Acceptable usages include 'train', 'val', 'train_noval' and 'all'. Default: 'all'.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: 1, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: None.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.

    Raises:
        RuntimeError: If `dataset_dir` is not valid or does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `dataset_dir` is not exist.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `task` is not in ['Boundaries', 'Segmentation'].
        ValueError: If `usage` is not in ['train', 'val', 'train_noval', 'all'].
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a sampler. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> sb_dataset_dir = "/path/to/sb_dataset_directory"
        >>>
        >>> # 1) Get all samples from Semantic Boundaries Dataset in sequence
        >>> dataset = ds.SBDataset(dataset_dir=sb_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from Semantic Boundaries Dataset
        >>> dataset = ds.SBDataset(dataset_dir=sb_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from Semantic Boundaries Dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.SBDataset(dataset_dir=sb_dataset_dir, num_shards=2, shard_id=0)
        >>>
        >>> # In Semantic Boundaries Dataset, each dictionary has keys "image" and "task"

    About Semantic Boundaries Dataset:

    The Semantic Boundaries Dataset consists of 11355 color images. There are 8498 images' name in the train.txt,
    2857 images' name in the val.txt and 5623 images' name in the train_noval.txt. The category cls/
    contains the Segmentation and Boundaries results of category-level, the category inst/ contains the
    Segmentation and Boundaries results of instance-level.

    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

         .
         └── benchmark_RELEASE
              ├── dataset
              ├── img
              │    ├── 2008_000002.jpg
              │    ├── 2008_000003.jpg
              │    ├── ...
              ├── cls
              │    ├── 2008_000002.mat
              │    ├── 2008_000003.mat
              │    ├── ...
              ├── inst
              │    ├── 2008_000002.mat
              │    ├── 2008_000003.mat
              │    ├── ...
              ├── train.txt
              └── val.txt

    .. code-block::

        @InProceedings{BharathICCV2011,
            author       = "Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and
                            Subhransu Maji and Jitendra Malik",
            title        = "Semantic Contours from Inverse Detectors",
            booktitle    = "International Conference on Computer Vision (ICCV)",
            year         = "2011",
        }
    """

    @check_sb_dataset
    def __init__(self, dataset_dir, task='Boundaries', usage='all', num_samples=None, num_parallel_workers=1,
                 shuffle=None, decode=None, sampler=None, num_shards=None, shard_id=None):
        dataset = _SBDataset(dataset_dir, task, usage, decode)
        super().__init__(dataset, column_names=dataset.column_list, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)


class SBUDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the SBU dataset.

    The generated dataset has two columns :py:obj:`[image, caption]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`caption` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, will use value set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> sbu_dataset_dir = "/path/to/sbu_dataset_directory"
        >>> # Read 3 samples from SBU dataset
        >>> dataset = ds.SBUDataset(dataset_dir=sbu_dataset_dir, num_samples=3)

    About SBU dataset:

    SBU dataset is a large captioned photo collection.
    It contains one million images with associated visually relevant captions.

    You should manually download the images using official download.m by replacing 'urls{i}(24, end)' with
    'urls{i}(24:1:end)' and keep the directory as below.

    .. code-block::

        .
        └─ dataset_dir
           ├── SBU_captioned_photo_dataset_captions.txt
           ├── SBU_captioned_photo_dataset_urls.txt
           └── sbu_images
               ├── m_3326_3596303505_3ce4c20529.jpg
               ├── ......
               └── m_2522_4182181099_c3c23ab1cc.jpg

    Citation:

    .. code-block::

        @inproceedings{Ordonez:2011:im2text,
          Author    = {Vicente Ordonez and Girish Kulkarni and Tamara L. Berg},
          Title     = {Im2Text: Describing Images Using 1 Million Captioned Photographs},
          Booktitle = {Neural Information Processing Systems ({NIPS})},
          Year      = {2011},
        }
    """

    @check_sbu_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.SBUNode(self.dataset_dir, self.decode, self.sampler)


class SemeionDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses Semeion dataset.

    The generated dataset has two columns :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> semeion_dataset_dir = "/path/to/semeion_dataset_directory"
        >>>
        >>> # 1) Get all samples from SEMEION dataset in sequence
        >>> dataset = ds.SemeionDataset(dataset_dir=semeion_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 10 samples from SEMEION dataset
        >>> dataset = ds.SemeionDataset(dataset_dir=semeion_dataset_dir, num_samples=10, shuffle=True)
        >>>
        >>> # 3) Get samples from SEMEION dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.SemeionDataset(dataset_dir=semeion_dataset_dir, num_shards=2, shard_id=0)
        >>>
        >>> # In SEMEION dataset, each dictionary has keys: image, label.

    About SEMEION dataset:

    The dataset was created by Tactile Srl, Brescia, Italy (http://www.tattile.it) and donated in 1994
    to Semeion Research Center of Sciences of Communication, Rome, Italy (http://www.semeion.it),
    for machine learning research.

    This dataset consists of 1593 records (rows) and 256 attributes (columns). Each record represents
    a handwritten digit, originally scanned with a resolution of 256 grey scale. Each pixel of the each
    original scanned image was first stretched, and after scaled between 0 and 1
    (setting to 0 every pixel whose value was under the value 127 of the grey scale (127 included)
    and setting to 1 each pixel whose original value in the grey scale was over 127). Finally, each binary image
    was scaled again into a 16x16 square box (the final 256 binary attributes).

    .. code-block::

        .
        └── semeion_dataset_dir
            └──semeion.data
            └──semeion.names

    Citation:

    .. code-block::

        @article{
          title={The Theory of Independent Judges, in Substance Use & Misuse 33(2)1998, pp 439-461},
          author={M Buscema, MetaNet},
        }
    """

    @check_semeion_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir

    def parse(self, children=None):
        return cde.SemeionNode(self.dataset_dir, self.sampler)


class STL10Dataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses STL10 dataset.

    The generated dataset has two columns: :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of int32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test',
            'unlabeled', 'train+unlabeled' or 'all' . 'train' will read from 5,000
            train samples, 'test' will read from 8,000 test samples,
            'unlabeled' will read from all 100,000 samples, and 'train+unlabeled'
            will read from 105000 samples, 'all' will read all the samples
            Default: None, all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` is not valid or does not exist or does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `usage` is invalid.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> stl10_dataset_dir = "/path/to/stl10_dataset_directory"
        >>>
        >>> # 1) Get all samples from STL10 dataset in sequence
        >>> dataset = ds.STL10Dataset(dataset_dir=stl10_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from STL10 dataset
        >>> dataset = ds.STL10Dataset(dataset_dir=stl10_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from STL10 dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.STL10Dataset(dataset_dir=stl10_dataset_dir, num_shards=2, shard_id=0)

    About STL10 dataset:

    STL10 dataset consists of 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
    Images are 96x96 pixels, color.
    500 training images, 800 test images per class and 100000 unlabeled images.
    Labels are 0-indexed, and unlabeled images have -1 as their labels.

    Here is the original STL10 dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── stl10_dataset_dir
             ├── train_X.bin
             ├── train_y.bin
             ├── test_X.bin
             ├── test_y.bin
             └── unlabeled_X.bin

    Citation of STL10 dataset:

    .. code-block::

        @techreport{Coates10,
        author       = {Adam Coates},
        title        = {Learning multiple layers of features from tiny images},
        year         = {20010},
        howpublished = {https://cs.stanford.edu/~acoates/stl10/},
        description  = {The STL-10 dataset consists of 96x96 RGB images in 10 classes,
                        with 500 training images and 800 testing images per class.
                        There are 5000 training images and 8000 test images.
                        It also has 100000 unlabeled images for unsupervised learning.
                        These examples are extracted from a similar but broader distribution of images.
                        }
        }
    """

    @check_stl10_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.STL10Node(self.dataset_dir, self.usage, self.sampler)


class SUN397Dataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses SUN397 dataset.

    The generated dataset has two columns: :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Whether or not to decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard. Default: None.
        shard_id (int, optional): The shard ID within `num_shards` . This
            argument can only be specified when `num_shards` is also specified. Default: None.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> sun397_dataset_dir = "/path/to/sun397_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in sun397_dataset_dir with 8 threads
        >>> dataset = ds.SUN397Dataset(dataset_dir=sun397_dataset_dir, num_parallel_workers=8)

    About SUN397Dataset:

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of 397 categories with
    108,754 images. The number of images varies across categories, but there are at least 100 images per category.
    Images are in jpg, png, or gif format.

    Here is the original SUN397 dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── sun397_dataset_directory
            ├── ClassName.txt
            ├── README.txt
            ├── a
            │   ├── abbey
            │   │   ├── sun_aaaulhwrhqgejnyt.jpg
            │   │   ├── sun_aacphuqehdodwawg.jpg
            │   │   ├── ...
            │   ├── apartment_building
            │   │   └── outdoor
            │   │       ├── sun_aamyhslnsnomjzue.jpg
            │   │       ├── sun_abbjzfrsalhqivis.jpg
            │   │       ├── ...
            │   ├── ...
            ├── b
            │   ├── badlands
            │   │   ├── sun_aabtemlmesogqbbp.jpg
            │   │   ├── sun_afbsfeexggdhzshd.jpg
            │   │   ├── ...
            │   ├── balcony
            │   │   ├── exterior
            │   │   │   ├── sun_aaxzaiuznwquburq.jpg
            │   │   │   ├── sun_baajuldidvlcyzhv.jpg
            │   │   │   ├── ...
            │   │   └── interior
            │   │       ├── sun_babkzjntjfarengi.jpg
            │   │       ├── sun_bagjvjynskmonnbv.jpg
            │   │       ├── ...
            │   └── ...
            ├── ...


    Citation:

    .. code-block::

        @inproceedings{xiao2010sun,
        title        = {Sun database: Large-scale scene recognition from abbey to zoo},
        author       = {Xiao, Jianxiong and Hays, James and Ehinger, Krista A and Oliva, Aude and Torralba, Antonio},
        booktitle    = {2010 IEEE computer society conference on computer vision and pattern recognition},
        pages        = {3485--3492},
        year         = {2010},
        organization = {IEEE}
        }
    """

    @check_sun397_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.SUN397Node(self.dataset_dir, self.decode, self.sampler)


class _SVHNDataset:
    """
    Mainly for loading SVHN Dataset, and return two rows each time.
    """

    def __init__(self, dataset_dir, usage):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.usage = usage
        self.column_names = ["image", "label"]
        self.usage_all = ["train", "test", "extra"]
        self.data = np.array([], dtype=np.uint8)
        self.labels = np.array([], dtype=np.uint32)

        if self.usage == "all":
            for _usage in self.usage_all:
                data, label = self._load_mat(_usage)
                self.data = np.concatenate((self.data, data)) if self.data.size else data
                self.labels = np.concatenate((self.labels, label)) if self.labels.size else label
        else:
            self.data, self.labels = self._load_mat(self.usage)

    def _load_mat(self, mode):
        filename = mode + "_32x32.mat"
        mat_data = loadmat(os.path.join(self.dataset_dir, filename))
        data = np.transpose(mat_data['X'], [3, 0, 1, 2])
        label = mat_data['y'].astype(np.uint32).squeeze()
        np.place(label, label == 10, 0)
        return data, label

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class SVHNDataset(GeneratorDataset):
    """
    A source dataset that reads and parses SVHN dataset.

    The generated dataset has two columns: :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Specify the 'train', 'test', 'extra' or 'all' parts of dataset.
            Default: None, will read all samples.
        num_samples (int, optional): The number of samples to be included in the dataset. Default: None, all images.
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Sampler, optional): Object used to choose samples from the dataset. Random accessible
            input is required. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, 'num_samples' reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This argument must be specified only
            when num_shards is also specified.

    Raises:
        RuntimeError: If `dataset_dir` is not valid or does not exist or does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `usage` is invalid.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> svhn_dataset_dir = "/path/to/svhn_dataset_directory"
        >>> dataset = ds.SVHNDataset(dataset_dir=svhn_dataset_dir, usage="train")

    About SVHN dataset:

    SVHN dataset consists of 10 digit classes and is obtained from house numbers in Google Street View images.

    Here is the original SVHN dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── svhn_dataset_dir
             ├── train_32x32.mat
             ├── test_32x32.mat
             └── extra_32x32.mat

    Citation:

    .. code-block::

        @article{
          title={Reading Digits in Natural Images with Unsupervised Feature Learning},
          author={Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng},
          conference={NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.},
          year={2011},
          publisher={NIPS}
          url={http://ufldl.stanford.edu/housenumbers}
        }

    """

    @check_svhn_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=1, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.usage = replace_none(usage, "all")
        dataset = _SVHNDataset(self.dataset_dir, self.usage)

        super().__init__(dataset, column_names=dataset.column_names, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)


class USPSDataset(SourceDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses the USPS dataset.

    The generated dataset has two columns: :py:obj:`[image, label]` .
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test' or 'all'. 'train' will read from 7,291
            train samples, 'test' will read from 2,007 test samples, 'all' will read from all 9,298 samples.
            Default: None, will read all samples.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, will use value set in `mindspore.dataset.config` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Bool type and Shuffle enum are both supported to pass in. Default: `Shuffle.GLOBAL` .
            If shuffle is False, no shuffling will be performed.
            If shuffle is True, it is equivalent to setting `shuffle` to mindspore.dataset.Shuffle.GLOBAL.
            Set the mode of data shuffling by passing in enumeration variables:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` is not valid or does not exist or does not contain data files.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `usage` is invalid.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Examples:
        >>> usps_dataset_dir = "/path/to/usps_dataset_directory"
        >>>
        >>> # Read 3 samples from USPS dataset
        >>> dataset = ds.USPSDataset(dataset_dir=usps_dataset_dir, num_samples=3)

    About USPS dataset:

    USPS is a digit dataset automatically scanned from envelopes by the U.S. Postal Service
    containing a total of 9,298 16×16 pixel grayscale samples.
    The images are centered, normalized and show a broad range of font styles.

    Here is the original USPS dataset structure.
    You can download and unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── usps_dataset_dir
             ├── usps
             ├── usps.t

    Citation:

    .. code-block::

        @article{hull1994database,
          title={A database for handwritten text recognition research},
          author={Hull, Jonathan J.},
          journal={IEEE Transactions on pattern analysis and machine intelligence},
          volume={16},
          number={5},
          pages={550--554},
          year={1994},
          publisher={IEEE}
        }
    """

    @check_usps_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.USPSNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                            self.shard_id)


class VOCDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses VOC dataset.

    The generated dataset with different task setting has different output columns:

    - task = :py:obj:`Detection` , output columns: :py:obj:`[image, dtype=uint8]` , :py:obj:`[bbox, dtype=float32]` , \
        :py:obj:`[label, dtype=uint32]` , :py:obj:`[difficult, dtype=uint32]` , :py:obj:`[truncate, dtype=uint32]` .
    - task = :py:obj:`Segmentation` , output columns: :py:obj:`[image, dtype=uint8]` , :py:obj:`[target,dtype=uint8]` .

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        task (str, optional): Set the task type of reading voc data, now only support 'Segmentation' or 'Detection'.
            Default: 'Segmentation'.
        usage (str, optional): Set the task type of ImageSets. Default: 'train'. If task is 'Segmentation', image and
            annotation list will be loaded in ./ImageSets/Segmentation/usage + ".txt"; If task is 'Detection', image and
            annotation list will be loaded in ./ImageSets/Main/usage + ".txt"; if task and usage are not set, image and
            annotation list will be loaded in ./ImageSets/Segmentation/train.txt as default.
        class_indexing (dict, optional): A str-to-int mapping from label name to index, only valid in
            'Detection' task. Default: None, the folder names will be sorted alphabetically and each
            class will be given a unique index starting from 0.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, number set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None, expected
            order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: None. When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.
        extra_metadata(bool, optional): Flag to add extra meta-data to row. If True, an additional column named
            :py:obj:`[_meta-filename, dtype=string]` will be output at the end. Default: False.
        decrypt (callable, optional): Image decryption function, which accepts the path of the encrypted image file
            and returns the decrypted bytes data. Default: None, no decryption.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If xml of Annotations is an invalid format.
        RuntimeError: If xml of Annotations loss attribution of `object` .
        RuntimeError: If xml of Annotations loss attribution of `bndbox` .
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If task is not equal 'Segmentation' or 'Detection'.
        ValueError: If task equal 'Segmentation' but class_indexing is not None.
        ValueError: If txt related to mode is not exist.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - Column '[_meta-filename, dtype=string]' won't be output unless an explicit rename dataset op
          is added to remove the prefix('_meta-').
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> voc_dataset_dir = "/path/to/voc_dataset_directory"
        >>>
        >>> # 1) Read VOC data for segmentation training
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Segmentation", usage="train")
        >>>
        >>> # 2) Read VOC data for detection training
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train")
        >>>
        >>> # 3) Read all VOC dataset samples in voc_dataset_dir with 8 threads in random order
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train",
        ...                         num_parallel_workers=8)
        >>>
        >>> # 4) Read then decode all VOC dataset samples in voc_dataset_dir in sequence
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train",
        ...                         decode=True, shuffle=False)
        >>>
        >>> # In VOC dataset, if task='Segmentation', each dictionary has keys "image" and "target"
        >>> # In VOC dataset, if task='Detection', each dictionary has keys "image" and "annotation"

    About VOC dataset:

    The PASCAL Visual Object Classes (VOC) challenge is a benchmark in visual
    object category recognition and detection, providing the vision and machine
    learning communities with a standard dataset of images and annotation, and
    standard evaluation procedures.

    You can unzip the original VOC-2012 dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── voc2012_dataset_dir
            ├── Annotations
            │    ├── 2007_000027.xml
            │    ├── 2007_000032.xml
            │    ├── ...
            ├── ImageSets
            │    ├── Action
            │    ├── Layout
            │    ├── Main
            │    └── Segmentation
            ├── JPEGImages
            │    ├── 2007_000027.jpg
            │    ├── 2007_000032.jpg
            │    ├── ...
            ├── SegmentationClass
            │    ├── 2007_000032.png
            │    ├── 2007_000033.png
            │    ├── ...
            └── SegmentationObject
                 ├── 2007_000032.png
                 ├── 2007_000033.png
                 ├── ...

    Citation:

    .. code-block::

        @article{Everingham10,
        author       = {Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.},
        title        = {The Pascal Visual Object Classes (VOC) Challenge},
        journal      = {International Journal of Computer Vision},
        volume       = {88},
        year         = {2012},
        number       = {2},
        month        = {jun},
        pages        = {303--338},
        biburl       = {http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.html#bibtex},
        howpublished = {http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html}
        }
    """

    @check_vocdataset
    def __init__(self, dataset_dir, task="Segmentation", usage="train", class_indexing=None, num_samples=None,
                 num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None,
                 cache=None, extra_metadata=False, decrypt=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.task = replace_none(task, "Segmentation")
        self.usage = replace_none(usage, "train")
        self.class_indexing = replace_none(class_indexing, {})
        self.decode = replace_none(decode, False)
        self.extra_metadata = extra_metadata
        self.decrypt = decrypt

    def parse(self, children=None):
        return cde.VOCNode(self.dataset_dir, self.task, self.usage, self.class_indexing, self.decode, self.sampler,
                           self.extra_metadata, self.decrypt)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.

        Examples:
            >>> voc_dataset_dir = "/path/to/voc_dataset_directory"
            >>>
            >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection")
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.task != "Detection":
            raise NotImplementedError("Only 'Detection' support get_class_indexing.")
        if self.class_indexing is None or not self.class_indexing:
            if self._class_indexing is None:
                runtime_getter = self._init_tree_getters()
                self._class_indexing = runtime_getter[0].GetClassIndexing()
            self.class_indexing = {}
            for pair in self._class_indexing:
                self.class_indexing[pair[0]] = pair[1][0]
        return self.class_indexing


class WIDERFaceDataset(MappableDataset, VisionBaseDataset):
    """
    A source dataset that reads and parses WIDERFace dataset.

    When usage is "train", "valid" or "all", the generated dataset has eight columns ["image", "bbox", "blur",
    "expression", "illumination", "occlusion", "pose", "invalid"]. The data type of the `image` column is uint8,
    and all other columns are uint32. When usage is "test", it only has one column
    ["image"], with uint8 data type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be 'train', 'test', 'valid' or 'all'. 'train' will read
            from 12,880 samples, 'test' will read from 16,097 samples, 'valid' will read from 3,226 test samples
            and 'all' will read all 'train' and 'valid' samples. Default: None, will be set to 'all'.
        num_samples (int, optional): The number of images to be included in the dataset.
            Default: None, will read all images.
        num_parallel_workers (int, optional): Number of workers to read the data.
            Default: None, will use value set in `mindspore.dataset.config` .
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        decode (bool, optional): Decode the images after reading. Default: False.
        sampler (Sampler, optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This argument can only be specified
            when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing. More details:
            `Single-Node Data Cache <https://www.mindspore.cn/tutorials/experts/en/master/dataset/cache.html>`_ .
            Default: None, which means no cache is used.

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and `num_shards`/`shard_id` are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `usage` is not in ['train', 'test', 'valid', 'all'].
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `annotation_file` is not exist.
        ValueError: If `dataset_dir` is not exist.

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> wider_face_dir = "/path/to/wider_face_dataset"
        >>>
        >>> # Read 3 samples from WIDERFace dataset
        >>> dataset = ds.WIDERFaceDataset(dataset_dir=wider_face_dir, num_samples=3)

    About WIDERFace dataset:

    The WIDERFace database has a training set of 12,880 samples, a testing set of 16,097 examples
    and a validating set of 3,226 examples. It is a subset of a larger set available from WIDER. The digits have
    been size-normalized and centered in a fixed-size image.

    The following is the original WIDERFace dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── wider_face_dir
             ├── WIDER_test
             │    └── images
             │         ├── 0--Parade
             │         │     ├── 0_Parade_marchingband_1_9.jpg
             │         │     ├── ...
             │         ├──1--Handshaking
             │         ├──...
             ├── WIDER_train
             │    └── images
             │         ├── 0--Parade
             │         │     ├── 0_Parade_marchingband_1_11.jpg
             │         │     ├── ...
             │         ├──1--Handshaking
             │         ├──...
             ├── WIDER_val
             │    └── images
             │         ├── 0--Parade
             │         │     ├── 0_Parade_marchingband_1_102.jpg
             │         │     ├── ...
             │         ├──1--Handshaking
             │         ├──...
             └── wider_face_split
                  ├── wider_face_test_filelist.txt
                  ├── wider_face_train_bbx_gt.txt
                  └── wider_face_val_bbx_gt.txt

    Citation:

    .. code-block::

        @inproceedings{2016WIDER,
          title={WIDERFACE: A Detection Benchmark},
          author={Yang, S. and Luo, P. and Loy, C. C. and Tang, X.},
          booktitle={IEEE},
          pages={5525-5533},
          year={2016},
        }
    """

    @check_wider_face_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 decode=False, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.WIDERFaceNode(self.dataset_dir, self.usage, self.decode, self.sampler)
