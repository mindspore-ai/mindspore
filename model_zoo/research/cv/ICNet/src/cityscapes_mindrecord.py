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
"""Prepare Cityscapes dataset"""
import os
import random
import argparse
import numpy as np
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import mindspore.dataset as de
from mindspore.mindrecord import FileWriter
import mindspore.dataset.vision.py_transforms as transforms
import mindspore.dataset.transforms.py_transforms as tc


def _get_city_pairs(folder, split='train'):
    """Return two path arrays of data set img and mask"""

    def get_path_pairs(image_folder, masks_folder):
        image_paths = []
        masks_paths = []
        for root, _, files in os.walk(image_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(masks_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        image_paths.append(imgpath)
                        masks_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(image_paths), image_folder))
        return image_paths, masks_paths

    if split in ('train', 'val'):
        # "./Cityscapes/leftImg8bit/train" or "./Cityscapes/leftImg8bit/val"
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        # "./Cityscapes/gtFine/train" or "./Cityscapes/gtFine/val"
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        # The order of img_paths and mask_paths is one-to-one correspondence
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths


def _sync_transform(img, mask):
    """img and mask augmentation"""
    a = random.Random()
    a.seed(1234)
    base_size = 1024
    crop_size = 960

    # random mirror
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    crop_size = crop_size
    # random scale (short edge)
    short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
    w, h = img.size
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
    # random crop crop_size
    w, h = img.size
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    # gaussian blur as in PSP
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
    # final transform
    output = _img_mask_transform(img, mask)

    return output


def _class_to_index(mask):
    """class to index"""
    # reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    _key = np.array([-1, -1, -1, -1, -1, -1,
                     -1, -1, 0, 1, -1, -1,
                     2, 3, 4, -1, -1, -1,
                     5, -1, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15,
                     -1, -1, 16, 17, 18])
    # [-1, ..., 33]
    _mapping = np.array(range(-1, len(_key) - 1)).astype('int32')

    # assert the value
    values = np.unique(mask)
    for value in values:
        assert value in _mapping
    # Get the index of each pixel value in the mask corresponding to _mapping
    index = np.digitize(mask.ravel(), _mapping, right=True)
    # According to the above index, according to _key, get the corresponding
    return _key[index].reshape(mask.shape)


def _img_transform(img):
    return np.array(img)


def _mask_transform(mask):
    target = _class_to_index(np.array(mask).astype('int32'))
    return np.array(target).astype('int32')


def _img_mask_transform(img, mask):
    """img and mask transform"""
    input_transform = tc.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = _img_transform(img)
    mask = _mask_transform(mask)
    img = input_transform(img)

    img = np.array(img).astype(np.float32)
    mask = np.array(mask).astype(np.float32)

    return (img, mask)


def data_to_mindrecord_img(prefix='cityscapes-2975.mindrecord', file_num=1,
                           root='./', split='train', mindrecord_dir="./"):
    """to mindrecord"""
    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writter = FileWriter(mindrecord_path, file_num)

    img_paths, mask_paths = _get_city_pairs(root, split)

    cityscapes_json = {
        "images": {"type": "int32", "shape": [1024, 2048, 3]},
        "mask": {"type": "int32", "shape": [1024, 2048]},
    }

    writter.add_schema(cityscapes_json, "cityscapes_json")

    images_files_num = len(img_paths)
    for index in range(images_files_num):
        img = Image.open(img_paths[index]).convert('RGB')
        img = np.array(img, dtype=np.int32)

        mask = Image.open(mask_paths[index])
        mask = np.array(mask, dtype=np.int32)

        row = {"images": img, "mask": mask}
        if (index + 1) % 10 == 0:
            print("writing {}/{} into mindrecord".format(index + 1, images_files_num))
        writter.write_raw_data([row])
    writter.commit()


def get_Image_crop_nor(img, mask):
    image = np.uint8(img)
    mask = np.uint8(mask)
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    output = _sync_transform(image, mask)

    return output


def create_icnet_dataset(mindrecord_file, batch_size=16, device_num=1, rank_id=0):
    """create dataset for training"""
    a = random.Random()
    a.seed(1234)
    ds = de.MindDataset(mindrecord_file, columns_list=["images", "mask"],
                        num_shards=device_num, shard_id=rank_id, shuffle=True)
    ds = ds.map(operations=get_Image_crop_nor, input_columns=["images", "mask"], output_columns=["image", "masks"])

    ds = ds.batch(batch_size=batch_size, drop_remainder=False)

    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dataset_to_mindrecord")
    parser.add_argument("--dataset_path", type=str, default="/data/cityscapes/", help="dataset path")
    parser.add_argument("--mindrecord_path", type=str, default="/data/cityscapes_mindrecord/",
                        help="mindrecord_path")

    args_opt = parser.parse_args()
    data_to_mindrecord_img(root=args_opt.dataset_path, mindrecord_dir=args_opt.mindrecord_path)
