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
import argparse
import os
import numpy as np
import PIL.Image as Image

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True
                    , help='directory to store the image after being cropped')
parser.add_argument('--image_path', type=str, required=True,
                    help='directory of image to crop')
parser.add_argument('--image_height', type=int, default=768
                    , help='image height after being cropped')
parser.add_argument('--image_width', type=int, default=768
                    , help='image width after being cropped')
args = parser.parse_args()

valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                 23, 24, 25, 26, 27, 28, 31, 32, 33]

_key = np.array([-1, -1, -1, -1, -1, -1,
                 -1, -1, 0, 1, -1, -1,
                 2, 3, 4, -1, -1, -1,
                 5, -1, 6, 7, 8, 9,
                 10, 11, 12, 13, 14, 15,
                 -1, -1, 16, 17, 18])
_mapping = np.array(range(-1, len(_key) - 1)).astype('int32')

def _get_city_pairs(folder, split='train'):
    '''_get_city_pairs'''
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.startswith('._'):
                    continue
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit' + os.sep + split)
        mask_folder = os.path.join(folder, 'gtFine' + os.sep + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    assert split == 'trainval'
    print('trainval set')
    train_img_folder = os.path.join(folder, 'leftImg8bit' + os.sep + 'train')
    train_mask_folder = os.path.join(folder, 'gtFine' + os.sep + 'train')
    val_img_folder = os.path.join(folder, 'leftImg8bit' + os.sep + 'val')
    val_mask_folder = os.path.join(folder, 'gtFine' + os.sep + 'val')
    train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
    val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
    img_paths = train_img_paths + val_img_paths
    mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths

def _val_sync_transform(outsize, img, mask):
    '''_val_sync_transform'''
    short_size = min(outsize)
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize[1]) / 2.))
    y1 = int(round((h - outsize[0]) / 2.))
    img = img.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))
    mask = mask.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))

    # final transform
    img, mask = np.array(img), _mask_transform(mask)
    return img, mask

def _class_to_index(mask):
    # assert the value
    values = np.unique(mask)
    for value in values:
        assert value in _mapping
    index = np.digitize(mask.ravel(), _mapping, right=True)
    return _key[index].reshape(mask.shape)

def _mask_transform(mask):
    target = _class_to_index(np.array(mask).astype('int32'))
    return np.array(target).astype('int32')

def crop_imageAndLabel(out_dir, image_path, image_height, image_width):
    if not os.path.exists(os.path.join(out_dir, "images")):
        os.makedirs(os.path.join(out_dir, "images"))
    if not os.path.exists(os.path.join(out_dir, "labels")):
        os.makedirs(os.path.join(out_dir, "labels"))

    assert os.path.exists(image_path), "Please put dataset in {SEG_ROOT}/datasets/cityscapes"
    images, mask_paths = _get_city_pairs(image_path, 'val')
    assert len(images) == len(mask_paths)
    if not images:
        raise RuntimeError("Found 0 images in subfolders of:" + image_path + "\n")

    for index in range(len(images)):
        print("Processing ", images[index])
        img = Image.open(images[index]).convert('RGB')
        mask = Image.open(mask_paths[index])
        img, mask = _val_sync_transform((image_height, image_width), img, mask)

        img = img.astype(np.float32)
        mask = mask.astype(np.int32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = img.transpose((2, 0, 1))#HWC->CHW
        for channel, _ in enumerate(img):
            # Normalization
            img[channel] /= 255
            img[channel] -= mean[channel]
            img[channel] /= std[channel]

        img = np.expand_dims(img, 0)#NCHW
        mask = np.expand_dims(mask, 0)#NHW
        filename = images[index].split(os.sep)[-1].split('.')[0]    # get the name of image file
        img.tofile(os.path.join(os.path.join(out_dir, "images"), filename+'_img.bin'))
        mask.tofile(os.path.join(os.path.join(out_dir, "labels"), filename+'_label.bin'))

if __name__ == "__main__":
    crop_imageAndLabel(args.out_dir, args.image_path, args.image_height, args.image_width)
