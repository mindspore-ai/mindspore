# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import argparse
import cv2

import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO as ReadJson

from config import params

class DataLoader():
    def __init__(self, coco, dir_name, data_mode='train'):
        self.train = coco
        self.dir_name = dir_name
        assert data_mode in ['train', 'val'], 'Data loading mode is invalid.'
        self.mode = data_mode
        self.catIds = coco.getCatIds()  # catNms=['person']
        self.imgIds = sorted(coco.getImgIds(catIds=self.catIds))

    def __len__(self):
        return len(self.imgIds)

    def gen_masks(self, image, anns):
        _mask_all = np.zeros(image.shape[:2], 'bool')
        _mask_miss = np.zeros(image.shape[:2], 'bool')
        for ann in anns:
            mask = self.train.annToMask(ann).astype('bool')
            if ann['iscrowd'] == 1:
                intxn = _mask_all & mask
                _mask_miss = np.bitwise_or(_mask_miss.astype(int), np.subtract(mask, intxn, dtype=np.int32))
                _mask_all = np.bitwise_or(_mask_all.astype(int), mask.astype(int))
            elif ann['num_keypoints'] < params['min_keypoints'] or ann['area'] <= params['min_area']:
                _mask_all = np.bitwise_or(_mask_all.astype(int), mask.astype(int))
                _mask_miss = np.bitwise_or(_mask_miss.astype(int), mask.astype(int))
            else:
                _mask_all = np.bitwise_or(_mask_all.astype(int), mask.astype(int))
        return _mask_all, _mask_miss

    def dwaw_gen_masks(self, image, mask, color=(0, 0, 1)):
        bimsk = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = image * bimsk.astype(np.int32)
        clmsk = np.ones(bimsk.shape) * bimsk
        for ind in range(3):
            clmsk[:, :, ind] = clmsk[:, :, ind] * color[ind] * 255
        image = image + 0.7 * clmsk - 0.7 * mskd
        return image.astype(np.uint8)

    def draw_masks_and_keypoints(self, image, anns):
        for ann in anns:
            # masks
            mask = self.train.annToMask(ann).astype(np.uint8)
            if ann['iscrowd'] == 1:
                color = (0, 0, 1)
            elif ann['num_keypoints'] == 0:
                color = (0, 1, 0)
            else:
                color = (1, 0, 0)
            bimsk = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mskd = image * bimsk.astype(np.int32)
            clmsk = np.ones(bimsk.shape) * bimsk
            for ind in range(3):
                clmsk[:, :, ind] = clmsk[:, :, ind] * color[ind] * 255
            image = image + 0.7 * clmsk - 0.7 * mskd

            # keypoints
            for x, y, v in np.array(ann['keypoints']).reshape(-1, 3):
                if v == 1:
                    cv2.circle(image, (x, y), 3, (255, 255, 0), -1)
                elif v == 2:
                    cv2.circle(image, (x, y), 3, (255, 0, 255), -1)
        return image.astype(np.uint8)

    def get_img_annotation(self, ind=None, image_id=None):
        if ind is not None:
            image_id = self.imgIds[ind]

        anno_ids = self.train.getAnnIds(imgIds=[image_id])
        _annotations = self.train.loadAnns(anno_ids)

        img_file = os.path.join(params['data_dir'], self.dir_name, self.train.loadImgs([image_id])[0]['file_name'])
        _image = cv2.imread(img_file)
        return _image, _annotations, image_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true', help='visualize annotations and ignore masks')
    parser.add_argument('--train_ann', type=str, help='train annotations json')
    parser.add_argument('--val_ann', type=str, help='val annotations json')
    parser.add_argument('--train_dir', type=str, help='name of train dir')
    parser.add_argument('--val_dir', type=str, help='name of val dir')
    args = parser.parse_args()
    path_list = [args.train_ann, args.val_ann, args.train_dir, args.val_dir]
    for index, mode in enumerate(['train', 'val']):
        train = ReadJson(path_list[index])
        data_loader = DataLoader(train, path_list[index+2], mode=mode)

        save_dir = os.path.join(params['data_dir'], 'ignore_mask_{}'.format(mode))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in tqdm(range(len(data_loader))):
            img, annotations, img_id = data_loader.get_img_annotation(ind=i)
            mask_all, mask_miss = data_loader.gen_masks(img, annotations)

            if args.vis:
                ann_img = data_loader.draw_masks_and_keypoints(img, annotations)
                msk_img = data_loader.dwaw_gen_masks(img, mask_miss)
                cv2.imshow('image', np.hstack((ann_img, msk_img)))
                k = cv2.waitKey()
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    cv2.imwrite('aaa.png', np.hstack((ann_img, msk_img)))

            if np.any(mask_miss) and not args.vis:
                mask_miss = mask_miss.astype(np.uint8) * 255
                save_path = os.path.join(save_dir, '{:012d}.png'.format(img_id))
                cv2.imwrite(save_path, mask_miss)
