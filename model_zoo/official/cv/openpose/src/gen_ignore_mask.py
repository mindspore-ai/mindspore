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
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO as ReadJson
from model_utils.config import config


class DataLoader():
    def __init__(self, train_, dir_name, mode_='train'):
        self.train = train_
        self.dir_name = dir_name
        assert mode_ in ['train', 'val'], 'Data loading mode is invalid.'
        self.mode = mode_
        self.catIds = train_.getCatIds()  # catNms=['person']
        self.imgIds = sorted(train_.getImgIds(catIds=self.catIds))

    def __len__(self):
        return len(self.imgIds)

    def gen_masks(self, image_, annotations_):
        mask_all_1 = np.zeros(image_.shape[:2], 'bool')
        mask_miss_1 = np.zeros(image_.shape[:2], 'bool')
        for ann in annotations_:
            mask = self.train.annToMask(ann).astype('bool')
            if ann['iscrowd'] == 1:
                intxn = mask_all_1 & mask
                mask_miss_1 = np.bitwise_or(mask_miss_1.astype(int), np.subtract(mask, intxn, dtype=np.int32))
                mask_all_1 = np.bitwise_or(mask_all_1.astype(int), mask.astype(int))
            elif ann['num_keypoints'] < config.min_keypoints or ann['area'] <= config.min_area:
                mask_all_1 = np.bitwise_or(mask_all_1.astype(int), mask.astype(int))
                mask_miss_1 = np.bitwise_or(mask_miss_1.astype(int), mask.astype(int))
            else:
                mask_all_1 = np.bitwise_or(mask_all_1.astype(int), mask.astype(int))
        return mask_all_1, mask_miss_1

    def dwaw_gen_masks(self, image_, mask, color=(0, 0, 1)):
        bimsk = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = image_ * bimsk.astype(np.int32)
        clmsk = np.ones(bimsk.shape) * bimsk
        for index_1 in range(3):
            clmsk[:, :, index_1] = clmsk[:, :, index_1] * color[index_1] * 255
        image_ = image_ + 0.7 * clmsk - 0.7 * mskd
        return image_.astype(np.uint8)

    def draw_masks_and_keypoints(self, image_, annotations_):
        for ann in annotations_:
            # masks
            mask = self.train.annToMask(ann).astype(np.uint8)
            if ann['iscrowd'] == 1:
                color = (0, 0, 1)
            elif ann['num_keypoints'] == 0:
                color = (0, 1, 0)
            else:
                color = (1, 0, 0)
            bimsk = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mskd = image_ * bimsk.astype(np.int32)
            clmsk = np.ones(bimsk.shape) * bimsk
            for index_1 in range(3):
                clmsk[:, :, index_1] = clmsk[:, :, index_1] * color[index_1] * 255
            image_ = image_ + 0.7 * clmsk - 0.7 * mskd

            # keypoints
            for x, y, v in np.array(ann['keypoints']).reshape(-1, 3):
                if v == 1:
                    cv2.circle(image_, (x, y), 3, (255, 255, 0), -1)
                elif v == 2:
                    cv2.circle(image_, (x, y), 3, (255, 0, 255), -1)
        return image_.astype(np.uint8)

    def get_img_annotation(self, ind=None, img_id_=None):
        if ind is not None:
            img_id_ = self.imgIds[ind]

        anno_ids = self.train.getAnnIds(imgIds=[img_id_])
        annotations_ = self.train.loadAnns(anno_ids)

        img_file = os.path.join(self.dir_name, self.train.loadImgs([img_id_])[0]['file_name'])
        image_ = cv2.imread(img_file)
        return image_, annotations_, img_id_


if __name__ == '__main__':
    path_list = [config.train_ann, config.val_ann, config.train_dir, config.val_dir]
    for index, mode in enumerate(['train', 'val']):
        train = ReadJson(path_list[index])
        data_loader = DataLoader(train, path_list[index+2], mode_=mode)

        save_dir = os.path.join(os.path.dirname(path_list[index+2]), 'ignore_mask_{}'.format(mode))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in tqdm(range(len(data_loader))):
            img, annotations, img_id = data_loader.get_img_annotation(ind=i)
            mask_all, mask_miss = data_loader.gen_masks(img, annotations)

            if config.vis:
                ann_img = data_loader.draw_masks_and_keypoints(img, annotations)
                msk_img = data_loader.dwaw_gen_masks(img, mask_miss)
                cv2.imshow('image', np.hstack((ann_img, msk_img)))
                k = cv2.waitKey()
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    cv2.imwrite('aaa.png', np.hstack((ann_img, msk_img)))

            if np.any(mask_miss) and not config.vis:
                mask_miss = mask_miss.astype(np.uint8) * 255
                save_path = os.path.join(save_dir, '{:012d}.png'.format(img_id))
                cv2.imwrite(save_path, mask_miss)
