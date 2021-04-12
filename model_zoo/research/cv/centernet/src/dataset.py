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
"""
Data operations, will be used in train.py
"""

import os
import math
import argparse
import cv2
import numpy as np
import pycocotools.coco as coco

import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.mindrecord import FileWriter
from src.image import get_affine_transform, affine_transform
from src.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_dense_reg
from src.visual import visual_image

_current_dir = os.path.dirname(os.path.realpath(__file__))
cv2.setNumThreads(0)


class COCOHP(ds.Dataset):
    """
    Encapsulation class of COCO person keypoints datast.
    Initialize and preprocess of image for training and testing.

    Args:
        data_dir(str): Path of coco dataset.
        data_opt(edict): Config info for coco dataset.
        net_opt(edict): Config info for CenterNet.
        run_mode(str): Training or testing.

    Returns:
        Prepocessed training or testing dataset for CenterNet network.
    """

    def __init__(self, data_opt, run_mode="train", net_opt=None, enable_visual_image=False, save_path=None):
        super(COCOHP, self).__init__()
        self._data_rng = np.random.RandomState(123)
        self.data_opt = data_opt
        self.data_opt.mean = self.data_opt.mean.reshape(1, 1, 3)
        self.data_opt.std = self.data_opt.std.reshape(1, 1, 3)
        assert run_mode in ["train", "test", "val"], "only train/test/val mode are supported"
        self.run_mode = run_mode

        if net_opt is not None:
            self.net_opt = net_opt
        self.enable_visual_image = enable_visual_image
        if self.enable_visual_image:
            self.save_path = os.path.join(save_path, self.run_mode, "input_image")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

    def init(self, data_dir, keep_res=False):
        """initialize additional info"""
        logger.info('Initializing coco 2017 {} data.'.format(self.run_mode))
        if not os.path.isdir(data_dir):
            raise RuntimeError("Invalid dataset path")
        if self.run_mode != "test":
            self.annot_path = os.path.join(data_dir, 'annotations',
                                           'person_keypoints_{}2017.json').format(self.run_mode)
        else:
            self.annot_path = os.path.join(data_dir, 'annotations', 'image_info_test-dev2017.json')
        self.image_path = os.path.join(data_dir, '{}2017').format(self.run_mode)
        logger.info('Image path: {}'.format(self.image_path))
        logger.info('Annotations: {}'.format(self.annot_path))

        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()
        if self.run_mode != "test":
            self.images = []
            self.anns = {}
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if idxs:
                    self.images.append(img_id)
                    self.anns[img_id] = idxs
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        self.keep_res = keep_res
        if self.run_mode != "train":
            self.pad = 31
        logger.info('Loaded {} {} samples'.format(self.run_mode, self.num_samples))

    def __len__(self):
        return self.num_samples

    def transfer_coco_to_mindrecord(self, mindrecord_dir, file_name="coco_hp.train.mind", shard_num=1):
        """Create MindRecord file by image_dir and anno_path."""
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if os.path.isdir(self.image_path) and os.path.exists(self.annot_path):
            logger.info("Create MindRecord based on COCO_HP dataset")
        else:
            raise ValueError('data_dir {} or anno_path {} does not exist'.format(self.image_path, self.annot_path))

        mindrecord_path = os.path.join(mindrecord_dir, file_name)
        writer = FileWriter(mindrecord_path, shard_num)
        centernet_json = {
            "image": {"type": "bytes"},
            "num_objects": {"type": "int32"},
            "keypoints": {"type": "int32", "shape": [-1, self.data_opt.num_joints * 3]},
            "bbox": {"type": "float32", "shape": [-1, 4]},
            "category_id": {"type": "int32", "shape": [-1]},
        }
        writer.add_schema(centernet_json, "centernet_json")

        for img_id in self.images:
            image_info = self.coco.loadImgs([img_id])
            annos = self.coco.loadAnns(self.anns[img_id])
            # get image
            img_name = image_info[0]['file_name']
            img_name = os.path.join(self.image_path, img_name)
            with open(img_name, 'rb') as f:
                image = f.read()
            # parse annos info
            keypoints = []
            category_id = []
            bbox = []
            num_objects = len(annos)
            for anno in annos:
                keypoints.append(anno['keypoints'])
                category_id.append(anno['category_id'])
                bbox.append(anno['bbox'])

            row = {"image": image, "num_objects": num_objects,
                   "keypoints": np.array(keypoints, np.int32),
                   "bbox": np.array(bbox, np.float32),
                   "category_id": np.array(category_id, np.int32)}
            writer.write_raw_data([row])
        writer.commit()
        logger.info("Create Mindrecord Done, at {}".format(mindrecord_dir))

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.image_path, file_name)
        img = cv2.imread(img_path)
        image_id = np.array([img_id], dtype=np.int32).reshape((-1))
        ret = (img, image_id)
        return ret

    def pre_process_for_test(self, image, img_id, scale):
        """image pre-process for evaluation"""
        b, h, w, ch = image.shape
        assert b == 1, "only single image was supported here"
        image = image.reshape((h, w, ch))
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.keep_res:
            inp_height = (new_height | self.pad) + 1
            inp_width = (new_width | self.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
        else:
            inp_height, inp_width = self.data_opt.input_res[0], self.data_opt.input_res[1]
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height),
                                   flags=cv2.INTER_LINEAR)
        inp_img = (inp_image.astype(np.float32) / 255. - self.data_opt.mean) / self.data_opt.std

        eval_image = inp_img.reshape((1,) + inp_img.shape)
        eval_image = eval_image.transpose(0, 3, 1, 2)

        meta = {'c': c, 's': s,
                'out_height': inp_height // self.net_opt.down_ratio,
                'out_width': inp_width // self.net_opt.down_ratio}

        if self.enable_visual_image:
            if self.run_mode != "test":
                annos = self.coco.loadAnns(self.anns[img_id])
                num_objs = min(len(annos), self.data_opt.max_objs)
                num_joints = self.data_opt.num_joints
                ground_truth = []
                for k in range(num_objs):
                    ann = annos[k]
                    bbox = self._coco_box_to_bbox(ann['bbox']) * scale
                    cls_id = int(ann['category_id']) - 1
                    pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
                    bbox[:2] = affine_transform(bbox[:2], trans_input)
                    bbox[2:] = affine_transform(bbox[2:], trans_input)
                    bbox[0::2] = np.clip(bbox[0::2], 0, inp_width - 1)
                    bbox[1::2] = np.clip(bbox[1::2], 0, inp_height - 1)
                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if h <= 0 or w <= 0:
                        continue
                    for j in range(num_joints):
                        if pts[j, 2] > 0:
                            pts[j, :2] = affine_transform(pts[j, :2] * scale, trans_input)
                    bbox = [bbox[0], bbox[1], w, h]
                    gt = {
                        "image_id": int(img_id),
                        "category_id": int(cls_id + 1),
                        "bbox": bbox,
                        "score": float("{:.2f}".format(1)),
                        "keypoints": pts.reshape(num_joints * 3).tolist(),
                        "id": self.anns[img_id][k]
                    }
                    ground_truth.append(gt)
                visual_image(inp_image, ground_truth, self.save_path, height=inp_height, width=inp_width,
                             name="_scale" + str(scale))
            else:
                image_name = "gt_" + self.run_mode + "_image_" + str(img_id) + "_scale_" + str(scale) + ".png"
                cv2.imwrite("{}/{}".format(self.save_path, image_name), inp_image)

        return eval_image, meta

    def get_aug_param(self, img):
        """get data augmentation parameters"""
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        width = img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        flipped = False
        if self.data_opt.rand_crop:
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            h_border = self._get_border(self.data_opt.input_res[0], img.shape[0])
            w_border = self._get_border(self.data_opt.input_res[1], img.shape[1])
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        else:
            sf = self.data_opt.scale
            cf = self.data_opt.shift
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        if np.random.random() < self.data_opt.aug_rot:
            rf = self.data_opt.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

        if np.random.random() < self.data_opt.flip_prop:
            flipped = True
            img = img[:, ::-1, :]
            c[0] = width - c[0] - 1
        return img, width, c, s, rot, flipped

    def preprocess_fn(self, img, num_objects, keypoints, bboxes, category_id):
        """image pre-process and augmentation"""
        num_objs = min(num_objects, self.data_opt.max_objs)
        img, width, c, s, rot, flipped = self.get_aug_param(img)

        trans_input = get_affine_transform(c, s, rot, self.data_opt.input_res)
        inp = cv2.warpAffine(img, trans_input, (self.data_opt.input_res[0], self.data_opt.input_res[1]),
                             flags=cv2.INTER_LINEAR)

        assert self.data_opt.output_res[0] == self.data_opt.output_res[1]
        output_res = self.data_opt.output_res[0]
        num_joints = self.data_opt.num_joints
        max_objs = self.data_opt.max_objs
        num_classes = self.data_opt.num_classes

        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])

        hm = np.zeros((num_classes, output_res, output_res), dtype=np.float32)
        hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        dense_kps = np.zeros((num_joints, 2, output_res, output_res), dtype=np.float32)
        dense_kps_mask = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        wh = np.zeros((max_objs, 2), dtype=np.float32)
        kps = np.zeros((max_objs, num_joints * 2), dtype=np.float32)
        reg = np.zeros((max_objs, 2), dtype=np.float32)
        ind = np.zeros((max_objs), dtype=np.int32)
        reg_mask = np.zeros((max_objs), dtype=np.int32)
        kps_mask = np.zeros((max_objs, num_joints * 2), dtype=np.int32)
        hp_offset = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((max_objs * num_joints), dtype=np.int32)
        hp_mask = np.zeros((max_objs * num_joints), dtype=np.int32)

        draw_gaussian = draw_msra_gaussian if self.net_opt.mse_loss else draw_umich_gaussian
        ground_truth = []
        for k in range(num_objs):
            bbox = self._coco_box_to_bbox(bboxes[k])
            cls_id = int(category_id[k]) - 1
            pts = np.array(keypoints[k], np.float32).reshape(num_joints, 3)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1  # index begin from zero
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.data_opt.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

            lt, rb = [bbox[0], bbox[3]], [bbox[2], bbox[1]]
            bbox[:2] = affine_transform(bbox[:2], trans_output_rot)
            bbox[2:] = affine_transform(bbox[2:], trans_output_rot)
            if rot != 0:
                lt = affine_transform(lt, trans_output_rot)
                rb = affine_transform(rb, trans_output_rot)
                for i in range(2):
                    bbox[i] = min(lt[i], rb[i], bbox[i], bbox[i+2])
                    bbox[i+2] = max(lt[i], rb[i], bbox[i], bbox[i+2])
            bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h <= 0 or w <= 0:
                continue
            hp_radius = radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_res + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            num_kpts = pts[:, 2].sum()
            if num_kpts == 0:
                hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                reg_mask[k] = 0

            for j in range(num_joints):
                if pts[j, 2] > 0:
                    pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                    if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                            pts[j, 1] >= 0 and pts[j, 1] < output_res:
                        kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                        kps_mask[k, j * 2: j * 2 + 2] = 1
                        pt_int = pts[j, :2].astype(np.int32)
                        hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                        hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                        hp_mask[k * num_joints + j] = 1
                        if self.net_opt.dense_hp:
                            # must be before draw center hm gaussian
                            draw_dense_reg(dense_kps[j], hm[cls_id], ct_int, pts[j, :2] - ct_int,
                                           radius, is_offset=True)
                            draw_gaussian(dense_kps_mask[j], ct_int, radius)
                        draw_gaussian(hm_hp[j], pt_int, hp_radius)
            draw_gaussian(hm[cls_id], ct_int, radius)

            if self.enable_visual_image:
                gt = {
                    "category_id": int(cls_id + 1),
                    "bbox": [ct[0] - w / 2, ct[1] - h / 2, w, h],
                    "score": float("{:.2f}".format(1)),
                    "keypoints": pts.reshape(num_joints * 3).tolist(),
                }
                ground_truth.append(gt)
        ret = (inp, hm, reg_mask, ind, wh)
        if self.net_opt.dense_hp:
            dense_kps = dense_kps.reshape((num_joints * 2, output_res, output_res))
            dense_kps_mask = dense_kps_mask.reshape((num_joints, 1, output_res, output_res))
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
            dense_kps_mask = dense_kps_mask.reshape((num_joints * 2, output_res, output_res))
            ret += (dense_kps, dense_kps_mask)
        else:
            ret += (kps, kps_mask)

        ret += (reg, hm_hp, hp_offset, hp_ind, hp_mask)
        if self.enable_visual_image:
            out_img = cv2.warpAffine(img, trans_output_rot, (output_res, output_res), flags=cv2.INTER_LINEAR)
            visual_image(out_img, ground_truth, self.save_path, ratio=self.data_opt.input_res[0] // output_res)
        return ret

    def create_train_dataset(self, mindrecord_dir, prefix="coco_hp.train.mind", batch_size=1,
                             device_num=1, rank=0, num_parallel_workers=1, do_shuffle=True):
        """create train dataset based on mindrecord file"""
        if not os.path.isdir(mindrecord_dir):
            raise ValueError('MindRecord data_dir {} does not exist'.format(mindrecord_dir))

        files = os.listdir(mindrecord_dir)
        data_files = []
        for file_name in files:
            if prefix in file_name and "db" not in file_name:
                data_files.append(os.path.join(mindrecord_dir, file_name))
        if not data_files:
            raise ValueError('data_dir {} have no data files'.format(mindrecord_dir))

        columns = ["image", "num_objects", "keypoints", "bbox", "category_id"]
        data_set = ds.MindDataset(data_files,
                                  columns_list=columns,
                                  num_parallel_workers=num_parallel_workers, shuffle=do_shuffle,
                                  num_shards=device_num, shard_id=rank)
        ori_dataset_size = data_set.get_dataset_size()
        logger.info('origin dataset size: {}'.format(ori_dataset_size))

        data_set = data_set.map(operations=self.preprocess_fn,
                                input_columns=["image", "num_objects", "keypoints", "bbox", "category_id"],
                                output_columns=["image", "hm", "reg_mask", "ind", "wh", "kps", "kps_mask",
                                                "reg", "hm_hp", "hp_offset", "hp_ind", "hp_mask"],
                                column_order=["image", "hm", "reg_mask", "ind", "wh", "kps", "kps_mask",
                                              "reg", "hm_hp", "hp_offset", "hp_ind", "hp_mask"],
                                num_parallel_workers=num_parallel_workers,
                                python_multiprocessing=True)
        data_set = data_set.batch(batch_size, drop_remainder=True, num_parallel_workers=8)
        logger.info("data size: {}".format(data_set.get_dataset_size()))
        logger.info("repeat count: {}".format(data_set.get_repeat_count()))
        return data_set

    def create_eval_dataset(self, batch_size=1, num_parallel_workers=1):
        """create testing dataset based on coco format"""

        def generator():
            for i in range(self.num_samples):
                yield self.__getitem__(i)

        column = ["image", "image_id"]
        data_set = ds.GeneratorDataset(generator, column, num_parallel_workers=num_parallel_workers)
        data_set = data_set.batch(batch_size, drop_remainder=True, num_parallel_workers=8)
        return data_set


if __name__ == '__main__':
    # Convert coco2017 dataset to mindrecord to improve performance on host
    from src.config import dataset_config

    parser = argparse.ArgumentParser(description='CenterNet MindRecord dataset')
    parser.add_argument("--coco_data_dir", type=str, default="", help="Coco dataset directory.")
    parser.add_argument("--mindrecord_dir", type=str, default="", help="MindRecord dataset dir.")
    parser.add_argument("--mindrecord_prefix", type=str, default="coco_hp.train.mind",
                        help="Prefix of MindRecord dataset filename.")
    args_opt = parser.parse_args()
    dsc = COCOHP(dataset_config, run_mode="train")
    dsc.init(args_opt.coco_data_dir)
    dsc.transfer_coco_to_mindrecord(args_opt.mindrecord_dir, args_opt.mindrecord_prefix, shard_num=8)
