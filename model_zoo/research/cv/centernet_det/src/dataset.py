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
Data operations, will be used in train.py
"""

import os
import sys
import math
import cv2
import numpy as np
import pycocotools.coco as coco
import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.mindrecord import FileWriter

try:
    from src.model_utils.config import config, dataset_config
    from src.model_utils.moxing_adapter import moxing_wrapper
    from src.image import color_aug, get_affine_transform, affine_transform
    from src.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_dense_reg
    from src.visual import visual_image
except ImportError as import_error:
    print('Import Error: {}, trying append path/centernet_det/src/../'.format(import_error))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from src.model_utils.config import config, dataset_config
    from src.model_utils.moxing_adapter import moxing_wrapper
    from src.image import color_aug, get_affine_transform, affine_transform
    from src.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_dense_reg
    from src.visual import visual_image


_current_dir = os.path.dirname(os.path.realpath(__file__))
cv2.setNumThreads(0)


class COCOHP(ds.Dataset):
    """
    Encapsulation class of COCO datast.
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
        self._data_rng = np.random.RandomState(123)
        self.data_opt = data_opt
        self.data_opt.mean = self.data_opt.mean.reshape(1, 1, 3)
        self.data_opt.std = self.data_opt.std.reshape(1, 1, 3)
        self.pad = 127
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
                                           'instances_{}2017.json').format(self.run_mode)
        else:
            self.annot_path = os.path.join(data_dir, 'annotations', 'image_info_test-dev2017.json')
        self.image_path = os.path.join(data_dir, '{}2017').format(self.run_mode)
        logger.info('Image path: {}'.format(self.image_path))
        logger.info('Annotations: {}'.format(self.annot_path))

        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        self.train_cls = self.data_opt.coco_classes
        self.train_cls_dict = {}
        for i, cls in enumerate(self.train_cls):
            self.train_cls_dict[cls] = i

        self.classs_dict = {}
        cat_ids = self.coco.loadCats(self.coco.getCatIds())
        for cat in cat_ids:
            self.classs_dict[cat["id"]] = cat["name"]

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
        logger.info('Loaded {} {} samples'.format(self.run_mode, self.num_samples))

    def __len__(self):
        return self.num_samples

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def transfer_coco_to_mindrecord(self, mindrecord_dir, file_name="coco_det.train.mind", shard_num=1):
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
            "img_id": {"type": "int32", "shape": [1]},
            "image": {"type": "bytes"},
            "num_objects": {"type": "int32"},
            "bboxes": {"type": "float32", "shape": [-1, 4]},
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

            bboxes = []
            category_id = []
            num_objects = len(annos)
            for anno in annos:
                bbox = self._coco_box_to_bbox(anno['bbox'])
                class_name = self.classs_dict[anno["category_id"]]
                if class_name in self.train_cls:
                    x_min, x_max = bbox[0], bbox[2]
                    y_min, y_max = bbox[1], bbox[3]
                    bboxes.append([x_min, y_min, x_max, y_max])
                    category_id.append(self.train_cls_dict[class_name])

            row = {"img_id": np.array([img_id], dtype=np.int32),
                   "image": image,
                   "num_objects": num_objects,
                   "bboxes": np.array(bboxes, np.float32),
                   "category_id": np.array(category_id, np.int32)}
            writer.write_raw_data([row])

        writer.commit()
        logger.info("Create Mindrecord Done, at {}".format(mindrecord_dir))

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
                ground_truth = []
                for k in range(num_objs):
                    ann = annos[k]
                    bbox = self._coco_box_to_bbox(ann['bbox']) * scale
                    cls_id = int(ann['category_id']) - 1
                    bbox[:2] = affine_transform(bbox[:2], trans_input)
                    bbox[2:] = affine_transform(bbox[2:], trans_input)
                    bbox[0::2] = np.clip(bbox[0::2], 0, inp_width - 1)
                    bbox[1::3] = np.clip(bbox[1::3], 0, inp_height - 1)
                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if h <= 0 or w <= 0:
                        continue
                    bbox = [bbox[0], bbox[1], w, h]
                    gt = {
                        "image_id": int(img_id),
                        "category_id": int(cls_id + 1),
                        "bbox": bbox,
                        "score": float("{:.2f}".format(1)),
                        "id": self.anns[img_id][k]
                    }
                    ground_truth.append(gt)
                visual_image(inp_image, ground_truth, self.save_path, height=inp_height, width=inp_width,
                             name="_scale" + str(scale))
            else:
                image_name = "gt_" + self.run_mode + "_image_" + str(img_id) + "_scale_" + str(scale) + ".png"
                cv2.imwrite("{}/{}".format(self.save_path, image_name), inp_image)

        return eval_image, meta

    def preprocess_fn(self, image, num_objects, bboxes, category_id):
        """image pre-process and augmentation"""
        num_objs = min(num_objects, self.data_opt.max_objs)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        height = img.shape[0]
        width = img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        input_h, input_w = self.data_opt.input_res[0], self.data_opt.input_res[1]
        rot = 0

        flipped = False
        if self.data_opt.rand_crop:
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            h_border = self._get_border(128, img.shape[0])
            w_border = self._get_border(128, img.shape[1])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        else:
            sf = self.data_opt.scale
            cf = self.data_opt.shift
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.data_opt.flip_prop:
            flipped = True
            img = img[:, ::-1, :]
            c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, rot, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.run_mode == "train" and self.data_opt.color_aug:
            color_aug(self._data_rng, inp, self.data_opt.eig_val, self.data_opt.eig_vec)

        if self.data_opt.output_res[0] != self.data_opt.output_res[1]:
            raise ValueError("Only square image was supported to used as output for convenient")

        output_h = input_h // self.data_opt.down_ratio
        output_w = input_w // self.data_opt.down_ratio
        max_objs = self.data_opt.max_objs
        num_classes = self.data_opt.num_classes
        trans_output_rot = get_affine_transform(c, s, rot, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((max_objs, 2), dtype=np.float32)
        ind = np.zeros((max_objs), dtype=np.int32)
        reg_mask = np.zeros((max_objs), dtype=np.int32)
        cat_spec_wh = np.zeros((max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((max_objs, num_classes * 2), dtype=np.int32)

        draw_gaussian = draw_msra_gaussian if self.net_opt.mse_loss else draw_umich_gaussian

        ground_truth = []
        for k in range(num_objs):
            bbox = bboxes[k]
            cls_id = category_id[k] - 1
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1  # index begin from zero
            bbox[:2] = affine_transform(bbox[:2], trans_output_rot)
            bbox[2:] = affine_transform(bbox[2:], trans_output_rot)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h <= 0 and w <= 0:
                continue
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm[cls_id], ct_int, radius)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
            cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1

            if self.net_opt.dense_wh:
                draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
            ground_truth.append([ct[0] - w / 2, ct[1] - h / 2,
                                 ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        ret = (inp, hm, reg_mask, ind, wh)
        if self.net_opt.dense_wh:
            hm_a = hm.max(axis=0)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret += (dense_wh, dense_wh_mask)
        elif self.net_opt.cat_spec_wh:
            ret += (cat_spec_wh, cat_spec_mask)
        if self.net_opt.reg_offset:
            ret += (reg,)
        return ret

    def create_train_dataset(self, mindrecord_dir, prefix="coco_det.train.mind", batch_size=1,
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

        columns = ["img_id", "image", "num_objects", "bboxes", "category_id"]
        data_set = ds.MindDataset(data_files,
                                  columns_list=columns,
                                  num_parallel_workers=num_parallel_workers, shuffle=do_shuffle,
                                  num_shards=device_num, shard_id=rank)
        ori_dataset_size = data_set.get_dataset_size()
        logger.info('origin dataset size: {}'.format(ori_dataset_size))

        data_set = data_set.map(operations=self.preprocess_fn,
                                input_columns=["image", "num_objects", "bboxes", "category_id"],
                                output_columns=["image", "hm", "reg_mask", "ind", "wh", "reg"],
                                column_order=["image", "hm", "reg_mask", "ind", "wh", "reg"],
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


def modelarts_pre_process():
    """modelarts pre process function."""
    config.coco_data_dir = config.data_path
    config.mindrecord_dir = config.output_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def coco2mindrecord():
    """Convert coco2017 dataset to mindrecord"""
    dsc = COCOHP(dataset_config, run_mode="train")
    dsc.init(config.coco_data_dir)
    dsc.transfer_coco_to_mindrecord(config.mindrecord_dir, config.mindrecord_prefix, shard_num=8)


if __name__ == '__main__':
    coco2mindrecord()
