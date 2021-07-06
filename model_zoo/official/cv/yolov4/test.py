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
"""YoloV4 test-dev."""

import os
import sys
import time
import datetime
from collections import defaultdict
import json
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.yolo import YOLOV4CspDarkNet53
from src.logger import get_logger
from src.yolo_dataset import create_yolo_datasetv2

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Davinci", save_graphs=False, device_id=devid)

config.data_root = os.path.join(config.data_dir, 'test2017')
config.nms_thresh = config.test_nms_thresh

class DetectionEngine():
    """Detection engine"""
    def __init__(self, args_engine):
        self.test_ignore_threshold = args_engine.test_ignore_threshold
        self.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                       'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                       'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                       'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                       'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.num_classes = len(self.labels)
        self.results = {}  # img_id->class
        self.file_path = ''  # path to save predict result
        self.save_prefix = args_engine.outputs_dir
        self.det_boxes = []
        self.nms_thresh = args_engine.nms_thresh
        self.coco_catids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
                            28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
                            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                            81, 82, 84, 85, 86, 87, 88, 89, 90]

    def do_nms_for_results(self):
        """nms result"""
        for img_id in self.results:
            for clsi in self.results[img_id]:
                dets = self.results[img_id][clsi]
                dets = np.array(dets)
                keep_index = self._diou_nms(dets, thresh=0.6)

                keep_box = [{'image_id': int(img_id),
                             'category_id': int(clsi),
                             'bbox': list(dets[i][:4].astype(float)),
                             'score': dets[i][4].astype(float)}
                            for i in keep_index]
                self.det_boxes.extend(keep_box)

    def _nms(self, dets, thresh):
        """nms function"""
        # convert xywh -> xmin ymin xmax ymax
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


    def _diou_nms(self, dets, thresh=0.5):
        """convert xywh -> xmin ymin xmax ymax"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            center_x1 = (x1[i] + x2[i]) / 2
            center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
            center_y1 = (y1[i] + y2[i]) / 2
            center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
            inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
            out_max_x = np.maximum(x2[i], x2[order[1:]])
            out_max_y = np.maximum(y2[i], y2[order[1:]])
            out_min_x = np.minimum(x1[i], x1[order[1:]])
            out_min_y = np.minimum(y1[i], y1[order[1:]])
            outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
            diou = ovr - inter_diag / outer_diag
            diou = np.clip(diou, -1, 1)
            inds = np.where(diou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def write_result(self):
        """write result to json file"""
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.det_boxes, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def detect(self, outputs, batch, image_shape, image_id):
        """post process"""
        outputs_num = len(outputs)
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_id in range(outputs_num):
                # 32, 52, 52, 3, 85
                out_item = outputs[out_id]
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                # get number of items in one head, [B, gx, gy, anchors, 5+80]
                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                ori_w, ori_h = image_shape[batch_id]
                img_id = int(image_id[batch_id])
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]

                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x = x.reshape(-1)
                y = y.reshape(-1)
                w = w.reshape(-1)
                h = h.reshape(-1)
                cls_emb = cls_emb.reshape(-1, 80)
                conf = conf.reshape(-1)
                cls_argmax = cls_argmax.reshape(-1)

                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                # create all False
                flag = np.random.random(cls_emb.shape) > sys.maxsize
                for i in range(flag.shape[0]):
                    c = cls_argmax[i]
                    flag[i, c] = True
                confidence = cls_emb[flag] * conf
                for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence, cls_argmax):
                    if confi < self.test_ignore_threshold:
                        continue
                    if img_id not in self.results:
                        self.results[img_id] = defaultdict(list)
                    x_lefti = max(0, x_lefti)
                    y_lefti = max(0, y_lefti)
                    wi = min(wi, ori_w)
                    hi = min(hi, ori_h)
                    # transform catId to match coco
                    coco_clsi = self.coco_catids[clsi]
                    self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_test():
    """test method"""

    # init distributed
    if config.is_distributed:
        init()
        config.rank = get_rank()
        config.group_size = get_group_size()

    # logger
    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    config.logger = get_logger(config.outputs_dir, config.rank)

    context.reset_auto_parallel_context()
    if config.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
    else:
        parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    config.logger.info('Creating Network....')
    network = YOLOV4CspDarkNet53()

    config.logger.info(config.pretrained)
    if os.path.isfile(config.pretrained):
        param_dict = load_checkpoint(config.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        config.logger.info('load_model %s success', config.pretrained)
    else:
        config.logger.info('%s not exists or not a pre-trained file', config.pretrained)
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(config.pretrained))
        exit(1)

    data_root = config.data_root

    data_txt = os.path.join(config.data_dir, 'testdev2017.txt')
    ds, data_size = create_yolo_datasetv2(data_root, data_txt=data_txt, batch_size=config.per_batch_size,
                                          max_epoch=1, device_num=config.group_size, rank=config.rank, shuffle=False,
                                          default_config=config)

    config.logger.info('testing shape : %s', config.test_img_shape)
    config.logger.info('totol %d images to eval', data_size)

    network.set_train(False)

    # init detection engine
    detection = DetectionEngine(config)

    config.logger.info('Start inference....')
    for i, data in enumerate(ds.create_dict_iterator()):
        image = Tensor(data["image"])

        image_shape = Tensor(data["image_shape"])
        image_id = Tensor(data["img_id"])

        prediction = network(image)
        output_big, output_me, output_small = prediction
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        image_id = image_id.asnumpy()
        image_shape = image_shape.asnumpy()

        detection.detect([output_small, output_me, output_big], config.per_batch_size, image_shape, image_id)
        if i % 1000 == 0:
            config.logger.info('Processing... {:.2f}% '.format(i * config.per_batch_size / data_size * 100))

    config.logger.info('Calculating mAP...')
    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    config.logger.info('result file path: %s', result_file_path)

if __name__ == "__main__":
    run_test()
