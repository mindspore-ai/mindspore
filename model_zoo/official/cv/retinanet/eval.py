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

"""Evaluation for retinanet"""

import os
import time
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.retinanet import retinanet50, resnet50, retinanetInferWithDecoder
from src.dataset import create_retinanet_dataset, data_to_mindrecord_byte_image, voc_data_to_mindrecord
from src.box_utils import default_boxes
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep


def make_dataset_dir(mindrecord_dir, mindrecord_file, prefix):
    if config.dataset == "voc":
        config.coco_root = config.voc_root
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", False, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        elif config.dataset == "voc":
            if os.path.isdir(config.voc_dir) and os.path.isdir(config.voc_root):
                print("Create Mindrecord.")
                voc_data_to_mindrecord(mindrecord_dir, False, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("voc_root or voc_dir not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", False, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")
    print("Start Eval!")


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
def retinanet_eval():

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())
    prefix = "retinanet_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    make_dataset_dir(mindrecord_dir, mindrecord_file, prefix)

    batch_size = 1
    ds = create_retinanet_dataset(mindrecord_file, batch_size=batch_size, repeat_num=1, is_training=False)
    backbone = resnet50(config.num_classes)
    net = retinanet50(backbone, config)
    net = retinanetInferWithDecoder(net, Tensor(default_boxes), config)
    print("Load Checkpoint!")
    param_dict = load_checkpoint(config.checkpoint_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)

    net.set_train(False)
    i = batch_size
    total = ds.get_dataset_size() * batch_size
    start = time.time()
    predictions = []
    img_ids = []
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    num_classes = config.num_classes
    coco_root = config.coco_root
    data_type = config.val_data_type
    #Classes need to train or test.
    val_cls = config.coco_classes
    val_cls_dict = {}
    for i, cls in enumerate(val_cls):
        val_cls_dict[i] = cls
    anno_json = os.path.join(coco_root, config.instances_set.format(data_type))
    coco_gt = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco_gt.loadCats(coco_gt.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["name"]] = cat["id"]

    for data in ds.create_dict_iterator(output_numpy=True):
        pred_data = []
        img_id = data['img_id']
        img_np = data['image']
        image_shape = data['image_shape']

        output = net(Tensor(img_np))
        for batch_idx in range(img_np.shape[0]):
            pred_data.append({"boxes": output[0].asnumpy()[batch_idx],
                              "box_scores": output[1].asnumpy()[batch_idx],
                              "img_id": int(np.squeeze(img_id[batch_idx])),
                              "image_shape": image_shape[batch_idx]})
        i += batch_size
        for sample in pred_data:
            pred_boxes = sample['boxes']
            box_scores = sample['box_scores']
            img_id = sample['img_id']
            h, w = sample['image_shape']

            final_boxes = []
            final_label = []
            final_score = []
            img_ids.append(img_id)

            for c in range(1, num_classes):
                class_box_scores = box_scores[:, c]
                score_mask = class_box_scores > config.min_score
                class_box_scores = class_box_scores[score_mask]
                class_boxes = pred_boxes[score_mask] * [h, w, h, w]

                if score_mask.any():
                    nms_index = apply_nms(class_boxes, class_box_scores, config.nms_thershold, config.max_boxes)
                    class_boxes = class_boxes[nms_index]
                    class_box_scores = class_box_scores[nms_index]
                    final_boxes += class_boxes.tolist()
                    final_score += class_box_scores.tolist()
                    final_label += [classs_dict[val_cls_dict[c]]] * len(class_box_scores)
            for loc, label, score in zip(final_boxes, final_label, final_score):
                res = {}
                res['image_id'] = img_id
                res['bbox'] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]]
                res['score'] = score
                res['category_id'] = label
                predictions.append(res)
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

    cost_time = int((time.time() - start) * 1000)
    print(f'    100% [{total}/{total}] cost {cost_time} ms')
    coco_dt = coco_gt.loadRes('predictions.json')
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    mAP = E.stats[0]
    print("\n========================================\n")
    print(f"mAP: {mAP}")


if __name__ == '__main__':
    retinanet_eval()
