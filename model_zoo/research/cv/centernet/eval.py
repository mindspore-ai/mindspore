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
CenterNet evaluation script.
"""

import os
import time
import copy
import json
import argparse
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.log as logger
from src import COCOHP, CenterNetMultiPoseEval
from src import convert_eval_format, post_process, merge_outputs
from src import visual_image
from src.config import dataset_config, net_config, eval_config

_current_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='CenterNet evaluation')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'CPU'],
                    help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--load_checkpoint_path", type=str, default="", help="Load checkpoint file path")
parser.add_argument("--data_dir", type=str, default="", help="Dataset directory, "
                                                             "the absolute image path is joined by the data_dir "
                                                             "and the relative path in anno_path")
parser.add_argument("--run_mode", type=str, default="test", help="test or validation, default is test.")
parser.add_argument("--visual_image", type=str, default="false", help="Visulize the ground truth and predicted image")
parser.add_argument("--enable_eval", type=str, default="true", help="Whether evaluate accuracy after prediction")
parser.add_argument("--save_result_dir", type=str, default="", help="The path to save the predict results")

args_opt = parser.parse_args()

def predict():
    '''
    Predict function
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=args_opt.device_id)
        enable_nms_fp16 = True
    else:
        enable_nms_fp16 = False

    logger.info("Begin creating {} dataset".format(args_opt.run_mode))
    coco = COCOHP(dataset_config, run_mode=args_opt.run_mode, net_opt=net_config,
                  enable_visual_image=(args_opt.visual_image == "true"), save_path=args_opt.save_result_dir,)
    coco.init(args_opt.data_dir, keep_res=eval_config.keep_res)
    dataset = coco.create_eval_dataset()

    net_for_eval = CenterNetMultiPoseEval(net_config, eval_config.K, enable_nms_fp16)
    net_for_eval.set_train(False)

    param_dict = load_checkpoint(args_opt.load_checkpoint_path)
    load_param_into_net(net_for_eval, param_dict)

    # save results
    save_path = os.path.join(args_opt.save_result_dir, args_opt.run_mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args_opt.visual_image == "true":
        save_pred_image_path = os.path.join(save_path, "pred_image")
        if not os.path.exists(save_pred_image_path):
            os.makedirs(save_pred_image_path)
        save_gt_image_path = os.path.join(save_path, "gt_image")
        if not os.path.exists(save_gt_image_path):
            os.makedirs(save_gt_image_path)

    total_nums = dataset.get_dataset_size()
    print("\n========================================\n")
    print("Total images num: ", total_nums)
    print("Processing, please wait a moment.")

    pred_annos = {"images": [], "annotations": []}

    index = 0
    for data in dataset.create_dict_iterator(num_epochs=1):
        index += 1
        image = data['image']
        image_id = data['image_id'].asnumpy().reshape((-1))[0]

        # run prediction
        start = time.time()
        detections = []
        for scale in eval_config.multi_scales:
            images, meta = coco.pre_process_for_test(image.asnumpy(), image_id, scale)
            detection = net_for_eval(Tensor(images))
            dets = post_process(detection.asnumpy(), meta, scale)
            detections.append(dets)
        end = time.time()
        print("Image {}/{} id: {} cost time {} ms".format(index, total_nums, image_id, (end - start) * 1000.))

        # post-process
        detections = merge_outputs(detections, eval_config.soft_nms)
        # get prediction result
        pred_json = convert_eval_format(detections, image_id)
        gt_image_info = coco.coco.loadImgs([image_id])

        for image_info in pred_json["images"]:
            pred_annos["images"].append(image_info)
        for image_anno in pred_json["annotations"]:
            pred_annos["annotations"].append(image_anno)
        if args_opt.visual_image == "true":
            img_file = os.path.join(coco.image_path, gt_image_info[0]['file_name'])
            gt_image = cv2.imread(img_file)
            if args_opt.run_mode != "test":
                annos = coco.coco.loadAnns(coco.anns[image_id])
                visual_image(copy.deepcopy(gt_image), annos, save_gt_image_path)
            anno = copy.deepcopy(pred_json["annotations"])
            visual_image(gt_image, anno, save_pred_image_path, score_threshold=eval_config.score_thresh)

    # save results
    save_path = os.path.join(args_opt.save_result_dir, args_opt.run_mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pred_anno_file = os.path.join(save_path, '{}_pred_result.json').format(args_opt.run_mode)
    json.dump(pred_annos, open(pred_anno_file, 'w'))
    pred_res_file = os.path.join(save_path, '{}_pred_eval.json').format(args_opt.run_mode)
    json.dump(pred_annos["annotations"], open(pred_res_file, 'w'))

    if args_opt.run_mode != "test" and args_opt.enable_eval:
        run_eval(coco.annot_path, pred_res_file)


def run_eval(gt_anno, pred_anno):
    """evaluation by coco api"""
    coco = COCO(gt_anno)
    coco_dets = coco.loadRes(pred_anno)
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval = COCOeval(coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    predict()
