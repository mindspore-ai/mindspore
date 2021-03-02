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
import argparse
import time
import numpy as np
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.retinanet import  retinanet50, resnet50, retinanetInferWithDecoder
from src.dataset import create_retinanet_dataset, data_to_mindrecord_byte_image, voc_data_to_mindrecord
from src.config import config
from src.coco_eval import metrics
from src.box_utils import default_boxes

def retinanet_eval(dataset_path, ckpt_path):
    """retinanet evaluation."""
    batch_size = 1
    ds = create_retinanet_dataset(dataset_path, batch_size=batch_size, repeat_num=1, is_training=False)
    backbone = resnet50(config.num_classes)
    net = retinanet50(backbone, config)
    net = retinanetInferWithDecoder(net, Tensor(default_boxes), config)
    print("Load Checkpoint!")
    param_dict = load_checkpoint(ckpt_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)

    net.set_train(False)
    i = batch_size
    total = ds.get_dataset_size() * batch_size
    start = time.time()
    pred_data = []
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_id = data['img_id']
        img_np = data['image']
        image_shape = data['image_shape']

        output = net(Tensor(img_np))
        for batch_idx in range(img_np.shape[0]):
            pred_data.append({"boxes": output[0].asnumpy()[batch_idx],
                              "box_scores": output[1].asnumpy()[batch_idx],
                              "img_id": int(np.squeeze(img_id[batch_idx])),
                              "image_shape": image_shape[batch_idx]})
        percent = round(i / total * 100., 2)

        print(f'    {str(percent)} [{i}/{total}]', end='\r')
        i += batch_size
    cost_time = int((time.time() - start) * 1000)
    print(f'    100% [{total}/{total}] cost {cost_time} ms')
    mAP = metrics(pred_data)
    print("\n========================================\n")
    print(f"mAP: {mAP}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retinanet evaluation')
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend"),
                        help="run platform, only support Ascend.")
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform, device_id=args_opt.device_id)

    prefix = "retinanet_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if args_opt.dataset == "voc":
        config.coco_root = config.voc_root
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args_opt.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", False, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        elif args_opt.dataset == "voc":
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
    retinanet_eval(mindrecord_file, config.checkpoint_path)
