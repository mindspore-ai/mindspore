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

"""Evaluation for SSD"""

import os
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.ssd import SSD300, SsdInferWithDecoder, ssd_mobilenet_v2, ssd_mobilenet_v1_fpn, ssd_resnet50_fpn, ssd_vgg16
from src.dataset import create_ssd_dataset, create_mindrecord
from src.eval_utils import apply_eval
from src.box_utils import default_boxes
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

def ssd_eval(dataset_path, ckpt_path, anno_json):
    """SSD evaluation."""
    batch_size = 1
    ds = create_ssd_dataset(dataset_path, batch_size=batch_size, repeat_num=1,
                            is_training=False, use_multiprocessing=False)
    if config.model_name == "ssd300":
        net = SSD300(ssd_mobilenet_v2(), config, is_training=False)
    elif config.model_name == "ssd_vgg16":
        net = ssd_vgg16(config=config)
    elif config.model_name == "ssd_mobilenet_v1_fpn":
        net = ssd_mobilenet_v1_fpn(config=config)
    elif config.model_name == "ssd_resnet50_fpn":
        net = ssd_resnet50_fpn(config=config)
    else:
        raise ValueError(f'config.model: {config.model_name} is not supported')
    net = SsdInferWithDecoder(net, Tensor(default_boxes), config)

    print("Load Checkpoint!")
    param_dict = load_checkpoint(ckpt_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)

    net.set_train(False)
    total = ds.get_dataset_size() * batch_size
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    eval_param_dict = {"net": net, "dataset": ds, "anno_json": anno_json}
    mAP = apply_eval(eval_param_dict)
    print("\n========================================\n")
    print(f"mAP: {mAP}")

@moxing_wrapper()
def eval_net():
    if hasattr(config, 'num_ssd_boxes') and config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.steps[i]) * (w // config.steps[i]) * config.num_default[i]
        config.num_ssd_boxes = num

    if config.dataset == "coco":
        coco_root = os.path.join(config.data_path, config.coco_root)
        json_path = os.path.join(coco_root, config.instances_set.format(config.val_data_type))
    elif config.dataset == "voc":
        voc_root = os.path.join(config.data_path, config.voc_root)
        json_path = os.path.join(voc_root, config.voc_json)
    else:
        raise ValueError('SSD eval only support dataset mode is coco and voc!')

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

    mindrecord_file = create_mindrecord(config.dataset, "ssd_eval.mindrecord", False)

    print("Start Eval!")
    ssd_eval(mindrecord_file, config.checkpoint_file_path, json_path)

if __name__ == '__main__':
    eval_net()
