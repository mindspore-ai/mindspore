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
"""export checkpoint file into air, mindir models"""
import re
import numpy as np
from mindspore import Tensor, context, load_checkpoint, export, load_param_into_net
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper
from src.network_define import MaskRcnn_Mobilenetv1_Infer

def config_(cfg):
    train_cls = [i for i in re.findall(r'[a-zA-Z\s]+', cfg.coco_classes) if i != ' ']
    cfg.coco_classes = np.array(train_cls)
    lss = [int(re.findall(r'[0-9]+', i)[0]) for i in cfg.feature_shapes]
    cfg.feature_shapes = [(lss[2*i], lss[2*i+1]) for i in range(int(len(lss)/2))]
    cfg.roi_layer = dict(type='RoIAlign', out_size=7, mask_out_size=14, sample_num=2)
    cfg.warmup_ratio = 1/3.0
    cfg.mask_shape = (28, 28)
    return cfg

config = config_(config)

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

def modelarts_process():
    pass

@moxing_wrapper(pre_process=modelarts_process)
def export_maskrcnn_mobilenetv1():
    """ export_maskrcnn_mobilenetv1 """
    config.test_batch_size = config.batch_size_export
    net = MaskRcnn_Mobilenetv1_Infer(config)

    config.batch_size = config.batch_size_export

    param_dict = load_checkpoint(config.ckpt_file)
    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value

    load_param_into_net(net, param_dict_new)
    net.set_train(False)

    img_data = Tensor(np.zeros([config.batch_size, 3, config.img_height, config.img_width], np.float16))
    img_metas = Tensor(np.zeros([config.batch_size, 4], np.float16))

    input_data = [img_data, img_metas]
    export(net, *input_data, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_maskrcnn_mobilenetv1()
