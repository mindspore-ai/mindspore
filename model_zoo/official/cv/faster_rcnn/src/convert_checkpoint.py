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
# ===========================================================================
"""
convert resnet50 pretrain model to faster_rcnn backbone pretrain model
"""
import argparse
from mindspore.train.serialization import load_checkpoint, save_checkpoint
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype

parser = argparse.ArgumentParser(description='load_ckpt')
parser.add_argument('--ckpt_file', type=str, default='', help='ckpt file path')
args_opt = parser.parse_args()
def load_weights(model_path, use_fp16_weight):
    """
    load resnet50 pretrain checkpoint file.

    Args:
        model_path (str): resnet50 pretrain checkpoint file .
        use_fp16_weight(bool): whether save weight into float16.

    Returns:
        parameter list(list): pretrain model weight list.
    """
    ms_ckpt = load_checkpoint(model_path)
    weights = {}
    for msname in ms_ckpt:
        if msname.startswith("layer") or msname.startswith("conv1") or msname.startswith("bn"):
            param_name = "backbone." + msname
        else:
            param_name = msname
        if "down_sample_layer.0" in param_name:
            param_name = param_name.replace("down_sample_layer.0", "conv_down_sample")
        if "down_sample_layer.1" in param_name:
            param_name = param_name.replace("down_sample_layer.1", "bn_down_sample")
        weights[param_name] = ms_ckpt[msname].data.asnumpy()
    if use_fp16_weight:
        dtype = mstype.float16
    else:
        dtype = mstype.float32
    parameter_dict = {}
    for name in weights:
        parameter_dict[name] = Parameter(Tensor(weights[name], dtype), name=name)
    param_list = []
    for key, value in parameter_dict.items():
        param_list.append({"name": key, "data": value})
    return param_list

if __name__ == "__main__":
    parameter_list = load_weights(args_opt.ckpt_file, use_fp16_weight=False)
    save_checkpoint(parameter_list, "resnet50_backbone.ckpt")
