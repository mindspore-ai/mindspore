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
"""do post training quantization for Ascend310"""
import os
import sys
import cv2
import numpy as np

from amct_mindspore.quantize_tool import create_quant_config
from amct_mindspore.quantize_tool import quantize_model
from amct_mindspore.quantize_tool import save_model
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


def pre_process(args, img_, crop_size=513):
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(args.image_mean)
    image_std = np.array(args.image_std)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w


def eval_batch(args, img_lst, crop_size=513):
    batch_size = len(img_lst)
    batch_img = np.zeros((batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for l in range(batch_size):
        img_ = img_lst[l]
        img_, resize_h, resize_w = pre_process(args, img_, crop_size)
        batch_img[l] = img_
        resize_hw.append([resize_h, resize_w])

    batch_img = np.ascontiguousarray(batch_img)
    return batch_img


def eval_batch_scales(args, img_lst, scales, base_crop_size):
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    return eval_batch(args, img_lst, crop_size=sizes_[0])


def generate_batch_data():
    config.scales = config.scales_list[config.scales_type]
    args = config
    # data list
    with open(args.data_lst) as f:
        img_lst = f.readlines()

    # evaluate
    batch_img_lst = []
    ori_img_path, _ = img_lst[0].strip().split(" ")
    img_path = "VOCdevkit" + ori_img_path.split("VOCdevkit")[1]

    img_path = os.path.join(args.data_root, img_path)
    img_ = cv2.imread(img_path)
    batch_img_lst.append(img_)
    return eval_batch_scales(args, batch_img_lst, scales=args.scales, base_crop_size=args.crop_size)


def quant_deeplabv3(network, dataset, input_data):
    """
    Export post training quantization model of AIR format.

    Args:
        network: the origin network for inference.
        dataset: the data for inference.
        input_data: the data used for constructing network. The shape and format of input data should be the same as
                    actual data for inference.
    """

    # step2: create the quant config json file
    create_quant_config("./config.json", network, input_data, config_defination="./config.cfg")
    # There is value beyond 1e30 in this layer. So this layer will not be quantized.

    # step3: do some network modification and return the modified network
    calibration_network = quantize_model("./config.json", network, input_data)
    calibration_network.set_train(False)

    # step4: perform the evaluation of network to do activation calibration
    _ = calibration_network(Tensor(dataset))

    # step5: export the air file
    save_model("results/deeplabv3_quant", calibration_network, input_data)
    print("[INFO] the quantized AIR file has been stored at: \n {}".format("results/deeplabv3_quant.air"))


class BuildEvalNetwork(nn.Cell):
    def __init__(self, net, input_format="NCHW"):
        super(BuildEvalNetwork, self).__init__()
        self.network = net
        self.softmax = nn.Softmax(axis=1)
        self.transpose = ops.Transpose()
        self.format = input_format

    def construct(self, x):
        if self.format == "NHWC":
            x = self.transpose(x, (0, 3, 1, 2))
        output = self.network(x)
        output = self.softmax(output)
        return output


def run_export():
    '''run export.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    config.freeze_bn = True
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=config.device_id)

    if config.export_model == 'deeplab_v3_s16':
        network = net_factory.nets_map['deeplab_v3_s16']('eval', config.num_classes, 16, True)
    else:
        network = net_factory.nets_map['deeplab_v3_s8']('eval', config.num_classes, 8, True)
    network = BuildEvalNetwork(network, config.input_format)
    param_dict = load_checkpoint(config.ckpt_file)

    # load the parameter into net
    load_param_into_net(network, param_dict)
    batch_size = 1
    if config.input_format == "NHWC":
        input_data = Tensor(
            np.ones([batch_size, config.input_size, config.input_size, 3]).astype(np.float32))
    else:
        input_data = Tensor(
            np.ones([batch_size, 3, config.input_size, config.input_size]).astype(np.float32))
    batch_data = generate_batch_data()
    quant_deeplabv3(network, batch_data, input_data)


if __name__ == "__main__":
    sys.path.append("..")
    from src.nets import net_factory
    from model_utils.config import config

    run_export()
