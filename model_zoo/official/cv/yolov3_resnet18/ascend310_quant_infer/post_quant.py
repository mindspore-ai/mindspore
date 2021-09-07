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
import numpy as np

from amct_mindspore.quantize_tool import create_quant_config
from amct_mindspore.quantize_tool import quantize_model
from amct_mindspore.quantize_tool import save_model
import mindspore as ms
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def quant_yolov3_resnet(network, dataset, input_data):
    """
    Export post training quantization model of AIR format.

    Args:
        network: the origin network for inference.
        dataset: the data for inference.
        input_data: the data used for constructing network. The shape and format of input data should be the same as
                    actual data for inference.
    """

    # step2: create the quant config json file
    create_quant_config("./config.json", network, *input_data)

    # step3: do some network modification and return the modified network
    calibration_network = quantize_model("./config.json", network, *input_data)
    calibration_network.set_train(False)

    # step4: perform the evaluation of network to do activation calibration
    concat = ops.Concat()
    index = 0
    image_data = []
    for data in dataset.create_dict_iterator(num_epochs=1):
        index += 1
        if index == 1:
            image_data = data["image"]
        else:
            image_data = concat((image_data, data["image"]))
        if index == dataset.get_dataset_size():
            _ = calibration_network(image_data, data["image_shape"])

    # step5: export the air file
    save_model("results/yolov3_resnet_quant", calibration_network, *input_data)
    print("[INFO] the quantized AIR file has been stored at: \n {}".format("results/yolov3_resnet_quant.air"))


def export_yolov3_resnet():
    """ prepare for quantization of yolov3_resnet """
    cfg = ConfigYOLOV3ResNet18()
    net = yolov3_resnet18(cfg)
    eval_net = YoloWithEval(net, cfg)
    param_dict = load_checkpoint(default_config.ckpt_file)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)

    default_config.export_batch_size = 1
    shape = [default_config.export_batch_size, 3] + cfg.img_shape
    input_data = Tensor(np.zeros(shape), ms.float32)
    input_shape = Tensor(np.zeros([1, 2]), ms.float32)
    inputs = (input_data, input_shape)

    if not os.path.isdir(default_config.eval_mindrecord_dir):
        os.makedirs(default_config.eval_mindrecord_dir)

    yolo_prefix = "yolo.mindrecord"
    mindrecord_file = os.path.join(default_config.eval_mindrecord_dir, yolo_prefix + "0")
    if not os.path.exists(mindrecord_file):
        if os.path.isdir(default_config.image_dir) and os.path.exists(default_config.anno_path):
            print("Create Mindrecord")
            data_to_mindrecord_byte_image(default_config.image_dir,
                                          default_config.anno_path,
                                          default_config.eval_mindrecord_dir,
                                          prefix=yolo_prefix,
                                          file_num=8)
            print("Create Mindrecord Done, at {}".format(default_config.eval_mindrecord_dir))
        else:
            print("image_dir or anno_path not exits")
    datasets = create_yolo_dataset(mindrecord_file, is_training=False)
    ds = datasets.take(16)
    quant_yolov3_resnet(eval_net, ds, inputs)


if __name__ == "__main__":
    sys.path.append("..")
    from src.yolov3 import yolov3_resnet18, YoloWithEval
    from src.config import ConfigYOLOV3ResNet18
    from src.dataset import create_yolo_dataset, data_to_mindrecord_byte_image
    from model_utils.config import config as default_config

    export_yolov3_resnet()
