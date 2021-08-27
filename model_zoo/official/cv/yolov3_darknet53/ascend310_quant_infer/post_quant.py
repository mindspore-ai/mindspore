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
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net


def quant_yolov3(network, dataset, input_data):
    """
    Export post training quantization model of AIR format.

    Args:
        network: the origin network for inference.
        dataset: the data for inference.
        input_data: the data used for constructing network. The shape and format of input data should be the same as
                    actual data for inference.
    """

    # step2: create the quant config json file
    create_quant_config("./config.json", network, input_data)

    # step3: do some network modification and return the modified network
    calibration_network = quantize_model("./config.json", network, input_data)
    calibration_network.set_train(False)

    # step4: perform the evaluation of network to do activation calibration
    for _, data in enumerate(dataset.create_dict_iterator(num_epochs=1)):
        image = data["image"]
        _ = calibration_network(image)

    # step5: export the air file
    save_model("results/yolov3_quant", calibration_network, input_data)
    print("[INFO] the quantized AIR file has been stored at: \n {}".format("results/yolov3_quant.air"))


if __name__ == "__main__":
    sys.path.append("..")
    from src.yolo import YOLOV3DarkNet53
    from src.yolo_dataset import create_yolo_dataset
    from model_utils.config import config

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = YOLOV3DarkNet53(is_training=False)

    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    config.batch_size = 16
    data_path = os.path.join(config.data_dir, "val2014")
    datasets, data_size = create_yolo_dataset(data_path, config.annFile, is_training=False,
                                              batch_size=config.batch_size, max_epoch=1, device_num=1, rank=0,
                                              shuffle=False, config=config)
    ds = datasets.take(1)
    export_batch_size = 1
    shape = [export_batch_size, 3] + config.test_img_shape
    inputs = Tensor(np.zeros(shape), ms.float32)
    quant_yolov3(net, ds, inputs)
