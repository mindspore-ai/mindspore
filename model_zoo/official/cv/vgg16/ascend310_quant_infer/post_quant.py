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
import sys
import numpy as np

from amct_mindspore.quantize_tool import create_quant_config
from amct_mindspore.quantize_tool import quantize_model
from amct_mindspore.quantize_tool import save_model
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint
from mindspore.common import dtype as mstype


def quant_vgg(network, dataset, input_data):
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
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, calibration_network.get_parameters()), 0.01, config.momentum,
                   weight_decay=config.weight_decay)
    model = Model(calibration_network, loss_fn=loss, optimizer=opt, metrics={"acc"})
    _ = model.eval(dataset, dataset_sink_mode=False)

    # step5: export the air file
    save_model("results/vgg_quant", calibration_network, input_data)
    print("[INFO] the quantized AIR file has been stored at: \n {}".format("results/vgg_quant.air"))


def run_export():
    """
    Prepare input parameters needed for exporting quantization model.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        config.device_id = get_device_id()
        context.set_context(device_id=config.device_id)

    config.image_size = list(map(int, config.image_size.split(',')))
    if config.dataset == "cifar10":
        net = vgg16(num_classes=config.num_classes, args=config)
    else:
        net = vgg16(config.num_classes, config, phase="test")

    load_checkpoint(config.ckpt_file, net=net)
    net.set_train(False)
    batch_size = 1
    input_data = Tensor(np.zeros([batch_size, 3, config.image_size[0], config.image_size[1]]), mstype.float32)
    dataset = vgg_create_dataset(config.data_dir, config.image_size, batch_size, training=False)
    ds = dataset.take(1)
    quant_vgg(net, ds, input_data)


if __name__ == "__main__":
    sys.path.append("..")
    from src.vgg import vgg16
    from src.dataset import vgg_create_dataset
    from model_utils.moxing_adapter import config
    from model_utils.device_adapter import get_device_id
    run_export()
