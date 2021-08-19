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
import mindspore
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net


def quant_ssd(network, dataset, input_data):
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

    # step3: do some network modification and return the modified network
    calibration_network = quantize_model("./config.json", network, input_data)
    calibration_network.set_train(False)

    # step4: perform the evaluation of network to do activation calibration
    for data in dataset.create_dict_iterator(num_epochs=1):
        img_data = data["image"]
        _ = calibration_network(img_data)

    # step5: export the air file
    save_model("results/ssd_quant", calibration_network, input_data)
    print("[INFO] the quantized AIR file has been stored at: \n {}".format("results/ssd_quant.air"))


def run_export():
    """
    Prepare input parameters needed for exporting quantization model.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=config.device_id)
    if hasattr(config, "num_ssd_boxes") and config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.step[i]) * (w // config.step[i]) * config.num_default[i]
        config.num_ssd_boxes = num
    net = SSD300(ssd_mobilenet_v2(), config, is_training=False)
    net = SsdInferWithDecoder(net, Tensor(default_boxes), config)
    param_dict = load_checkpoint(config.checkpoint_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)
    batch_size = 1
    input_shp = [batch_size, 3] + config.img_shape
    inputs = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp), mindspore.float32)
    mindrecord_file = create_mindrecord("coco", "ssd_eval.mindrecord", False)
    batch_size = 1
    datasets = create_ssd_dataset(mindrecord_file, batch_size=batch_size, repeat_num=1, is_training=False,
                                  use_multiprocessing=False)
    ds = datasets.take(1)
    quant_ssd(net, ds, inputs)


if __name__ == "__main__":
    sys.path.append("..")
    from src.ssd import SSD300, SsdInferWithDecoder, ssd_mobilenet_v2
    from src.model_utils.config import config
    from src.dataset import create_ssd_dataset, create_mindrecord
    from src.box_utils import default_boxes
    run_export()
