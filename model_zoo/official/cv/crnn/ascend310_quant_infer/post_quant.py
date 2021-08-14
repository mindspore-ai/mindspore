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
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint


def quant_crnn(network, dataset, input_data):
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
    for data in dataset.create_dict_iterator(num_epochs=1):
        _ = calibration_network(data["image"])

    # step5: export the air file
    save_model("results/crnn_quant", calibration_network, input_data)
    print("[INFO] the quantized AIR file has been stored at: \n {}".format("results/crnn_quant.air"))


def model_export():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=get_device_id())
    config.batch_size = 1
    net = crnn(config)
    load_checkpoint(config.ckpt_file, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([config.batch_size, 3, config.image_height, config.image_width]), ms.float32)

    ds = create_dataset(name=config.eval_dataset,
                        dataset_path=config.eval_dataset_path,
                        batch_size=config.batch_size,
                        is_training=False,
                        config=config)
    dataset = ds.take(1)
    quant_crnn(net, dataset, input_data)


if __name__ == '__main__':
    sys.path.append("..")
    from src.crnn import crnn
    from src.model_utils.config import config
    from src.model_utils.device_adapter import get_device_id
    from src.dataset import create_dataset

    model_export()
