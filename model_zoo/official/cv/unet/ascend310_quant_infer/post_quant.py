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
from mindspore import Tensor, context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

def quant_unet(network, dataset, input_data):
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
    model = Model(calibration_network, loss_fn=TempLoss(), metrics={"dice_coff": dice_coeff()})
    _ = model.eval(dataset, dataset_sink_mode=False)

    # step5: export the air file
    save_model("results/unet_quant", calibration_network, input_data)
    print("[INFO] the quantized AIR file has been stored at: \n {}".format("results/unet_quant.air"))


def run_export():
    """run export."""
    if config.model_name == 'unet_medical':
        net = UNetMedical(n_channels=config.num_channels, n_classes=config.num_classes)
    else:
        raise ValueError("post training quantization currently does not support model: {}".format(config.model_name))
    # return a parameter dict for model
    param_dict = load_checkpoint(config.checkpoint_file_path)
    # load the parameter into net
    load_param_into_net(net, param_dict)
    net = UnetEval(net, eval_activate="softmax")
    batch_size = 1
    input_data = Tensor(np.ones([batch_size, config.num_channels, config.height, config.width]).astype(np.float32))
    _, valid_dataset = create_dataset(config.data_path, 1, batch_size, False, 1, False, do_crop=config.crop,
                                      img_size=config.image_size)
    dataset = valid_dataset.take(1)
    quant_unet(net, dataset, input_data)


if __name__ == "__main__":
    sys.path.append("..")
    from src.data_loader import create_dataset
    from src.unet_medical import UNetMedical
    from src.utils import UnetEval, TempLoss, dice_coeff
    from src.model_utils.config import config
    from src.model_utils.device_adapter import get_device_id

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=get_device_id())

    run_export()
