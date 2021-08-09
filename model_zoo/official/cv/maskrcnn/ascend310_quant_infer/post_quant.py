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
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def quant_maskrcnn(network, dataset, input_data):
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
    for data in dataset.create_dict_iterator(num_epochs=1):
        _ = calibration_network(data["image"], data["image_shape"])

    # step5: export the air file
    save_model("results/maskrcnn_quant", calibration_network, *input_data)
    print("[INFO] the quantized AIR file has been stored at: \n {}".format("results/maskrcnn_quant.air"))


def export_maskrcnn():
    """ export_maskrcnn """
    config.test_batch_size = 1
    config.batch_size = config.test_batch_size
    net = MaskRcnn_Infer(config=config)
    param_dict = load_checkpoint(config.ckpt_file)
    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value
    load_param_into_net(net, param_dict_new)
    net.set_train(False)

    img = Tensor(np.zeros([config.batch_size, 3, config.img_height, config.img_width], np.float16))
    img_metas = Tensor(np.zeros([config.batch_size, 4], np.float16))
    input_data = [img, img_metas]

    mindrecord_file = os.path.join(config.mindrecord_dir, "MaskRcnn_eval.mindrecord")
    ds = create_maskrcnn_dataset(mindrecord_file, batch_size=config.batch_size, is_training=False)
    dataset = ds.take(1)
    quant_maskrcnn(net, dataset, input_data)


if __name__ == "__main__":
    sys.path.append("..")
    from src.model_utils.config import config
    from src.maskrcnn.mask_rcnn_r50 import MaskRcnn_Infer
    from src.dataset import create_maskrcnn_dataset

    export_maskrcnn()
