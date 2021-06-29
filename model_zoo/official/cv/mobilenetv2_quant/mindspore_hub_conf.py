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
"""hub config"""
from mindspore.compression.quant import QuantizationAwareTraining
from src.config import config_ascend_quant
from src.mobilenetV2 import mobilenetV2

def mobilenetv2_quant_net(*args, **kwargs):
    symmetric_list = [False, False]
    # define fusion network
    network = mobilenetV2(num_classes=config_ascend_quant.num_classes)
    # convert fusion network to quantization aware network
    quantizer = QuantizationAwareTraining(bn_fold=True,
                                          per_channel=[True, False],
                                          symmetric=symmetric_list)
    network = quantizer.quantize(network)
    return network

def create_network(name, *args, **kwargs):
    """create_network about mobilenetv2_quant"""
    if name == "mobilenetv2_quant":
        return mobilenetv2_quant_net(*args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
