# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
##############export checkpoint file into air and onnx models#################
"""
import numpy as np

from mindspore import Tensor, nn
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.wide_and_deep import WideDeepModel
from src.config import WideDeepConfig

class PredictWithSigmoid(nn.Cell):
    """
    PredictWithSigmoid
    """
    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__()
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, batch_ids, batch_wts):
        logits, _, = self.network(batch_ids, batch_wts)
        pred_probs = self.sigmoid(logits)
        return pred_probs

def get_WideDeep_net(config):
    """
    Get network of wide&deep predict model.
    """
    WideDeep_net = WideDeepModel(config)
    eval_net = PredictWithSigmoid(WideDeep_net)
    return eval_net

if __name__ == '__main__':
    widedeep_config = WideDeepConfig()
    widedeep_config.argparse_init()
    ckpt_path = widedeep_config.ckpt_path
    net = get_WideDeep_net(widedeep_config)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    ids = Tensor(np.ones([widedeep_config.eval_batch_size, widedeep_config.field_size]).astype(np.int32))
    wts = Tensor(np.ones([widedeep_config.eval_batch_size, widedeep_config.field_size]).astype(np.float32))
    input_tensor_list = [ids, wts]
    export(net, *input_tensor_list, file_name='wide_and_deep', file_format="ONNX")
    export(net, *input_tensor_list, file_name='wide_and_deep', file_format="AIR")
