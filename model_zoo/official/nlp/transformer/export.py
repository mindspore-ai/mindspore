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
"""export checkpoint file into air models"""

import numpy as np

from mindspore import Tensor, context
from mindspore.train.serialization import load_param_into_net, export

from src.transformer_model import TransformerModel
from src.eval_config import cfg, transformer_net_cfg
from eval import load_weights

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

if __name__ == '__main__':
    tfm_model = TransformerModel(config=transformer_net_cfg, is_training=False, use_one_hot_embeddings=False)

    parameter_dict = load_weights(cfg.model_file)
    load_param_into_net(tfm_model, parameter_dict)

    source_ids = Tensor(np.ones((1, 128)).astype(np.int32))
    source_mask = Tensor(np.ones((1, 128)).astype(np.int32))

    dec_len = transformer_net_cfg.max_decode_length

    export(tfm_model, source_ids, source_mask, file_name="len" + str(dec_len) + ".air", file_format="AIR")
