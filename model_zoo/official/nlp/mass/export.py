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

import os
import numpy as np

from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export

from src.utils.load_weights import load_infer_weights
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.transformer.transformer_for_infer import TransformerInferModel


def get_config():
    config.compute_type = mstype.float16 if config.compute_type == "float16" else mstype.float32
    config.dtype = mstype.float16 if config.dtype == "float16" else mstype.float32

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    get_config()

    tfm_model = TransformerInferModel(config=config, use_one_hot_embeddings=False)
    tfm_model.init_parameters_data()

    params = tfm_model.trainable_params()
    weights = load_infer_weights(config)

    for param in params:
        value = param.data
        name = param.name

        if name not in weights:
            raise ValueError(f'{name} is not found in weights.')

        with open('weight_after_deal.txt', 'a+') as f:
            weights_name = name
            f.write(weights_name + '\n')

            if isinstance(value, Tensor):
                if weights_name in weights:
                    assert weights_name in weights
                    param.set_data(Tensor(weights[weights_name], mstype.float32))
                else:
                    raise ValueError(f'{weights_name} is not found in checkpoint')
            else:
                raise TypeError(f'Type of {weights_name} is not Tensor')

    print('    |    Load weights successfully.')
    tfm_model.set_train(False)

    source_ids = Tensor(np.ones((1, config.seq_length)).astype(np.int32))
    source_mask = Tensor(np.ones((1, config.seq_length)).astype(np.int32))

    export(tfm_model, source_ids, source_mask, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    run_export()
