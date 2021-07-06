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
"""export checkpoint file into air models"""
import numpy as np

from mindspore import context, load_distributed_checkpoint
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from eval import CPM_LAYER, create_ckpt_file_list

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target="Ascend")

def modelarts_pre_process():
    '''modelarts pre process function.'''

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''export cpm network'''
    finetune_test_standalone = config.finetune_test_standalone
    cpm_model = CPM_LAYER(finetune_test_standalone)

    if not config.has_train_strategy:
        weights = load_checkpoint(config.ckpt_path_doc)
        can_be_loaded = {}
        print("+++++++loading weights+++++")
        for name, _ in weights.items():
            print('oldname:           ' + name)
            if 'cpm_model.' not in name:
                can_be_loaded['cpm_model.' + name] = weights[name]
                print('newname: cpm_model.' + name)
            else:
                can_be_loaded[name] = weights[name]
        print("+++++++loaded weights+++++")
        load_param_into_net(cpm_model, parameter_dict=can_be_loaded)
    else:
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=config.ckpt_path_doc + "/train_strategy.ckpt"
        )
        ckpt_file_list = create_ckpt_file_list(config)
        print("Get checkpoint file lists++++", ckpt_file_list, flush=True)
        load_distributed_checkpoint(cpm_model, ckpt_file_list, None)

    input_ids = Tensor(np.ones((finetune_test_standalone.batch_size, finetune_test_standalone.seq_length)),
                       mstype.int64)
    position_ids = Tensor(np.random.randint(0, 10, [finetune_test_standalone.batch_size,
                                                    finetune_test_standalone.seq_length]), mstype.int64)
    attention_mask = Tensor(np.random.randn(finetune_test_standalone.batch_size,
                                            finetune_test_standalone.seq_length,
                                            finetune_test_standalone.seq_length), mstype.float16)

    export(cpm_model, input_ids, position_ids, attention_mask, file_name=config.file_name,
           file_format=config.file_format)

if __name__ == '__main__':
    run_export()
