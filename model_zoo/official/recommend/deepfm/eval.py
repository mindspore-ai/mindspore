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
"""train_criteo."""
import os
import sys
import time

from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.deepfm import ModelBuilder, AUCMetric
from src.dataset import create_dataset, DataType

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device_id = get_device_id() # int(os.getenv('DEVICE_ID', '0'))
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)

def add_write(file_path, print_str):
    with open(file_path, 'a+', encoding='utf-8') as file_out:
        file_out.write(print_str + '\n')


def modelarts_process():
    pass

@moxing_wrapper(pre_process=modelarts_process)
def eval_deepfm():
    """ eval_deepfm """
    ds_eval = create_dataset(config.dataset_path, train_mode=False,
                             epochs=1, batch_size=config.batch_size,
                             data_type=DataType(config.data_format))
    if config.convert_dtype:
        config.convert_dtype = config.device_target != "CPU"
    model_builder = ModelBuilder(config, config)
    train_net, eval_net = model_builder.get_train_eval_net()
    train_net.set_train()
    eval_net.set_train(False)
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})
    param_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(eval_net, param_dict)

    start = time.time()
    res = model.eval(ds_eval)
    eval_time = time.time() - start
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    out_str = f'{time_str} AUC: {list(res.values())[0]}, eval time: {eval_time}s.'
    print(out_str)
    add_write('./auc.log', str(out_str))

if __name__ == '__main__':
    eval_deepfm()
