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
import time
import numpy as np

from mindspore import Tensor, context, Parameter
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export

from src.gnmt_model.gnmt import GNMT
from src.gnmt_model.gnmt_for_infer import GNMTInferCell
from src.utils import zero_weight
from src.utils.load_weights import load_infer_weights
from src.utils.get_config import get_config

from model_utils.config import config as default_config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, default_config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if default_config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(default_config.data_path, default_config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(default_config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    default_config.file_name = os.path.join(default_config.output_path, default_config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''run export.'''
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend",
                        reserve_class_name_in_scope=False)

    config = get_config(default_config)

    tfm_model = GNMT(config=config,
                     is_training=False,
                     use_one_hot_embeddings=False)

    params = tfm_model.trainable_params()
    weights = load_infer_weights(config)

    for param in params:
        value = param.data
        weights_name = param.name
        if weights_name not in weights:
            raise ValueError(f"{weights_name} is not found in weights.")
        if isinstance(value, Tensor):
            if weights_name in weights:
                assert weights_name in weights
                if isinstance(weights[weights_name], Parameter):
                    if param.data.dtype == "Float32":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float32))
                    elif param.data.dtype == "Float16":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float16))

                elif isinstance(weights[weights_name], Tensor):
                    param.set_data(Tensor(weights[weights_name].asnumpy(), config.dtype))
                elif isinstance(weights[weights_name], np.ndarray):
                    param.set_data(Tensor(weights[weights_name], config.dtype))
                else:
                    param.set_data(weights[weights_name])
            else:
                print("weight not found in checkpoint: " + weights_name)
                param.set_data(zero_weight(value.asnumpy().shape))

    print(" | Load weights successfully.")
    tfm_infer = GNMTInferCell(tfm_model)
    tfm_infer.set_train(False)

    source_ids = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))
    source_mask = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))

    export(tfm_infer, source_ids, source_mask, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    run_export()
