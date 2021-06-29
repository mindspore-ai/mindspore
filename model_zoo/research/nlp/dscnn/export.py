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
# ===========================================================================
"""DSCNN export."""
import os
import numpy as np
from mindspore import Tensor
from mindspore.train.serialization import export
from src.ds_cnn import DSCNN
from src.models import load_ckpt
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=None)
def model_export():
    network = DSCNN(config, config.model_size_info)
    load_ckpt(network, config.export_ckpt_path, False)
    x = np.random.uniform(0.0, 1.0, size=[config.per_batch_size, 1, config.model_setting_spectrogram_length,
                                          config.model_setting_dct_coefficient_count]).astype(np.float32)
    export(network, Tensor(x), file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    model_export()
