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
import argparse

import numpy as np
from mindspore import Tensor
from mindspore.train.serialization import export

from src.config import eval_config
from src.ds_cnn import DSCNN
from src.models import load_ckpt

parser = argparse.ArgumentParser()

args, model_settings = eval_config(parser)
network = DSCNN(model_settings, args.model_size_info)
load_ckpt(network, args.model_dir, False)
x = np.random.uniform(0.0, 1.0, size=[1, 1, model_settings['spectrogram_length'],
                                      model_settings['dct_coefficient_count']]).astype(np.float32)
export(network, Tensor(x), file_name=args.model_dir.replace('.ckpt', '.air'), file_format='AIR')
