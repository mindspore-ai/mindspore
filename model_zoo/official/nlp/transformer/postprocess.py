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
"""Transformer evaluation script."""

import os
import numpy as np
from src.model_utils.config import config


def generate_output():
    '''
    Generate output.
    '''
    predictions = []
    file_num = len(os.listdir(config.result_dir))
    for i in range(file_num):
        batch = "transformer_bs_" + str(config.batch_size) + "_" + str(i) + "_0.bin"
        pred = np.fromfile(os.path.join(config.result_dir, batch), np.int32)
        predictions.append(pred.reshape(1, 1, config.max_decode_length + 1))

    # decode and write to file
    f = open(config.output_file, 'w')
    for batch_out in predictions:
        for i in range(config.batch_size):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]
            token_ids = [str(x) for x in batch_out[i].tolist()]
            f.write(" ".join(token_ids) + "\n")
    f.close()


if __name__ == "__main__":
    generate_output()
