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
""" preprocess """
import os
import json
import numpy as np
from src.args import get_args


if __name__ == "__main__":
    args_opt = get_args('pre310')
    with open(args_opt.config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())

    nz = generator_config["nz"]

    # initialize noise
    fixed_noise = np.random.normal(size=[args_opt.nimages, nz, 1, 1]).astype(np.float32)
    file_name = "wgan_bs" + str(args_opt.nimages) + ".bin"
    file_path = os.path.join(args_opt.pre_result_path, file_name)
    fixed_noise.tofile(file_path)
    print("*" * 20, "export bin files finished", "*" * 20)
