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
Save weight using mindspore, to load the parameters of gpt-2 model from npy file.
npy files should be in the same path with this script. Otherwise you should change the path name of the script.
"""
import os
import argparse
import numpy as np

from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

from .trans_dict import trans_dict_tf


def trans_model_parameter(ckpt_name):
    """
    transform model parameters
    Args:
        ckpt_name (str): the name of the transformed checkpoint.
    """
    file_names = [name for name in os.listdir() if name.endswith(".npy")]
    # to find all file names with suffix '.npy' in the current path.
    new_params_list = []
    for file_name in file_names:
        var_name = file_name[:-4]
        param_dict = {"name": var_name, "data": Tensor(np.load(file_name))}
        if var_name in trans_dict_tf.values():
            new_params_list.append(param_dict)
            print(var_name+" has been saved")

    save_checkpoint(new_params_list, ckpt_name)
    # to load the parameters from npy files and save them as mindspore checkpoint
    print("Finished:the parameters have been saved into mindspore checkpoint.")


def main():
    parser = argparse.ArgumentParser(description="Read GPT-2 model checkpoint weight")
    parser.add_argument("--output_file_name", type=str, default="",
                        help="The name of output checkpoint name")
    args_opt = parser.parse_args()
    ckpt_name = args_opt.output_file_name
    trans_model_parameter(ckpt_name=ckpt_name)


if __name__ == "__main__":
    main()
