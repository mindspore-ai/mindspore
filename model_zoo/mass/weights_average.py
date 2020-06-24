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
"""Weight average."""
import os
import argparse
import numpy as np
from mindspore.train.serialization import load_checkpoint

parser = argparse.ArgumentParser(description='transformer')
parser.add_argument("--input_files", type=str, default=None, required=False,
                    help="Multi ckpt files path.")
parser.add_argument("--input_folder", type=str, default=None, required=False,
                    help="Ckpt files folder.")
parser.add_argument("--output_file", type=str, default=None, required=True,
                    help="Output model file path.")


def average_me_models(ckpt_list):
    """
    Average multi ckpt params.

    Args:
        ckpt_list (list): Ckpt paths.

    Returns:
        dict, params dict.
    """
    avg_model = {}
    # load all checkpoint
    for ckpt in ckpt_list:
        if not ckpt.endswith(".ckpt"):
            continue
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint file is not existed.")

        print(f" | Loading ckpt from {ckpt}.")
        ms_ckpt = load_checkpoint(ckpt)
        for param_name in ms_ckpt:
            if param_name not in avg_model:
                avg_model[param_name] = []
            avg_model[param_name].append(ms_ckpt[param_name].data.asnumpy())

    for name in avg_model:
        avg_model[name] = sum(avg_model[name]) / float(len(ckpt_list))

    return avg_model


def main():
    """Entry point."""
    args, _ = parser.parse_known_args()

    if not args.input_files and not args.input_folder:
        raise ValueError("`--input_files` or `--input_folder` must be provided one as least.")

    ckpt_list = []
    if args.input_files:
        ckpt_list.extend(args.input_files.split(","))

    if args.input_folder and os.path.exists(args.input_folder) and os.path.isdir(args.input_folder):
        for file in os.listdir(args.input_folder):
            ckpt_list.append(os.path.join(args.input_folder, file))

    avg_weights = average_me_models(ckpt_list)
    np.savez(args.output_file, **avg_weights)


if __name__ == '__main__':
    main()
