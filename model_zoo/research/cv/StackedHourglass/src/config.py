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
"""
model config
"""
import argparse
import ast


def parse_args():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser(description="MindSpore Stacked Hourglass")

    # Model
    parser.add_argument("--nstack", type=int, default=2)
    parser.add_argument("--inp_dim", type=int, default=256)
    parser.add_argument("--oup_dim", type=int, default=16)
    parser.add_argument("--input_res", type=int, default=256)
    parser.add_argument("--output_res", type=int, default=64)
    parser.add_argument("--annot_dir", type=str, default="./MPII/annot")
    parser.add_argument("--img_dir", type=str, default="./MPII/images")
    # Context
    parser.add_argument("--context_mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"])
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "CPU"])
    # Train
    parser.add_argument("--parallel", type=ast.literal_eval, default=False)
    parser.add_argument("--amp_level", type=str, default="O2", choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--save_checkpoint_epochs", type=int, default=5)
    parser.add_argument("--keep_checkpoint_max", type=int, default=20)
    parser.add_argument("--loss_log_interval", type=int, default=1)
    parser.add_argument("--initial_lr", type=float, default=1e-3)
    parser.add_argument("--decay_rate", type=float, default=0.985)
    parser.add_argument("--decay_epoch", type=int, default=1)
    # Valid
    parser.add_argument("--num_eval", type=int, default=2958)
    parser.add_argument("--train_num_eval", type=int, default=300)
    parser.add_argument("--ckpt_file", type=str, default="")
    # Export
    parser.add_argument("--file_name", type=str, default="stackedhourglass")
    parser.add_argument("--file_format", type=str, default="MINDIR")

    args = parser.parse_known_args()[0]

    return args
