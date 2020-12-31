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
"""export ckpt to model"""
import argparse
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint

from src.bgcf import BGCF
from src.callback import ForwardBGCF

parser = argparse.ArgumentParser(description="bgcf export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="bgcf", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
parser.add_argument("--input_dim", type=int, choices=[64, 128], default=64, help="embedding dimension")
parser.add_argument("--embedded_dimension", type=int, default=64, help="output embedding dimension")
parser.add_argument("--row_neighs", type=int, default=40, help="num of sampling neighbors in raw graph")
parser.add_argument("--gnew_neighs", type=int, default=20, help="num of sampling neighbors in sample graph")
parser.add_argument("--activation", type=str, default="tanh", choices=["relu", "tanh"], help="activation function")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":
    num_user, num_item = 7068, 3570

    network = BGCF([args.input_dim, num_user, num_item],
                   args.embedded_dimension,
                   args.activation,
                   [0.0, 0.0, 0.0],
                   num_user,
                   num_item,
                   args.input_dim)

    load_checkpoint(args.ckpt_file, net=network)

    forward_net = ForwardBGCF(network)

    users = Tensor(np.zeros([num_user,]).astype(np.int32))
    items = Tensor(np.zeros([num_item,]).astype(np.int32))
    neg_items = Tensor(np.zeros([num_item, 1]).astype(np.int32))
    u_test_neighs = Tensor(np.zeros([num_user, args.row_neighs]).astype(np.int32))
    u_test_gnew_neighs = Tensor(np.zeros([num_user, args.gnew_neighs]).astype(np.int32))
    i_test_neighs = Tensor(np.zeros([num_item, args.row_neighs]).astype(np.int32))
    i_test_gnew_neighs = Tensor(np.zeros([num_item, args.gnew_neighs]).astype(np.int32))

    input_data = [users, items, neg_items, u_test_neighs, u_test_gnew_neighs, i_test_neighs, i_test_gnew_neighs]
    export(forward_net, *input_data, file_name=args.file_name, file_format=args.file_format)
