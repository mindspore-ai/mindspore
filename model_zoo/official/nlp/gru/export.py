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
"""export script."""
import argparse
import numpy as np
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.seq2seq import Seq2Seq
from src.gru_for_infer import GRUInferCell
from src.config import config

parser = argparse.ArgumentParser(description='export')
parser.add_argument("--device_target", type=str, default="Ascend",
                    help="device where the code will be implemented, default is Ascend")
parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend, default is 0')
parser.add_argument('--file_name', type=str, default="gru", help='output file name.')
parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format.")
parser.add_argument('--ckpt_file', type=str, required=True, help='ckpt file path')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, reserve_class_name_in_scope=False, \
                    device_id=args.device_id, save_graphs=False)

if __name__ == "__main__":
    network = Seq2Seq(config, is_training=False)
    network = GRUInferCell(network)
    network.set_train(False)
    if args.ckpt_file != "":
        parameter_dict = load_checkpoint(args.ckpt_file)
        load_param_into_net(network, parameter_dict)

    source_ids = Tensor(np.random.uniform(0.0, 1e5, size=[config.eval_batch_size, config.max_length]).astype(np.int32))
    target_ids = Tensor(np.random.uniform(0.0, 1e5, size=[config.eval_batch_size, config.max_length]).astype(np.int32))
    export(network, source_ids, target_ids, file_name=args.file_name, file_format=args.file_format)
