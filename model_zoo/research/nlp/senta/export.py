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
"""export checkpoint file into models"""
import argparse
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, context, load_checkpoint, export
from src.config import bertcfg as bert_net_cfg
from src.models.roberta_one_sent_classification_en import RobertaOneSentClassificationEn

parser = argparse.ArgumentParser(description="Bert export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument(
    "--ckpt_file",
    type=str,
    required=True,
    help="Senta ckpt file.")
parser.add_argument(
    "--file_name",
    type=str,
    default="Senta",
    help="Senta output air name.")
parser.add_argument(
    "--file_format",
    type=str,
    choices=[
        "AIR",
        "ONNX",
        "MINDIR"],
    default="MINDIR",
    help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU", "CPU"], help="device target (default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)


if __name__ == "__main__":

    net = RobertaOneSentClassificationEn(bert_net_cfg, False)
    load_checkpoint(args.ckpt_file, net=net)
    net.set_train(False)

    src_ids = Tensor(
        np.zeros([args.batch_size, bert_net_cfg.seq_length]), mstype.int32)
    sent_ids = Tensor(
        np.zeros([args.batch_size, bert_net_cfg.seq_length]), mstype.int32)
    mask_ids = Tensor(
        np.zeros([args.batch_size, bert_net_cfg.seq_length]), mstype.int32)

    input_data = [src_ids, sent_ids, mask_ids]

    export(
        net.bert,
        *input_data,
        file_name=args.file_name,
        file_format=args.file_format)
