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
"""export checkpoint file into air, onnx, mindir models"""
import argparse
import numpy as np

from mindspore import Tensor, context, load_checkpoint, export
import mindspore.common.dtype as mstype

from src.config import Config_CNNCTC
from src.cnn_ctc import CNNCTC_Model

parser = argparse.ArgumentParser(description="CNNCTC_export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--file_name", type=str, default="cnn_ctc", help="CNN&CTC output air name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                    help="device target")
parser.add_argument("--ckpt_file", type=str, default="./ckpts/cnn_ctc.ckpt", help="CNN&CTC ckpt file.")
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
if args_opt.device_target == "Ascend":
    context.set_context(device_id=args_opt.device_id)

if __name__ == "__main__":
    cfg = Config_CNNCTC()
    ckpt_path = cfg.CKPT_PATH

    if args_opt.ckpt_file != "":
        ckpt_path = args_opt.ckpt_file

    net = CNNCTC_Model(cfg.NUM_CLASS, cfg.HIDDEN_SIZE, cfg.FINAL_FEATURE_WIDTH)

    load_checkpoint(ckpt_path, net=net)

    bs = cfg.TEST_BATCH_SIZE

    input_data = Tensor(np.zeros([bs, 3, cfg.IMG_H, cfg.IMG_W]), mstype.float32)

    export(net, input_data, file_name=args_opt.file_name, file_format=args_opt.file_format)
