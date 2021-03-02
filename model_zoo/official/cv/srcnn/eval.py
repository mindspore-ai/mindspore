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
"""srcnn evaluation"""
import argparse
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import srcnn_cfg as config
from src.dataset import create_eval_dataset
from src.srcnn import SRCNN
from src.metric import SRCNNpsnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="srcnn eval")
    parser.add_argument('--dataset_path', type=str, required=True, help="Dataset, default is None.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="checkpoint file path")
    parser.add_argument('--device_target', type=str, default='GPU', choices=("GPU"),
                        help="Device target, support GPU.")
    args, _ = parser.parse_known_args()

    if args.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.device_target,
                            save_graphs=False)
    else:
        raise ValueError("Unsupported device target.")

    eval_ds = create_eval_dataset(args.dataset_path)

    net = SRCNN()
    lr = Tensor(config.lr, ms.float32)
    opt = nn.Adam(params=net.trainable_params(), learning_rate=lr, eps=1e-07)
    loss = nn.MSELoss(reduction='mean')
    param_dict = load_checkpoint(args.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'PSNR': SRCNNpsnr()})

    res = model.eval(eval_ds, dataset_sink_mode=False)
    print("result ", res)
