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
# ===========================================================================
""" eval model"""
import argparse
import os

from src.config import ModelConfig
from src.dataset import get_mindrecord_dataset
from src.fat_deepffm import ModelBuilder
from src.metrics import AUCMetric
from mindspore import context, Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser(description='CTR Prediction')
parser.add_argument('--dataset_path', type=str, default="/data/FM/mindrecord", help='Dataset path')
parser.add_argument('--ckpt_path', type=str, default="/checkpoint/Fat-DeepFFM-24_5166.ckpt", help='Checkpoint path')
parser.add_argument('--eval_file_name', type=str, default="./auc.log",
                    help='Auc log file path. Default: "./auc.log"')
parser.add_argument('--loss_file_name', type=str, default="./loss.log",
                    help='Loss log file path. Default: "./loss.log"')
parser.add_argument('--device_target', type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                    help="device target, support Ascend, GPU and CPU.")
parser.add_argument('--device_id', type=int, default=0, choices=(0, 1, 2, 3, 4, 5, 6, 7),
                    help="device target, support Ascend, GPU and CPU.")
args = parser.parse_args()
rank_size = int(os.environ.get("RANK_SIZE", 1))
print("rank_size", rank_size)

set_seed(1)

if __name__ == '__main__':
    model_config = ModelConfig()
    device_id = int(os.getenv('DEVICE_ID', default=args.device_id))
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        device_id=device_id)
    print("Load dataset...")
    train_net, test_net = ModelBuilder(model_config).get_train_eval_net()
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=test_net, metrics={"AUC": auc_metric})
    ds_test = get_mindrecord_dataset(args.dataset_path, train_mode=False)
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(train_net, param_dict)
    print("Training started...")
    res = model.eval(ds_test, dataset_sink_mode=False)
    out_str = f'AUC: {list(res.values())[0]}'
    print(res)
    print(out_str)
