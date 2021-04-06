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

"""Evaluation for CTPN"""
import argparse
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.ctpn import CTPN
from src.config import config
from src.dataset import create_ctpn_dataset
from src.eval_utils import eval_for_ctpn
set_seed(1)

parser = argparse.ArgumentParser(description="CTPN evaluation")
parser.add_argument("--dataset_path", type=str, default="", help="Dataset path.")
parser.add_argument("--image_path", type=str, default="", help="Image path.")
parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint file path.")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
args_opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)

def ctpn_infer_test(dataset_path='', ckpt_path='', img_dir=''):
    """ctpn infer."""
    print("ckpt path is {}".format(ckpt_path))
    ds = create_ctpn_dataset(dataset_path, batch_size=config.test_batch_size, repeat_num=1, is_training=False)
    total = ds.get_dataset_size()
    print("eval dataset size is {}".format(total))
    net = CTPN(config, batch_size=config.test_batch_size, is_training=False)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    eval_for_ctpn(net, ds, img_dir)

if __name__ == '__main__':
    ctpn_infer_test(args_opt.dataset_path, args_opt.checkpoint_path, img_dir=args_opt.image_path)
